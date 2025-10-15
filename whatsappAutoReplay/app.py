import os
import logging
import threading
from typing import Dict, List

from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import hmac
import hashlib

try:
    # OpenAI SDK v1.x
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# Load environment variables from .env
load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
APP_SECRET = os.getenv("APP_SECRET", "")

GRAPH_API_BASE = "https://graph.facebook.com/v21.0"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("whatsapp-llm-bot")


app = Flask(__name__)


# Simple in-memory chat history per user
user_history_lock = threading.Lock()
user_to_messages: Dict[str, List[Dict[str, str]]] = {}


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "***" + value
    return "*" * (len(value) - 4) + value[-4:]


def get_openai_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available. Ensure 'openai' is installed.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")
    return OpenAI(api_key=OPENAI_API_KEY)


@app.before_request
def log_request():
    try:
        logger.info("REQ %s %s args=%s", request.method, request.path, dict(request.args))
    except Exception:
        logger.exception("Failed to log request")


def llm_generate_reply(user_id: str, user_text: str) -> str:
    """Send the user's message (with minimal memory) to the LLM and return reply text."""
    # Prepare conversation context
    with user_history_lock:
        history = user_to_messages.get(user_id, [])

    system_prompt = (
        "You are a helpful WhatsApp assistant. Answer briefly, clearly, and be polite."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    # Include the last up to 6 turns (12 messages) to reduce context size
    if history:
        messages.extend(history[-12:])
    messages.append({"role": "user", "content": user_text})

    client = get_openai_client()
    # Use a small, fast model suitable for auto-replies
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.5,
        max_tokens=300,
    )

    reply_text = response.choices[0].message.content or ""

    # Persist to in-memory history
    with user_history_lock:
        prior = user_to_messages.get(user_id, [])
        prior.append({"role": "user", "content": user_text})
        prior.append({"role": "assistant", "content": reply_text})
        # Cap history to a reasonable size
        user_to_messages[user_id] = prior[-50:]

    return reply_text.strip()


def send_whatsapp_text(to_number: str, text: str, phone_number_id: str) -> requests.Response:
    """Send a text message via WhatsApp Cloud API."""
    if not ACCESS_TOKEN:
        raise RuntimeError("ACCESS_TOKEN is not set in environment variables.")
    if not phone_number_id:
        raise RuntimeError("PHONE_NUMBER_ID missing in webhook payload or environment.")

    url = f"{GRAPH_API_BASE}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {"body": text[:4096]},  # WhatsApp text limit safeguard
    }
    logger.info("Sending WhatsApp reply to %s via %s", to_number, url)
    params = None
    if APP_SECRET:
        proof = hmac.new(APP_SECRET.encode("utf-8"), ACCESS_TOKEN.encode("utf-8"), hashlib.sha256).hexdigest()
        params = {"appsecret_proof": proof}
    resp = requests.post(url, headers=headers, json=payload, params=params, timeout=20)
    logger.info("WhatsApp API status=%s response=%s", resp.status_code, resp.text)
    resp.raise_for_status()
    return resp


@app.route("/webhook", methods=["GET"])
def webhook_verify():
    """Meta Webhook verification endpoint."""
    mode = request.args.get("hub.mode")
    verify_token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    logger.info(
        "Webhook verify attempt: mode=%s provided_token=%s expected_token=%s match=%s",
        mode,
        _mask_secret(verify_token or ""),
        _mask_secret(VERIFY_TOKEN or ""),
        bool(verify_token == VERIFY_TOKEN),
    )

    if mode == "subscribe" and verify_token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return challenge, 200

    logger.warning("Webhook verification failed: mode=%s token_ok=%s", mode, verify_token == VERIFY_TOKEN)
    return "forbidden", 403


def extract_message_fields(body_json: dict) -> Dict[str, str]:
    """Extract sender, text (best-effort), and phone_number_id from the webhook payload.

    Returns keys: from, text, phone_number_id
    """
    try:
        entry = (body_json.get("entry") or [])[0]
        changes = (entry.get("changes") or [])[0]
        value = changes.get("value") or {}
        messages = value.get("messages") or []
        if not messages:
            logger.info("No 'messages' array in payload value; skipping")
            return {"from": "", "text": "", "phone_number_id": PHONE_NUMBER_ID}

        message = messages[0]
        sender = message.get("from") or ""
        msg_type = message.get("type") or ""

        # Best-effort extraction of text across common message types
        text = ""
        if msg_type == "text":
            text = (message.get("text") or {}).get("body") or ""
        elif msg_type == "button":
            text = (message.get("button") or {}).get("text") or (message.get("button") or {}).get("payload") or ""
        elif msg_type == "interactive":
            interactive = message.get("interactive") or {}
            button_reply = interactive.get("button_reply") or {}
            list_reply = interactive.get("list_reply") or {}
            text = (
                button_reply.get("title")
                or button_reply.get("id")
                or list_reply.get("title")
                or list_reply.get("id")
                or ""
            )
        elif msg_type == "image":
            text = (message.get("image") or {}).get("caption") or "[image]"
        elif msg_type == "audio":
            text = "[audio]"
        elif msg_type == "video":
            text = (message.get("video") or {}).get("caption") or "[video]"
        elif msg_type == "document":
            text = (message.get("document") or {}).get("caption") or "[document]"
        elif msg_type == "sticker":
            text = "[sticker]"
        else:
            # Unknown or unsupported type; keep empty to skip
            logger.info("Unsupported message type: %s", msg_type)

        phone_number_id = (value.get("metadata") or {}).get("phone_number_id") or PHONE_NUMBER_ID
        return {"from": sender, "text": text, "phone_number_id": phone_number_id}
    except Exception as exc:
        logger.exception("Failed to parse webhook payload: %s", exc)
        return {"from": "", "text": "", "phone_number_id": PHONE_NUMBER_ID}


@app.route("/webhook", methods=["POST"])
def webhook_receive():
    """Receive incoming WhatsApp messages and auto-reply via LLM."""
    try:
        # Optional: verify X-Hub-Signature when APP_SECRET is set
        if APP_SECRET:
            signature = request.headers.get("X-Hub-Signature-256") or request.headers.get("X-Hub-Signature")
            raw_body = request.get_data(cache=False)
            if not signature:
                logger.warning("Missing signature header while APP_SECRET configured")
                return "ok", 200
            algo, _, sig_value = signature.partition("=")
            algo = (algo or "sha1").lower()
            digestmod = hashlib.sha256 if algo == "sha256" else hashlib.sha1
            mac = hmac.new(APP_SECRET.encode("utf-8"), msg=raw_body, digestmod=digestmod)
            expected = mac.hexdigest()
            if not hmac.compare_digest(expected, sig_value):
                logger.warning("Invalid webhook signature")
                return "ok", 200
        body_json = request.get_json(force=True, silent=True) or {}
        logger.info("Webhook POST payload: %s", body_json)

        fields = extract_message_fields(body_json)
        sender = fields.get("from", "")
        text = fields.get("text", "")
        phone_number_id = fields.get("phone_number_id", PHONE_NUMBER_ID)

        # Ignore non-message notifications
        if not sender or not text:
            logger.info("No text message found or unsupported payload; returning ok")
            return "ok", 200

        logger.info("Received message from %s (type may vary): %s", sender, text)

        # Generate LLM reply
        reply_text = llm_generate_reply(sender, text)
        logger.info("Generated reply for %s: %s", sender, reply_text)

        # Send reply via WhatsApp Cloud API
        logger.info(
            "Attempting to send reply. to=%s phone_number_id=%s text_len=%s",
            sender,
            phone_number_id,
            len(reply_text),
        )
        send_whatsapp_text(sender, reply_text, phone_number_id)

    except requests.HTTPError as http_err:
        logger.exception("WhatsApp API HTTP error: %s", http_err)
    except Exception as exc:
        logger.exception("Error handling webhook: %s", exc)

    # Always 200 OK to acknowledge receipt to Meta
    return "ok", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/debug/env", methods=["GET"])
def debug_env():
    # Expose masked values for debugging only; remove for production
    return (
        jsonify(
            {
                "ACCESS_TOKEN_set": bool(ACCESS_TOKEN),
                "PHONE_NUMBER_ID": PHONE_NUMBER_ID,
                "VERIFY_TOKEN_masked": _mask_secret(VERIFY_TOKEN),
                "OPENAI_KEY_set": bool(OPENAI_API_KEY),
            }
        ),
        200,
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info("Starting Flask app on %s:%s", host, port)
    app.run(host=host, port=port, debug=False)


