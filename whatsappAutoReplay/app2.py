import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from openai import OpenAI

# Load environment variables
load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
GRAPH_API_BASE = "https://graph.facebook.com/v21.0"

# Flask app
app = Flask(__name__)

# Chat history (simple in-memory)
user_history = {}

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Verify webhook for Meta"""
    if (
        request.args.get("hub.mode") == "subscribe"
        and request.args.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return request.args.get("hub.challenge"), 200
    return "forbidden", 403


@app.route("/webhook", methods=["POST"])
def receive_message():
    """Handle incoming WhatsApp messages and reply"""
    data = request.get_json()

    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = message["from"]
        msg_type = message["type"]
        text = message.get("text", {}).get("body", "") if msg_type == "text" else ""
    except Exception:
        return "ok", 200  # Ignore other webhook events

    if not text:
        return "ok", 200

    # Generate AI reply
    history = user_history.get(sender, [])
    messages = [{"role": "system", "content": "You are a friendly WhatsApp assistant."}]
    messages.extend(history[-6:])  # keep last few messages
    messages.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5,
        max_tokens=300,
    )

    reply = response.choices[0].message.content.strip()

    # Save conversation
    history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
    user_history[sender] = history[-20:]

    # Send reply via WhatsApp API
    send_whatsapp_message(sender, reply)
    return "ok", 200


def send_whatsapp_message(to, text):
    """Send a message via WhatsApp Cloud API"""
    url = f"{GRAPH_API_BASE}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": text[:4096]},
    }
    requests.post(url, headers=headers, json=payload, timeout=15)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
