import os
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import OpenAI
from app2 import receive_message

# Load environment variables
load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GRAPH_API_BASE = "https://graph.facebook.com/v21.0"

# Create MCP server
mcp = FastMCP("WhatsApp LLM Tool")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Simple in-memory chat history
user_history = {}


@mcp.tool()
def send_whatsapp_message(to: str, text: str) -> str:
    """Send a text message via WhatsApp Cloud API."""
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        raise ValueError("Missing ACCESS_TOKEN or PHONE_NUMBER_ID in environment")

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

    response = requests.post(url, headers=headers, json=payload, timeout=15)
    if response.status_code >= 400:
        raise Exception(f"WhatsApp API Error: {response.status_code} {response.text}")

    return f"Message sent to {to} ✅"


@mcp.tool()
def generate_reply(user_id: str, text: str) -> str:
    """Generate an AI reply using OpenAI chat model."""
    history = user_history.get(user_id, [])
    messages = [{"role": "system", "content": "You are a helpful WhatsApp assistant."}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5,
        max_tokens=300,
    )

    reply = response.choices[0].message.content.strip()

    # Update memory
    history.extend([
        {"role": "user", "content": text},
        {"role": "assistant", "content": reply},
    ])
    user_history[user_id] = history[-20:]

    return reply


@mcp.tool()
def reply_whatsapp_message() :
   receive_message()

if __name__=="__main__": 
         mcp.run(transport="streamable-http",
                    host="127.0.0.1",
                    port=8004,
                    
                    )
