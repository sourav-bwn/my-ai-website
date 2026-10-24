from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Get API key from Render environment
API_KEY = os.getenv("API_KEY")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
# Never let a slow OpenAI response pin a gunicorn worker forever.
OPENAI_TIMEOUT = (5, 30)  # (connect, read) seconds
MAX_MESSAGE_CHARS = 4000


def validate_message(data):
    """Return (message, error). error is None when the input is usable."""
    if not isinstance(data, dict):
        return None, "Request body must be JSON with a 'message' field"
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return None, "Field 'message' must be a non-empty string"
    message = message.strip()
    if len(message) > MAX_MESSAGE_CHARS:
        return None, f"Message too long (max {MAX_MESSAGE_CHARS} characters)"
    return message, None


def call_openai(message):
    """One chat completion. Raises requests exceptions on transport failure."""
    response = requests.post(
        OPENAI_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": message}
            ]
        },
        timeout=OPENAI_TIMEOUT,
    )
    return response.json()


@app.route("/")
def home():
    return "AI Backend is running 🚀"


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    message, error = validate_message(data)
    if error:
        # Was a 200 "Server Error: 'NoneType'..." before - clients could not
        # tell a bad request from a real reply.
        return jsonify({"reply": f"Bad request: {error}"}), 400

    if not API_KEY:
        return jsonify({"reply": "Error: API key missing"}), 500

    try:
        result = call_openai(message)
    except requests.Timeout:
        return jsonify({"reply": "Error: the AI service timed out, please try again"}), 504
    except requests.RequestException:
        return jsonify({"reply": "Error: could not reach the AI service"}), 502

    # If API returns error (bad key, quota, model issue)
    if "error" in result:
        return jsonify({"reply": "API Error: " + str(result["error"])}), 502

    try:
        reply = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return jsonify({"reply": "Error: unexpected response from the AI service"}), 502

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
