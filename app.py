from flask import Flask, request, jsonify
from chat_service import add_super_chat_message
from context_builder import build_context
from llm import call_llm

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data["user_id"]
    message = data["message"]
    deepdive_id = data.get("deepdive_id")

    add_super_chat_message(user_id, "user", message)

    context = build_context(user_id, message, deepdive_id)
    context.append({"role": "user", "content": message})

    reply = call_llm(context)

    add_super_chat_message(user_id, "assistant", reply)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)