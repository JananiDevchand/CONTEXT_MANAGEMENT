import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(messages, model="openai/gpt-oss-120b"):
    res = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return res.choices[0].message.content