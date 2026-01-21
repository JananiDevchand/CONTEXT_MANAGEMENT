import json
from llm import call_llm

SYSTEM_PROMPT = """
You are a strict JSON generator.

Return ONLY valid JSON.
Do not include explanations, markdown, or text outside JSON.

Schema:
{
  "persona": {
    "preferences": {},
    "communication_style": {},
    "behavior_patterns": {}
  },
  "semantic_knowledge": [
    {
      "type": "knowledge|entity|process|skill",
      "subject": "",
      "content": {},
      "confidence_score": 0.0,
      "source_type": "user_stated|inferred"
    }
  ]
}

If nothing can be extracted, return:
{
  "persona": {},
  "semantic_knowledge": []
}
"""

def extract_semantics(messages):
    text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages
        if isinstance(m, dict)
    )

    response = call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ])

    if not response or not response.strip():
        return {"persona": {}, "semantic_knowledge": []}

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("⚠️ LLM returned invalid JSON, skipping semantic extraction")
        return {"persona": {}, "semantic_knowledge": []}
