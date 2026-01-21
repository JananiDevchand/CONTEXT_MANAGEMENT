# llm_evaluator.py
import json
from llm import call_llm


EVAL_PROMPT = """
You are a strict evaluator for a search system.

User Query:
{query}

Retrieved Context:
{context}

Evaluate the retrieved context.

Return a JSON object with:
- relevance_score (0 to 1)
- confidence_score (0 to 1)
- short_reason (1 sentence)

Scoring rules:
- relevance_score: semantic match to query
- confidence_score: correctness + completeness + clarity

Return ONLY valid JSON.
"""


def evaluate_search(query: str, context: str):
    prompt = EVAL_PROMPT.format(
        query=query,
        context=context
    )

    response = call_llm([
        {"role": "system", "content": "You evaluate search results."},
        {"role": "user", "content": prompt}
    ])

    try:
        return json.loads(response)
    except Exception:
        return {
            "relevance_score": 0.0,
            "confidence_score": 0.0,
            "short_reason": "Invalid evaluator response"
        }
