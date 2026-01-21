import os
import json
import psycopg2
from groq import Groq
from dotenv import load_dotenv

# ===================== ENV ===================== #

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set")

if not DB_URL:
    raise EnvironmentError("DATABASE_URL not set")

client = Groq(api_key=GROQ_API_KEY)

SELECTOR_MODEL = "openai/gpt-oss-120b"

# ===================== DB ===================== #

def get_db_conn():
    return psycopg2.connect(DB_URL)

def fetch_enabled_models():
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT model_id, provider, tier, max_tokens,
               cost_per_1k_tokens, avg_latency_ms,
               supports_coding, supports_agents,
               supports_reasoning, supports_multimodal
        FROM models
        WHERE enabled = TRUE
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# ===================== PROMPT BUILDING ===================== #

def build_model_context(models):
    """Build context string from available models"""
    lines = []
    for m in models:
        lines.append(
            f"- {m[0]} | provider={m[1]}, tier={m[2]}, "
            f"cost=${m[4]}/1k, latency={m[5]}ms, "
            f"coding={m[6]}, agents={m[7]}, "
            f"reasoning={m[8]}, multimodal={m[9]}"
        )
    return "\n".join(lines)

def build_retrieved_context_summary(retrieved_context):
    """Build a summary of retrieved context for the selector"""
    if not retrieved_context:
        return "No retrieved context available."
    
    summary_lines = [f"Total retrieved items: {len(retrieved_context)}"]
    
    for i, ctx in enumerate(retrieved_context[:5], 1):  # Use first 5 items for summary
        content = ctx.get("content", "")[:100]  # Limit to 100 chars
        source = ctx.get("source", "unknown")
        summary_lines.append(f"{i}. [{source}] {content}...")
    
    if len(retrieved_context) > 5:
        summary_lines.append(f"... and {len(retrieved_context) - 5} more items")
    
    return "\n".join(summary_lines)

SYSTEM_PROMPT = """
You are an expert AI model routing engine.

Steps:
1. Understand the user query
2. Analyze the retrieved context to determine complexity and scope
3. Determine the task type
4. Select the BEST model from the provided list

Task types:
- coding: Code generation, debugging, optimization
- agentic: Multi-step workflows, tool usage
- deep_reasoning: Complex analysis, problem-solving
- fast_response: Simple queries, general knowledge
- multimodal: Image or multi-format content
- general: Default general-purpose tasks

Rules:
- Consider context richness: more/better context may allow cheaper models
- Prefer cheapest capable model for the combined query + context
- Avoid overpowered models
- Use reasoning models only if query complexity requires it
- Use agent-capable models only for multi-step workflows
- If context is extensive and detailed, simpler models may suffice

STRICT:
- Output ONLY valid JSON
- Model MUST match exactly from list

JSON format:
{
  "model": "<model_id>",
  "task_type": "<task_type>",
  "reason": "<clear justification explaining both query needs and context contribution>"
}
"""

# ===================== ROUTER ===================== #

def select_best_model(user_query: str, retrieved_context=None) -> dict:
    """
    Select the best model based on user query and retrieved context.
    
    Args:
        user_query: The user's input query
        retrieved_context: List of dicts with 'content', 'source' keys (optional)
    
    Returns:
        Dict with 'model', 'task_type', and 'reason' keys
    """
    models = fetch_enabled_models()
    if not models:
        return {
            "model": "gpt-4.1-mini",
            "task_type": "general",
            "reason": "No enabled models in database"
        }

    model_context = build_model_context(models)
    model_ids = [m[0] for m in models]
    
    # Build context summary
    context_summary = build_retrieved_context_summary(retrieved_context)

    response = client.chat.completions.create(
        model=SELECTOR_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
User query:
{user_query}

Retrieved context:
{context_summary}

Available models:
{model_context}
"""
            }
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        chosen_model = parsed.get("model")

        if chosen_model not in model_ids:
            raise ValueError("Model not allowed")

        return {
            "model": chosen_model,
            "task_type": parsed.get("task_type", "unknown"),
            "reason": parsed.get("reason", "")
        }

    except Exception as e:
        return {
            "model": "gpt-4.1-mini",
            "task_type": "general",
            "reason": f"Fallback due to invalid selector output: {str(e)}"
        }

# ===================== CLI ===================== #

if __name__ == "__main__":
    query = input("Enter your query: ").strip()
    
    # Simulate retrieved context
    sample_context = [
        {"content": "This is sample context about the topic", "source": "document_1"},
        {"content": "Additional relevant information", "source": "episodic_memory"},
    ]
    
    decision = select_best_model(query, retrieved_context=sample_context)

    print("\nSelected Model:", decision["model"])
    print("Task Type:", decision["task_type"])
    print("Reason:", decision["reason"])
