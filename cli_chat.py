# cli_chat.py

from chat_service import add_super_chat_message
from context_builder import build_context, build_context_until_dedup
from llm import call_llm
from model_selector import select_best_model

from file_ingestion import ingest_markdown
from markdown_utils import count_tokens

USER_ID = "11111111-1111-1111-1111-111111111111"
TOKEN_THRESHOLD = 10


def run_cli():
    print("🧠 Context-Aware Chat (CLI)")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye")
            break

        # 1️⃣ Store user message (always)
        add_super_chat_message(USER_ID, "user", user_input)

        # 2️⃣ Long-term file ingestion (write-time)
        token_count = count_tokens(user_input)
        if token_count > TOKEN_THRESHOLD:
            print(f"📥 Long input detected ({token_count} tokens)")
            file_id = ingest_markdown(
                user_id=USER_ID,
                filename="cli_input.md",
                markdown=user_input
            )
            if file_id:
                print(f"📄 Stored as long-term file memory ({file_id})")

        # 3️⃣ Build context until deduplication (STM → File + Episodic → Dedup)
        context_after_dedup = build_context_until_dedup(USER_ID, user_input)

        # 4️⃣ Model Selection based on query + deduplicated context
        model_selection = select_best_model(user_input, retrieved_context=context_after_dedup)
        selected_model = model_selection["model"]
        
        print(f"\n🤖 Model Selected: {selected_model}")
        print(f"   Task Type: {model_selection['task_type']}")
        print(f"   Reason: {model_selection['reason']}\n")

        # 5️⃣ Build full context with pre-deduplicated context (reranking and compression)
        context = build_context(USER_ID, user_input, deduplicated_context=context_after_dedup)

        # 6️⃣ Print FINAL context that will be sent to LLM
        if context:
            print(f"────────── 🔎 FINAL CONTEXT SENT TO LLM ({len(context)}) ──────────")
            for i, ctx in enumerate(context, 1):
                print(f"[CONTEXT {i}]")
                print(ctx["content"])
                print()
            print("──────────────────────────────────────────────────────────────\n")
        else:
            print("🔍 No relevant context retrieved.\n")

        # 7️⃣ LLM call
        messages = [
            {"role": ctx["role"], "content": ctx["content"]}
            for ctx in context
        ]
        messages.append({
            "role": "user",
            "content": user_input
        })

        response = call_llm(messages)

        # 8️⃣ Store assistant reply
        add_super_chat_message(USER_ID, "assistant", response)

        print("Assistant:", response, "\n")


if __name__ == "__main__":
    run_cli()
