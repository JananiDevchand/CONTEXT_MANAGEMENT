# context_builder.py

from hybrid_retriever import HybridRetriever
from redis_stm import search_stm, store_stm
from llm_evaluator import evaluate_search
from file_rag import build_file_rag_context
from semantic_retriever import retrieve_semantic_memory

from context_deduplicator import remove_redundant_context
from context_reranker import rerank_context
from context_compressor import compress_context


MIN_RELEVANCE_SCORE = 0.60
MIN_CONFIDENCE_SCORE = 0.60


def build_context(user_id, user_input, deepdive_id=None, deduplicated_context=None):
    """
    FINAL CONTEXT PIPELINE (WITH SEMANTIC MEMORY)

    STM
      ↓ miss
    Semantic Memory
      ↓
    File Memory
      ↓
    Episodic Memory
      ↓
    Deduplication (bi-encoder)
      ↓
    Re-ranking (cross-encoder)
      ↓
    Compression
      ↓
    Evaluation
      ↓
    STM cache
      ↓
    LLM
    
    Args:
        user_id: User identifier
        user_input: User query
        deepdive_id: Optional deepdive conversation ID
        deduplicated_context: Optional pre-deduplicated context to skip retrieval step
    """

    # If deduplicated context is provided, skip retrieval and dedup steps
    if deduplicated_context is not None:
        print("\n⏭️  Skipping retrieval & dedup (using pre-deduplicated context)")
        combined_context = deduplicated_context
    else:
        # 1️⃣ Short-Term Memory (Redis)
        stm_context = search_stm(user_id, user_input)
        if stm_context:
            print("⚡ Using STM context")
            return stm_context

        combined_context = []

        # 2️⃣ Semantic Memory (facts, preferences, skills)
        semantic_items = retrieve_semantic_memory(user_id, user_input)
        if semantic_items:
            print("🧠 Semantic memory added")

            for s in semantic_items:
                print(
                    f"SEMANTIC → "
                    f"vector={s['vector_score']:.3f}, "
                    f"bm25={s['bm25_score']:.3f}, "
                    f"final={s['final_score']:.3f}"
                )

                combined_context.append({
                    "role": "system",
                    "content": (
                        f"SEMANTIC MEMORY:\n"
                        f"Subject: {s['subject']}\n"
                        f"Content: {s['content']}\n"
                        f"Confidence: {s.get('confidence_score', 'N/A')}"
                    ),
                    "metadata": {
                        "semantic_id": s["id"],
                        "vector_score": s["vector_score"],
                        "bm25_score": s["bm25_score"],
                        "final_score": s["final_score"]
                    }
                })


        # 3️⃣ File Memory (Markdown / RAG)
        file_context = build_file_rag_context(user_id, user_input)
        if file_context:
            print("📄 File memory added")
            combined_context.extend(file_context)

        # 4️⃣ Episodic Memory (always attempted)
        retriever = HybridRetriever()
        retriever.load(user_id, deepdive_id)
        episodic_results = retriever.search(user_input)

        if episodic_results:
            print("🧠 Episodic memory added")
            for r in episodic_results:
                ep = r["episode"]
                text = "\n".join(
                    f"{m['role']}: {m['content']}"
                    for m in ep["messages"]
                )
                combined_context.append({
                    "role": "system",
                    "content": f"PAST EPISODE:\n{text}",
                    "metadata": r
                })

        if not combined_context:
            print("🔍 No context found at all")
            return []

        # 🔎 PRINT RAW RETRIEVED CONTEXT
        print("\n📥 ORIGINAL CONTEXT RETRIEVED FROM MEMORY")
        for i, ctx in enumerate(combined_context, 1):
            print(f"\n[RAW CONTEXT {i}]")
            meta = ctx.get("metadata", {})

            if "episode" in meta:
                ep = meta["episode"]
                print("Source      : episodic_memory")
                print(f"Episode ID  : {ep.get('id')}")
                print(f"Vector Score: {meta.get('vector_score', 0):.4f}")
                print(f"BM25 Score  : {meta.get('bm25_score', 0):.4f}")

            elif "semantic_id" in meta:
                print("Source      : semantic_memory")

            else:
                print("Source      : file_memory")

            print("CONTENT:")
            print(ctx["content"])
        print("─" * 60)

        # 5️⃣ Deduplication (bi-encoder)
        combined_context, dedup_debug = remove_redundant_context(combined_context)

        print("\n🧹 DEDUPLICATION")
        for i, d in enumerate(dedup_debug, 1):
            print(f"{i}. {d['action']} (similarity={d['score']:.3f})")

        # ⭐ RETURN CONTEXT AFTER DEDUP FOR MODEL SELECTION
        deduplicated_context = combined_context

    # 6️⃣ Re-ranking (cross-encoder)
    combined_context = rerank_context(
        query=user_input,
        contexts=combined_context,
        top_k=5
    )

    print("\n🏆 RE-RANKING")
    for i, c in enumerate(combined_context, 1):
        print(f"{i}. score={c.get('rerank_score', 0.0):.3f}")

    # 7️⃣ Compression
    final_context, _ = compress_context(combined_context)

    print("\n🧠 FINAL COMPRESSED CONTEXT")
    print(final_context[0]["content"])
    print("─" * 60)

    # 8️⃣ LLM-based evaluation
    evaluation = evaluate_search(
        query=user_input,
        context=final_context[0]["content"]
    )

    relevance = evaluation.get("relevance_score", 0.0)
    confidence = evaluation.get("confidence_score", 0.0)

    print(
        f"\n🧪 Evaluation → "
        f"relevance={relevance:.2f}, "
        f"confidence={confidence:.2f}"
    )

    

    # 9️⃣ Cache ONLY final compressed context
    store_stm(user_id, user_input, final_context)
    print("💾 Final context stored in STM")

    return final_context


def build_context_until_dedup(user_id, user_input, deepdive_id=None):
    """
    CONTEXT PIPELINE UP TO DEDUPLICATION (for model selection)
    Returns context after dedup but before reranking.
    """

    # 1️⃣ Short-Term Memory (Redis)
    stm_context = search_stm(user_id, user_input)
    if stm_context:
        print("⚡ Using STM context")
        return stm_context

    combined_context = []

    # 2️⃣ Semantic Memory (facts, preferences, skills)
    semantic_items = retrieve_semantic_memory(user_id, user_input)
    if semantic_items:
        print("🧠 Semantic memory added")

        for s in semantic_items:
            print(
                f"SEMANTIC → "
                f"vector={s['vector_score']:.3f}, "
                f"bm25={s['bm25_score']:.3f}, "
                f"final={s['final_score']:.3f}"
            )

            combined_context.append({
                "role": "system",
                "content": (
                    f"SEMANTIC MEMORY:\n"
                    f"Subject: {s['subject']}\n"
                    f"Content: {s['content']}\n"
                    f"Confidence: {s.get('confidence_score', 'N/A')}"
                ),
                "metadata": {
                    "semantic_id": s["id"],
                    "vector_score": s["vector_score"],
                    "bm25_score": s["bm25_score"],
                    "final_score": s["final_score"]
                }
            })


    # 3️⃣ File Memory (Markdown / RAG)
    file_context = build_file_rag_context(user_id, user_input)
    if file_context:
        print("📄 File memory added")
        combined_context.extend(file_context)

    # 4️⃣ Episodic Memory (always attempted)
    retriever = HybridRetriever()
    retriever.load(user_id, deepdive_id)
    episodic_results = retriever.search(user_input)

    if episodic_results:
        print("🧠 Episodic memory added")
        for r in episodic_results:
            ep = r["episode"]
            text = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in ep["messages"]
            )
            combined_context.append({
                "role": "system",
                "content": f"PAST EPISODE:\n{text}",
                "metadata": r
            })

    if not combined_context:
        print("🔍 No context found at all")
        return []

    # 🔎 PRINT RAW RETRIEVED CONTEXT
    print("\n📥 ORIGINAL CONTEXT RETRIEVED FROM MEMORY")
    for i, ctx in enumerate(combined_context, 1):
        print(f"\n[RAW CONTEXT {i}]")
        meta = ctx.get("metadata", {})

        if "episode" in meta:
            ep = meta["episode"]
            print("Source      : episodic_memory")
            print(f"Episode ID  : {ep.get('id')}")
            print(f"Vector Score: {meta.get('vector_score', 0):.4f}")
            print(f"BM25 Score  : {meta.get('bm25_score', 0):.4f}")

        elif "semantic_id" in meta:
            print("Source      : semantic_memory")

        else:
            print("Source      : file_memory")

        print("CONTENT:")
        print(ctx["content"])
    print("─" * 60)

    # 5️⃣ Deduplication (bi-encoder)
    deduplicated_context, dedup_debug = remove_redundant_context(combined_context)

    print("\n🧹 DEDUPLICATION")
    for i, d in enumerate(dedup_debug, 1):
        print(f"{i}. {d['action']} (similarity={d['score']:.3f})")

    return deduplicated_context
