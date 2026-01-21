from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_context(query, contexts, top_k=5):
    pairs = [
        (query, c["content"])
        for c in contexts
    ]

    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(contexts, scores),
        key=lambda x: x[1],
        reverse=True
    )

    ranked_contexts = [
        {**ctx, "rerank_score": float(score)}
        for ctx, score in ranked[:top_k]
    ]

    print("🏆 Re-ranking scores:",
          [round(c["rerank_score"], 3) for c in ranked_contexts])

    return ranked_contexts
