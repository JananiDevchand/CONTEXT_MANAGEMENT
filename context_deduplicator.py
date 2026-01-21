from embeddings import EmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity

embedder = EmbeddingModel()


def remove_redundant_context(contexts, threshold=0.60):
    if len(contexts) <= 1:
        return contexts, []

    embeddings = [embedder.encode(c["content"]) for c in contexts]

    kept = []
    kept_embs = []
    debug = []

    for ctx, emb in zip(contexts, embeddings):
        if not kept_embs:
            kept.append(ctx)
            kept_embs.append(emb)
            debug.append({"action": "kept", "score": 1.0})
            continue

        sims = cosine_similarity([emb], kept_embs)[0]
        max_sim = float(max(sims))

        if max_sim < threshold:
            kept.append(ctx)
            kept_embs.append(emb)
            debug.append({"action": "kept", "score": max_sim})
        else:
            debug.append({"action": "dropped", "score": max_sim})

    return kept, debug
