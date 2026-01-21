# semantic_retriever.py

import json
from embeddings import EmbeddingModel
from bm25_index import BM25Index
from db import get_conn

embedder = EmbeddingModel()


def _semantic_text(row):
    """
    Text used for BM25.
    We flatten subject + content safely.
    """
    content = row["content"]
    if isinstance(content, dict):
        content = json.dumps(content)

    return f"{row['subject']} {content}"


def retrieve_semantic_memory(user_id, query, k=5):
    """
    Hybrid semantic memory retrieval:
    - Vector similarity on content_embedding
    - BM25 on subject + content
    """

    qvec = embedder.encode(query)

    # 1️⃣ Vector search (candidate generation)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                id,
                type,
                subject,
                content,
                confidence_score,
                source_type,
                1 - (content_embedding <=> %s::vector) AS vector_score
            FROM semantic_knowledge
            WHERE user_id = %s
              AND content_embedding IS NOT NULL
            ORDER BY content_embedding <=> %s::vector
            LIMIT %s
        """, (qvec.tolist(), user_id, qvec.tolist(), k))

        rows = cur.fetchall()

    if not rows:
        return []

    # 2️⃣ BM25 on same semantic items
    bm25 = BM25Index()
    semantic_map = {}

    for r in rows:
        semantic_map[r["id"]] = r
        bm25.add(r["id"], _semantic_text(r))

    bm25_scores = bm25.search(query)
    max_bm25 = max(bm25_scores.values(), default=1.0)

    # 3️⃣ Score fusion
    results = []

    for sid, r in semantic_map.items():
        vector_score = float(r["vector_score"])

        bm25_raw = bm25_scores.get(sid, 0.0)
        bm25_norm = bm25_raw / max_bm25 if max_bm25 > 0 else 0.0

        final_score = (
            0.7 * vector_score +
            0.3 * bm25_norm
        )

        results.append({
            "id": r["id"],
            "type": r["type"],
            "subject": r["subject"],
            "content": r["content"],
            "confidence_score": r["confidence_score"],
            "source_type": r["source_type"],
            "vector_score": vector_score,
            "bm25_score": bm25_norm,
            "final_score": final_score
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    return results[:k]
