from datetime import datetime
from embeddings import EmbeddingModel
from bm25_index import BM25Index
from db import get_conn

embedder = EmbeddingModel()


class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Index()
        self.episodes = []
        self.episode_map = {}

    def load(self, user_id, deepdive_id=None):
        """
        Load episodes from DB and build BM25 index.
        """
        with get_conn() as conn, conn.cursor() as cur:
            if deepdive_id:
                cur.execute("""
                    SELECT *
                    FROM episodes
                    WHERE source_type = 'deepdive'
                      AND source_id = %s
                      AND vector IS NOT NULL
                """, (deepdive_id,))
            else:
                cur.execute("""
                    SELECT *
                    FROM episodes
                    WHERE user_id = %s
                      AND vector IS NOT NULL
                """, (user_id,))

            self.episodes = cur.fetchall()

        # Reset structures
        self.episode_map = {}
        self.bm25 = BM25Index()

        for ep in self.episodes:
            self.episode_map[ep["id"]] = ep
            text = " ".join(m["content"] for m in ep["messages"])
            self.bm25.add(ep["id"], text)

        print(f"📚 Loaded {len(self.episodes)} episodes for retrieval.")

    def search(self, query, k=3, min_score=0.30):
        """
        Hybrid search:
        - Vector similarity (pgvector)
        - BM25 keyword relevance
        - Recency bias
        Returns top-k results with total_score > min_score
        """
        if not self.episodes:
            return []

        # Encode query
        qvec = embedder.encode(query)

        # BM25 scores
        bm25_scores = self.bm25.search(query)
        max_bm25 = max(bm25_scores.values(), default=1.0)

        # Vector similarity search (top-k)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT id,
                       1 - (vector <=> %s::vector) AS similarity
                FROM episodes
                WHERE id = ANY(%s)
                ORDER BY vector <=> %s::vector
                LIMIT %s
            """, (
                qvec.tolist(),
                list(self.episode_map.keys()),
                qvec.tolist(),
                k
            ))
            vector_results = cur.fetchall()

        now = datetime.utcnow()
        scored = []

        for res in vector_results:
            ep_id = res["id"]
            vector_score = float(res["similarity"])

            ep = self.episode_map[ep_id]

            # BM25 normalization
            bm25_raw = bm25_scores.get(ep_id, 0.0)
            bm25_norm = bm25_raw / max_bm25 if max_bm25 > 0 else 0.0

            # Recency score (linear decay over 30 days)
            age_days = (now - ep["created_at"]).days
            recency = max(0.0, 1.0 - age_days / 30.0)

            # Final hybrid score
            total_score = (
                0.6 * vector_score +
                0.3 * bm25_norm +
                0.1 * recency
            )

            scored.append({
                "episode": ep,
                "total_score": total_score,
                "vector_score": vector_score,
                "bm25_score": bm25_norm,
                "recency_score": recency
            })

        # Sort by total score
        scored.sort(key=lambda x: x["total_score"], reverse=True)

        # Threshold filtering
        filtered = [r for r in scored if r["total_score"] > min_score]

        print(
            f"🏆 Returning {len(filtered[:k])}/{k} results "
            f"(threshold={min_score}): "
            f"{[round(r['total_score'], 3) for r in filtered[:k]]}"
        )

        return filtered[:k]
