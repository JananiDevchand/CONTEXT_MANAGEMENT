from db import get_conn
from embeddings import EmbeddingModel
import json

embedder = EmbeddingModel()

def store_semantic_knowledge(user_id, tenant_id, knowledge_items, source_episode_id):
    with get_conn() as conn, conn.cursor() as cur:
        for k in knowledge_items:
            text = f"{k['subject']} {json.dumps(k['content'])}"
            vec = embedder.encode(text)

            cur.execute("""
            INSERT INTO semantic_knowledge (
                user_id, tenant_id, type, subject, content,
                content_embedding, confidence_score,
                source_type, source_refs
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                user_id,
                tenant_id,
                k["type"],
                k["subject"],
                json.dumps(k["content"]),              # ✅ FIX
                vec.tolist(),
                k.get("confidence_score", 0.7),
                k.get("source_type"),
                json.dumps({"episode_id": source_episode_id})  # ✅ FIX
            ))

        conn.commit()
