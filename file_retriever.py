from db import get_conn
from embeddings import EmbeddingModel

embedder = EmbeddingModel()

def find_relevant_files(user_id: str, query: str, threshold=0.30):
    qvec = embedder.encode(query)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                id,
                filename,
                1 - (summary_embedding <=> %s::vector) AS similarity
            FROM user_files
            WHERE user_id = %s
            ORDER BY summary_embedding <=> %s::vector
            LIMIT 3
        """, (qvec.tolist(), user_id, qvec.tolist()))

        rows = cur.fetchall()

    print("🔎 Summary similarity scores:")
    for r in rows:
        print(r["filename"], round(r["similarity"], 3))

    return [r["id"] for r in rows if r["similarity"] >= threshold]



def retrieve_top_chunks(file_ids, query, k=3):
    qvec = embedder.encode(query)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                chunk_markdown,
                1 - (chunk_embedding <=> %s::vector) AS similarity
            FROM user_file_chunks
            WHERE file_id = ANY(%s::uuid[])
            ORDER BY chunk_embedding <=> %s::vector
            LIMIT %s
        """, (
            qvec.tolist(),
            file_ids,          # list[str] is OK now
            qvec.tolist(),
            k
        ))

        return cur.fetchall()
