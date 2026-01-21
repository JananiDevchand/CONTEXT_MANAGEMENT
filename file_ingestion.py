from db import get_conn
from embeddings import EmbeddingModel
from markdown_utils import count_tokens, chunk_markdown
from file_summarizer import summarize_markdown

embedder = EmbeddingModel()
TOKEN_THRESHOLD = 100


def ingest_markdown(user_id: str, filename: str, markdown: str):
    if count_tokens(markdown) <= TOKEN_THRESHOLD:
        return None

    summary = summarize_markdown(markdown)

    summary_vec = embedder.encode(summary)

    with get_conn() as conn, conn.cursor() as cur:
        # insert document
        cur.execute("""
            INSERT INTO user_files (
                user_id, filename,
                markdown_content,
                summary_markdown,
                summary_embedding
            )
            VALUES (%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            user_id,
            filename,
            markdown,
            summary,
            summary_vec.tolist()
        ))

        file_id = cur.fetchone()["id"]

        # chunk + embed
        chunks = chunk_markdown(markdown)
        for idx, chunk in enumerate(chunks):
            vec = embedder.encode(chunk)
            cur.execute("""
                INSERT INTO user_file_chunks (
                    file_id,
                    chunk_index,
                    chunk_markdown,
                    chunk_embedding
                )
                VALUES (%s,%s,%s,%s)
            """, (
                file_id,
                idx,
                chunk,
                vec.tolist()
            ))

        conn.commit()

    return file_id
