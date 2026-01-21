# db_setup.py

from db import get_conn
from embeddings import EmbeddingModel

embedder = EmbeddingModel()


def create_tables():
    with get_conn() as conn, conn.cursor() as cur:
        # -------------------------------------------------
        # Extensions
        # -------------------------------------------------
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        # -------------------------------------------------
        # Super Chat
        # -------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS super_chat (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS super_chat_messages (
                id SERIAL PRIMARY KEY,
                super_chat_id INTEGER REFERENCES super_chat(id),
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                episodized BOOLEAN DEFAULT FALSE,
                episodized_at TIMESTAMP
            )
        """)

        # -------------------------------------------------
        # Deep Dive
        # -------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS deepdive_conversations (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                tenant_id TEXT,
                title VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS deepdive_messages (
                id SERIAL PRIMARY KEY,
                deepdive_conversation_id INTEGER REFERENCES deepdive_conversations(id),
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # -------------------------------------------------
        # Episodes (Episodic Memory)
        # -------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                tenant_id TEXT,
                source_type VARCHAR(50) NOT NULL,
                source_id INTEGER NOT NULL,
                messages JSONB NOT NULL,
                message_count INTEGER NOT NULL,
                date_from TIMESTAMP NOT NULL,
                date_to TIMESTAMP NOT NULL,
                vector vector(384),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_vector
            ON episodes USING ivfflat (vector vector_cosine_ops)
            WITH (lists = 100);
        """)

        # -------------------------------------------------
        # Instances (Archived Episodes)
        # -------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS instances (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                tenant_id TEXT,
                source_type VARCHAR(50) NOT NULL,
                source_id INTEGER NOT NULL,
                original_episode_id INTEGER NOT NULL,
                messages JSONB NOT NULL,
                message_count INTEGER NOT NULL,
                date_from TIMESTAMP NOT NULL,
                date_to TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # -------------------------------------------------
        # Semantic Memory — User Persona
        # -------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_persona (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL UNIQUE,
                preferences JSONB NOT NULL DEFAULT '{}',
                communication_style JSONB NOT NULL DEFAULT '{}',
                behavior_patterns JSONB NOT NULL DEFAULT '{}',
                content_embedding vector(384),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_persona_user
            ON user_persona(user_id);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_persona_embedding
            ON user_persona USING ivfflat (content_embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        # -------------------------------------------------
        # Semantic Memory — Knowledge
        # -------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_knowledge (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                tenant_id TEXT,
                type VARCHAR(50) NOT NULL,
                subject VARCHAR(255) NOT NULL,
                content JSONB NOT NULL,
                content_embedding vector(384),
                confidence_score DECIMAL(3,2),
                source_type VARCHAR(50),
                source_refs JSONB,
                verified BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                CONSTRAINT chk_semantic_type
                    CHECK (type IN ('knowledge', 'entity', 'process', 'skill')),

                CONSTRAINT chk_semantic_source
                    CHECK (
                        source_type IS NULL
                        OR source_type IN ('user_stated', 'inferred', 'asset_derived')
                    )
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_know_user
            ON semantic_knowledge(user_id);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_know_type
            ON semantic_knowledge(type);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_know_subject
            ON semantic_knowledge(subject);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_know_embedding
            ON semantic_knowledge USING ivfflat (content_embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        conn.commit()

    print("✅ All tables and indexes created successfully.")

    populate_episode_vectors()


def populate_episode_vectors():
    """
    Backfill missing episode embeddings
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, messages
            FROM episodes
            WHERE vector IS NULL
        """)
        episodes = cur.fetchall()

        updated = 0
        for ep in episodes:
            messages = ep.get("messages")
            if not messages or not isinstance(messages, list):
                continue

            text = " ".join(
                m.get("content", "")
                for m in messages
                if isinstance(m, dict)
            ).strip()

            if not text:
                continue

            vec = embedder.encode(text)
            cur.execute(
                "UPDATE episodes SET vector = %s WHERE id = %s",
                (vec.tolist(), ep["id"])
            )
            updated += 1

        conn.commit()

    print(f"🧠 Populated vectors for {updated} episodes.")


if __name__ == "__main__":
    create_tables()
