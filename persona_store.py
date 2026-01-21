from db import get_conn
from embeddings import EmbeddingModel
import json

embedder = EmbeddingModel()

def upsert_user_persona(user_id, persona):
    combined_text = json.dumps(persona)
    vec = embedder.encode(combined_text)

    with get_conn() as conn, conn.cursor() as cur:


        cur.execute("""
        INSERT INTO user_persona
        (user_id, preferences, communication_style, behavior_patterns, content_embedding)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT (user_id)
        DO UPDATE SET
            preferences = user_persona.preferences || EXCLUDED.preferences,
            communication_style = user_persona.communication_style || EXCLUDED.communication_style,
            behavior_patterns = user_persona.behavior_patterns || EXCLUDED.behavior_patterns,
            content_embedding = EXCLUDED.content_embedding,
            updated_at = NOW()
        """, (
            user_id,
            json.dumps(persona.get("preferences", {})),          # ✅ FIX
            json.dumps(persona.get("communication_style", {})),  # ✅ FIX
            json.dumps(persona.get("behavior_patterns", {})),    # ✅ FIX
            vec.tolist()
        ))

      
        conn.commit()
