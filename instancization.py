import json
from db import get_conn

def instancize_old_episodes():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT * FROM episodes
            WHERE created_at < NOW() - INTERVAL '30 days'
        """)

        for ep in cur.fetchall():
            cur.execute("""
                INSERT INTO instances
                (user_id, tenant_id, source_type, source_id,
                 original_episode_id, messages,
                 message_count, date_from, date_to)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                ep["user_id"],
                ep["tenant_id"],
                ep["source_type"],
                ep["source_id"],
                ep["id"],
                json.dumps(ep["messages"]),
                ep["message_count"],
                ep["date_from"],
                ep["date_to"]
            ))
            cur.execute("DELETE FROM episodes WHERE id=%s", (ep["id"],))

        conn.commit()