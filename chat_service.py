from db import get_conn
def get_or_create_super_chat(user_id):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM super_chat WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        if row:
            return row["id"]

        cur.execute(
            "INSERT INTO super_chat (user_id) VALUES (%s) RETURNING id",
            (user_id,)
        )
        conn.commit()
        return cur.fetchone()["id"]



def add_super_chat_message(user_id, role, content):
    chat_id = get_or_create_super_chat(user_id)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO super_chat_messages (super_chat_id, role, content)
            VALUES (%s, %s, %s)
        """, (chat_id, role, content))
        conn.commit()

def create_deepdive(user_id, title, tenant_id=None):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO deepdive_conversations (user_id, tenant_id, title)
            VALUES (%s,%s,%s) RETURNING id
        """, (user_id, tenant_id, title))
        conn.commit()
        return cur.fetchone()["id"]

def add_deepdive_message(conversation_id, role, content):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO deepdive_messages
            (deepdive_conversation_id, role, content)
            VALUES (%s,%s,%s)
        """, (conversation_id, role, content))
        conn.commit()