# redis_stm.py
import time
import numpy as np
from redis.commands.search.query import Query
from embeddings import EmbeddingModel
from redis_client import get_redis

embedder = EmbeddingModel()
r = get_redis()

INDEX = "stm_idx"
TTL = 300
MAX_ITEMS = 5
SIM_THRESHOLD = 0.90


def _prune(user_id):
    keys = sorted(r.keys(f"stm:{user_id}:*"))
    for k in keys[:-MAX_ITEMS]:
        r.delete(k)


def store_stm(user_id, query, context):
    qvec = embedder.encode(query).astype(np.float32).tobytes()

    key = f"stm:{user_id}:{int(time.time())}"

    r.hset(key, mapping={
        "query": query,
        "query_vector": qvec,
        "context": "\n".join(c["content"] for c in context),
        "created_at": time.time()
    })

    r.expire(key, TTL)
    _prune(user_id)


def search_stm(user_id, query, k=3):
    qvec = embedder.encode(query).astype(np.float32).tobytes()

    q = (
        Query(f"*=>[KNN {k} @query_vector $vec AS score]")
        .sort_by("score")
        .return_fields("context", "score")
        .dialect(2)
    )

    res = r.ft(INDEX).search(q, query_params={"vec": qvec})

    if not res.docs:
        return None

    best = res.docs[0]
    similarity = 1 - float(best.score)

    if similarity < SIM_THRESHOLD:
        return None

    print(f"⚡ STM HIT similarity={round(similarity,3)}")

    return [{
        "role": "system",
        "content": best.context
    }]
