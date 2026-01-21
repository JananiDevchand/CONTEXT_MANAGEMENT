# redis_stm_index.py
from redis.commands.search.field import VectorField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis_client import get_redis

r = get_redis()

INDEX = "stm_idx"
DIM = 384

def create_index():
    try:
        r.ft(INDEX).info()
        print("✅ STM index exists")
        return
    except Exception:
        pass

    r.ft(INDEX).create_index(
        fields=[
            VectorField(
                "query_vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": DIM,
                    "DISTANCE_METRIC": "COSINE"
                }
            ),
            NumericField("created_at")
        ],
        definition=IndexDefinition(
            prefix=["stm:"],
            index_type=IndexType.HASH
        )
    )

    print("🚀 STM index created")

if __name__ == "__main__":
    create_index()
