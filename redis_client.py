# redis_client.py
import os
from redis import Redis
from dotenv import load_dotenv

load_dotenv()

def get_redis():
    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT")
    password = os.getenv("REDIS_PASSWORD")

    if not host or not port or not password:
        raise RuntimeError("❌ Missing Redis env vars")

    return Redis(
        host=host,
        port=int(port),
        username="default",
        password=password,
        decode_responses=False  # REQUIRED for vectors
    )
