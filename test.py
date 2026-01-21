from redis_client import get_redis

r = get_redis()
print(r.ping())
