"""缓存模块 — Redis 可选，自动降级为内存缓存。

用法:
    from cache import cache
    cache.set("key", value, ttl=300)
    value = cache.get("key")
"""

import json
import os
import time
import threading
from typing import Optional

# ── Redis 配置 ──
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_PROTOCOL = int(os.getenv("REDIS_PROTOCOL", "3"))
CACHE_TTL_DEFAULT = int(os.getenv("CACHE_TTL", "300"))


class _RedisCache:
    def __init__(self):
        self._client = None
        self._available = False
        self._try_connect()

    def _try_connect(self):
        try:
            import redis
            self._client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                password=REDIS_PASSWORD, protocol=REDIS_PROTOCOL,
                socket_connect_timeout=2, decode_responses=True,
            )
            self._client.ping()
            self._available = True
            print(f"[Cache] Redis connected: {REDIS_HOST}:{REDIS_PORT} (protocol={REDIS_PROTOCOL})")
        except Exception as e:
            self._available = False
            self._client = None
            msg = str(e).split("\n")[0][:80]
            print(f"[Cache] Redis unavailable ({msg}). Fallback: in-memory cache.")

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    def get(self, key: str) -> Optional[str]:
        if not self.available:
            return None
        try:
            return self._client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        if not self.available:
            return False
        try:
            self._client.setex(key, ttl, value)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        if not self.available:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception:
            return False

    def get_json(self, key: str) -> Optional[dict]:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: dict, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        try:
            return self.set(key, json.dumps(value, ensure_ascii=False), ttl)
        except Exception:
            return False


class _MemoryCache:
    def __init__(self):
        self._store: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            exp, val = entry
            if time.time() >= exp:
                del self._store[key]
                return None
            return val

    def set(self, key: str, value: str, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        with self._lock:
            self._store[key] = (time.time() + ttl, value)
        return True

    def delete(self, key: str) -> bool:
        with self._lock:
            self._store.pop(key, None)
        return True

    def get_json(self, key: str) -> Optional[dict]:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: dict, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        try:
            return self.set(key, json.dumps(value, ensure_ascii=False), ttl)
        except Exception:
            return False


class Cache:
    """统一缓存接口。优先级: Redis > 内存。"""

    def __init__(self):
        self._redis = _RedisCache()
        self._memory = _MemoryCache()

    def get(self, key: str) -> Optional[str]:
        val = self._redis.get(key)
        if val is not None:
            return val
        return self._memory.get(key)

    def set(self, key: str, value: str, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        if self._redis.available:
            return self._redis.set(key, value, ttl)
        return self._memory.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        self._redis.delete(key)
        return self._memory.delete(key)

    def get_json(self, key: str) -> Optional[dict]:
        val = self._redis.get_json(key)
        if val is not None:
            return val
        return self._memory.get_json(key)

    def set_json(self, key: str, value: dict, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        if self._redis.available:
            return self._redis.set_json(key, value, ttl)
        return self._memory.set_json(key, value, ttl)

    @property
    def backend(self) -> str:
        return "redis" if self._redis.available else "memory"


cache = Cache()
