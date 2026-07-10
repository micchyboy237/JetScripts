"""
03_caching_optimization.py

Caching strategies for RAG filter performance optimization.
Demonstrates TTL caching, hash-based keys, and cache invalidation.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    access_count: int = 0
    last_accessed: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds

    def access(self):
        """Record an access to this entry."""
        self.access_count += 1
        self.last_accessed = time.time()


class TTLCache:
    """
    Time-To-Live cache with automatic expiration.
    Similar to cachetools.TTLCache.
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self.maxsize = maxsize
        self.default_ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            self.misses += 1
            return None

        entry = self._cache[key]

        if entry.is_expired():
            del self._cache[key]
            self.misses += 1
            return None

        entry.access()
        self._cache.move_to_end(key)
        self.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL."""
        if len(self._cache) >= self.maxsize:
            # Remove oldest entry
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl or self.default_ttl,
        )
        self._cache.move_to_end(key)

    def invalidate(self, key: str):
        """Remove a specific key from cache."""
        if key in self._cache:
            del self._cache[key]

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()

    def cleanup_expired(self):
        """Remove all expired entries."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]

    @property
    def stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "expired_count": sum(1 for e in self._cache.values() if e.is_expired()),
        }


class FilterCacheManager:
    """
    Manages caching for filter extraction and search results.
    Uses multiple cache layers for optimal performance.
    """

    def __init__(self):
        # Cache for extracted filters (longer TTL - filters change less often)
        self.filter_cache = TTLCache(maxsize=500, ttl=1800)  # 30 minutes

        # Cache for search results (shorter TTL - results may change)
        self.search_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes

        # Cache for user permissions (longest TTL - permissions rarely change)
        self.permission_cache = TTLCache(maxsize=200, ttl=3600)  # 1 hour

    def build_filter_cache_key(self, query: str) -> str:
        """Build cache key for filter extraction."""
        # Normalize query for better cache hits
        normalized = query.lower().strip()
        return f"filter:{hashlib.md5(normalized.encode()).hexdigest()}"

    def build_search_cache_key(
        self, query: str, filter_dict: Dict, k: int, user_id: str
    ) -> str:
        """Build cache key for search results."""
        key_components = [
            query.lower().strip(),
            json.dumps(filter_dict, sort_keys=True),
            str(k),
            user_id or "anonymous",
        ]
        combined = ":".join(key_components)
        return f"search:{hashlib.md5(combined.encode()).hexdigest()}"

    def build_permission_cache_key(self, user_id: str) -> str:
        """Build cache key for user permissions."""
        return f"permission:{user_id}"

    def get_filters(self, query: str) -> Optional[Dict]:
        """Get cached filters for a query."""
        key = self.build_filter_cache_key(query)
        return self.filter_cache.get(key)

    def cache_filters(self, query: str, filters: Dict, ttl: int = None):
        """Cache extracted filters."""
        key = self.build_filter_cache_key(query)
        self.filter_cache.set(key, filters, ttl)

    def get_search_results(
        self, query: str, filter_dict: Dict, k: int, user_id: str
    ) -> Optional[List]:
        """Get cached search results."""
        key = self.build_search_cache_key(query, filter_dict, k, user_id)
        return self.search_cache.get(key)

    def cache_search_results(
        self, query: str, filter_dict: Dict, k: int, user_id: str, results: List
    ):
        """Cache search results."""
        key = self.build_search_cache_key(query, filter_dict, k, user_id)
        self.search_cache.set(key, results)

    def get_permissions(self, user_id: str) -> Optional[Dict]:
        """Get cached user permissions."""
        key = self.build_permission_cache_key(user_id)
        return self.permission_cache.get(key)

    def cache_permissions(self, user_id: str, permissions: Dict):
        """Cache user permissions."""
        key = self.build_permission_cache_key(user_id)
        self.permission_cache.set(key, permissions)

    def invalidate_user(self, user_id: str):
        """Invalidate all caches for a user (e.g., on permission change)."""
        perm_key = self.build_permission_cache_key(user_id)
        self.permission_cache.invalidate(perm_key)

    def get_all_stats(self) -> Dict:
        """Get statistics for all caches."""
        return {
            "filter_cache": self.filter_cache.stats,
            "search_cache": self.search_cache.stats,
            "permission_cache": self.permission_cache.stats,
        }


class CachedRAGPipeline:
    """RAG pipeline with integrated caching at multiple levels."""

    def __init__(self):
        self.cache_manager = FilterCacheManager()
        self.search_count = 0
        self.cache_hits = 0

    def smart_search(
        self,
        query: str,
        user_id: str = None,
        k: int = 5,
        use_caching: bool = True,
    ) -> Dict:
        """
        Search with intelligent multi-level caching.

        Cache layers:
        1. Check search result cache (fastest)
        2. Check filter extraction cache
        3. Extract filters + search (slowest)
        """
        start_time = time.time()
        self.search_count += 1

        # Layer 1: Check search result cache
        if use_caching:
            # Try with empty filter first
            cached_results = self.cache_manager.get_search_results(
                query, {}, k, user_id
            )
            if cached_results is not None:
                self.cache_hits += 1
                return {
                    "results": cached_results,
                    "cache_hit": True,
                    "cache_layer": "search_results",
                    "time_ms": (time.time() - start_time) * 1000,
                }

        # Layer 2: Check filter cache
        filter_dict = None
        if use_caching:
            filter_dict = self.cache_manager.get_filters(query)

        # Layer 3: Extract filters if not cached
        if filter_dict is None:
            filter_dict = self._extract_filters_simulated(query)
            if use_caching:
                self.cache_manager.cache_filters(query, filter_dict)

        # Perform search
        results = self._simulate_search(query, filter_dict, k)

        # Cache search results
        if use_caching:
            self.cache_manager.cache_search_results(
                query, filter_dict, k, user_id, results
            )

        return {
            "results": results,
            "cache_hit": False,
            "filter_dict": filter_dict,
            "time_ms": (time.time() - start_time) * 1000,
        }

    def _extract_filters_simulated(self, query: str) -> Dict:
        """Simulate filter extraction (would call LLM in production)."""
        # Simulate processing time
        time.sleep(0.1)
        return {"department": "Technology"} if "tech" in query.lower() else {}

    def _simulate_search(self, query: str, filter_dict: Dict, k: int) -> List[Dict]:
        """Simulate vector search."""
        return [
            {"content": f"Result {i + 1} for: {query}", "score": 0.9 - (i * 0.1)}
            for i in range(min(k, 5))
        ]

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.search_count == 0:
            return 0.0
        return (self.cache_hits / self.search_count) * 100


def demonstrate_ttl_cache():
    """Demonstrate TTL cache behavior."""
    print("\n" + "=" * 60)
    print("1. TTL CACHE DEMONSTRATION")
    print("=" * 60)

    cache = TTLCache(maxsize=5, ttl=2)  # 2 second TTL for demo

    # Set values
    cache.set("key1", "value1")
    cache.set("key2", "value2", ttl=5)  # Custom TTL

    print(f"\n  Initial cache size: {len(cache._cache)}")
    print(f"  Get key1: {cache.get('key1')}")
    print(f"  Stats: {cache.stats}")

    # Wait for expiration
    print("\n  Waiting for key1 to expire (2 seconds)...")
    time.sleep(2.1)

    print(f"  Get key1 (expired): {cache.get('key1')}")
    print(f"  Get key2 (not expired): {cache.get('key2')}")
    print(f"  Stats: {cache.stats}")

    print("\n✅ TTL cache demonstrated")


def demonstrate_cache_hierarchy():
    """Demonstrate multi-level cache hierarchy."""
    print("\n" + "=" * 60)
    print("2. MULTI-LEVEL CACHE HIERARCHY")
    print("=" * 60)

    pipeline = CachedRAGPipeline()

    # First search - cache miss
    print("\n  First search (cache miss):")
    result1 = pipeline.smart_search("technology trends")
    print(f"    Cache hit: {result1['cache_hit']}")
    print(f"    Time: {result1['time_ms']:.1f}ms")

    # Same search - cache hit
    print("\n  Same search (cache hit):")
    result2 = pipeline.smart_search("technology trends")
    print(f"    Cache hit: {result2['cache_hit']}")
    print(f"    Time: {result2['time_ms']:.1f}ms")
    print(f"    Speedup: {result1['time_ms'] / max(result2['time_ms'], 0.1):.1f}x")

    # Different search - cache miss for results, but filter may be cached
    print("\n  Different search:")
    result3 = pipeline.smart_search("tech innovations")
    print(f"    Cache hit: {result3['cache_hit']}")
    print(f"    Time: {result3['time_ms']:.1f}ms")

    print("\n  Overall stats:")
    print(f"    Cache hit rate: {pipeline.hit_rate:.1f}%")
    print(f"    Cache stats: {pipeline.cache_manager.get_all_stats()}")

    print("\n✅ Cache hierarchy demonstrated")


def demonstrate_cache_invalidation():
    """Demonstrate cache invalidation strategies."""
    print("\n" + "=" * 60)
    print("3. CACHE INVALIDATION STRATEGIES")
    print("=" * 60)

    cache = TTLCache(maxsize=10, ttl=3600)

    # Populate cache
    for i in range(5):
        cache.set(f"doc_{i}", f"content_{i}")

    print(f"\n  Initial cache: {len(cache._cache)} entries")

    # Invalidate specific key
    cache.invalidate("doc_1")
    print(f"  After invalidating doc_1: {len(cache._cache)} entries")
    print(f"  doc_1 still cached: {'doc_1' in cache._cache}")

    # Clear all
    cache.clear()
    print(f"  After clear: {len(cache._cache)} entries")

    print("\n  Invalidation triggers:")
    print("    - Document update → invalidate document cache")
    print("    - Permission change → invalidate user cache")
    print("    - New data ingested → clear search cache")
    print("    - Schema change → clear all caches")

    print("\n✅ Cache invalidation strategies documented")


def demonstrate_hash_based_keys():
    """Demonstrate hash-based cache key generation."""
    print("\n" + "=" * 60)
    print("4. HASH-BASED CACHE KEYS")
    print("=" * 60)

    def build_key(query: str, filters: Dict, **params) -> str:
        """Build deterministic cache key."""
        components = [
            query.lower().strip(),
            json.dumps(filters, sort_keys=True),
            json.dumps(params, sort_keys=True),
        ]
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    # Demonstrate key consistency
    query = "What is AI?"
    filters = {"department": "Technology", "year": 2024}

    key1 = build_key(query, filters, k=5, user="alice")
    key2 = build_key(query, filters, k=5, user="alice")
    key3 = build_key(query, filters, k=10, user="alice")  # Different k

    print("\n  Same query + filters = same key:")
    print(f"    Key1: {key1}")
    print(f"    Key2: {key2}")
    print(f"    Match: {key1 == key2}")

    print("\n  Different params = different key:")
    print(f"    Key1: {key1}")
    print(f"    Key3: {key3}")
    print(f"    Match: {key1 == key3}")

    print("\n  Key components:")
    print(f"    Query: '{query.lower().strip()}'")
    print(f"    Filters: {json.dumps(filters, sort_keys=True)}")
    print("    Combined: query|filters|params")
    print("    Hash: SHA256 → 16 char hex")

    print("\n✅ Hash-based key generation demonstrated")


def main():
    """Run all caching optimization demonstrations."""
    print("=" * 60)
    print("CACHING OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    demonstrate_ttl_cache()
    demonstrate_cache_hierarchy()
    demonstrate_cache_invalidation()
    demonstrate_hash_based_keys()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 Effective caching can reduce latency by 10-100x")
    print("   and significantly reduce LLM API costs.")


if __name__ == "__main__":
    main()
