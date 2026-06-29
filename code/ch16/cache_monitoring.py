#!/usr/bin/env python3

import argparse
import random

"""
Cache Monitoring for LLM Inference (Chapter 16)

Implements monitoring for KV cache, prefix cache, and prompt-embedding cache
as described in Chapter 16. Tracks cache hits, misses, and prefix merging events.

Key metrics:
- Cache hit rate (KV cache, prefix cache)
- Prefix merge events
- Cache memory utilization
- Eviction statistics (LRU/LFU)

Usage:
    from cache_monitoring import CacheMonitor
    
    monitor = CacheMonitor()
    monitor.record_cache_access(cache_type="kv", hit=True)
    monitor.record_prefix_merge(tokens_deduplicated=128)
    monitor.export_metrics()
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from enum import Enum
import threading
from prometheus_client import Counter, Gauge, Histogram, start_http_server


class CacheType(Enum):
    """Types of caches used in LLM inference"""
    KV_CACHE = "kv_cache"
    PREFIX_CACHE = "prefix_cache"
    PROMPT_EMBEDDING_CACHE = "prompt_embedding_cache"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheStats:
    """Statistics for a single cache"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    capacity_bytes: int = 0
    entries: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def utilization(self) -> float:
        """Calculate cache utilization percentage"""
        return (self.size_bytes / self.capacity_bytes * 100) if self.capacity_bytes > 0 else 0.0


@dataclass
class PrefixMergeStats:
    """Statistics for prefix merging/deduplication"""
    merge_events: int = 0
    tokens_deduplicated: int = 0
    compute_saved_ms: float = 0.0
    
    @property
    def avg_tokens_per_merge(self) -> float:
        """Average tokens deduplicated per merge"""
        return (self.tokens_deduplicated / self.merge_events) if self.merge_events > 0 else 0.0


class CacheMonitor:
    """
    Monitor cache performance for LLM inference systems.
    
    Tracks cache hits/misses, prefix merging, and exports metrics to Prometheus.
    Implements monitoring as described in Chapter 16.
    """
    
    def __init__(self, export_interval: float = 10.0):
        """
        Initialize cache monitor.
        
        Args:
            export_interval: How often to export aggregated metrics (seconds)
        """
        self.export_interval = export_interval
        self.stats: Dict[CacheType, CacheStats] = {
            cache_type: CacheStats() for cache_type in CacheType
        }
        self.prefix_stats = PrefixMergeStats()
        self.lock = threading.Lock()
        
        # Prometheus metrics
        self.cache_hits = Counter(
            'llm_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses = Counter(
            'llm_cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate percentage',
            ['cache_type']
        )
        
        self.prefix_cache_hit_rate = Gauge(
            'prefix_cache_hit_rate',
            'Prefix cache hit rate percentage'
        )
        
        self.kv_cache_hit_rate = Gauge(
            'kv_cache_hit_rate',
            'KV cache hit rate percentage'
        )
        
        self.cache_evictions = Counter(
            'llm_cache_evictions_total',
            'Total cache evictions',
            ['cache_type']
        )
        
        self.cache_size = Gauge(
            'llm_cache_size_bytes',
            'Current cache size in bytes',
            ['cache_type']
        )
        
        self.cache_utilization = Gauge(
            'llm_cache_utilization_percent',
            'Cache utilization percentage',
            ['cache_type']
        )
        
        self.cache_entries = Gauge(
            'llm_cache_entries',
            'Number of entries in cache',
            ['cache_type']
        )
        
        self.prefix_merges = Counter(
            'llm_prefix_merges_total',
            'Total prefix merge events'
        )
        
        self.prefix_matching = Counter(
            'prefix_matching',
            'Prefix matching events (vLLM compatible metric)'
        )
        
        self.tokens_deduplicated = Counter(
            'llm_tokens_deduplicated_total',
            'Total tokens deduplicated via prefix merging'
        )
        
        self.compute_saved = Counter(
            'llm_compute_saved_milliseconds',
            'Compute time saved via caching (milliseconds)'
        )
        
        self.cache_access_latency = Histogram(
            'llm_cache_access_latency_seconds',
            'Cache access latency',
            ['cache_type'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        )
        
    def record_cache_access(
        self,
        cache_type: CacheType,
        hit: bool,
        latency_seconds: Optional[float] = None
    ):
        """
        Record a cache access (hit or miss).
        
        Args:
            cache_type: Type of cache accessed
            hit: True if cache hit, False if miss
            latency_seconds: Optional access latency
        """
        with self.lock:
            stats = self.stats[cache_type]
            
            if hit:
                stats.hits += 1
                self.cache_hits.labels(cache_type=cache_type.value).inc()
            else:
                stats.misses += 1
                self.cache_misses.labels(cache_type=cache_type.value).inc()
            
            if latency_seconds is not None:
                self.cache_access_latency.labels(cache_type=cache_type.value).observe(latency_seconds)
    
    def record_cache_eviction(self, cache_type: CacheType):
        """Record a cache eviction event."""
        with self.lock:
            self.stats[cache_type].evictions += 1
            self.cache_evictions.labels(cache_type=cache_type.value).inc()
    
    def update_cache_size(
        self,
        cache_type: CacheType,
        size_bytes: int,
        capacity_bytes: int,
        entries: int
    ):
        """
        Update cache size and utilization metrics.
        
        Args:
            cache_type: Type of cache
            size_bytes: Current size in bytes
            capacity_bytes: Total capacity in bytes
            entries: Number of entries in cache
        """
        with self.lock:
            stats = self.stats[cache_type]
            stats.size_bytes = size_bytes
            stats.capacity_bytes = capacity_bytes
            stats.entries = entries
            
            self.cache_size.labels(cache_type=cache_type.value).set(size_bytes)
            self.cache_utilization.labels(cache_type=cache_type.value).set(stats.utilization)
            self.cache_entries.labels(cache_type=cache_type.value).set(entries)
    
    def record_prefix_merge(
        self,
        tokens_deduplicated: int,
        compute_saved_ms: float = 0.0
    ):
        """
        Record a prefix merge/deduplication event.
        
        Args:
            tokens_deduplicated: Number of tokens deduplicated
            compute_saved_ms: Estimated compute time saved (milliseconds)
        """
        with self.lock:
            self.prefix_stats.merge_events += 1
            self.prefix_stats.tokens_deduplicated += tokens_deduplicated
            self.prefix_stats.compute_saved_ms += compute_saved_ms
            
            self.prefix_merges.inc()
            self.prefix_matching.inc()  # vLLM-compatible metric
            self.tokens_deduplicated.inc(tokens_deduplicated)
            
            if compute_saved_ms > 0:
                self.compute_saved.inc(compute_saved_ms)
    
    def export_metrics(self):
        """Export current metrics to Prometheus gauges."""
        with self.lock:
            for cache_type, stats in self.stats.items():
                self.cache_hit_rate.labels(cache_type=cache_type.value).set(stats.hit_rate)
                
                # Update specific cache hit rates for dashboard compatibility
                if cache_type == CacheType.KV_CACHE:
                    self.kv_cache_hit_rate.set(stats.hit_rate)
                elif cache_type == CacheType.PREFIX_CACHE:
                    self.prefix_cache_hit_rate.set(stats.hit_rate)
    
    def get_stats(self, cache_type: CacheType) -> CacheStats:
        """Get statistics for a specific cache type."""
        with self.lock:
            return self.stats[cache_type]
    
    def get_prefix_stats(self) -> PrefixMergeStats:
        """Get prefix merging statistics."""
        with self.lock:
            return self.prefix_stats
    
    def print_summary(self):
        """Print a human-readable summary of cache statistics."""
        print("\n" + "="*60)
        print("Cache Performance Summary")
        print("="*60)
        
        with self.lock:
            for cache_type, stats in self.stats.items():
                print(f"\n{cache_type.value.upper()}:")
                print(f"  Hits:         {stats.hits:,}")
                print(f"  Misses:       {stats.misses:,}")
                print(f"  Hit Rate:     {stats.hit_rate:.2f}%")
                print(f"  Evictions:    {stats.evictions:,}")
                print(f"  Entries:      {stats.entries:,}")
                print(f"  Size:         {stats.size_bytes / (1024**2):.2f} MB")
                print(f"  Capacity:     {stats.capacity_bytes / (1024**2):.2f} MB")
                print(f"  Utilization:  {stats.utilization:.2f}%")
            
            print(f"\nPREFIX MERGING:")
            print(f"  Merge Events:         {self.prefix_stats.merge_events:,}")
            print(f"  Tokens Deduplicated:  {self.prefix_stats.tokens_deduplicated:,}")
            print(f"  Avg per Merge:        {self.prefix_stats.avg_tokens_per_merge:.1f}")
            print(f"  Compute Saved:        {self.prefix_stats.compute_saved_ms:.2f} ms")
        
        print("="*60 + "\n")


class SimpleCache:
    """
    Simple LRU cache implementation with monitoring integration.
    
    Demonstrates how to integrate CacheMonitor with an actual cache.
    """
    
    def __init__(
        self,
        capacity_bytes: int,
        cache_type: CacheType,
        monitor: CacheMonitor,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    ):
        """
        Initialize cache.
        
        Args:
            capacity_bytes: Maximum cache size
            cache_type: Type of cache
            monitor: CacheMonitor instance for metrics
            eviction_policy: Eviction policy to use
        """
        self.capacity_bytes = capacity_bytes
        self.cache_type = cache_type
        self.monitor = monitor
        self.eviction_policy = eviction_policy
        
        self.cache: Dict[str, bytes] = {}
        self.access_order: OrderedDict[str, None] = OrderedDict()  # For LRU
        self.access_count: Dict[str, int] = defaultdict(int)  # For LFU
        self.size_bytes = 0
    
    def get(self, key: str) -> Optional[bytes]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.perf_counter()
        
        if key in self.cache:
            # Cache hit
            value = self.cache[key]
            self._update_access(key)
            
            latency = time.perf_counter() - start_time
            self.monitor.record_cache_access(self.cache_type, hit=True, latency_seconds=latency)
            
            return value
        else:
            # Cache miss
            latency = time.perf_counter() - start_time
            self.monitor.record_cache_access(self.cache_type, hit=False, latency_seconds=latency)
            
            return None
    
    def put(self, key: str, value: bytes):
        """
        Put value into cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        value_size = len(value)
        
        # Evict if necessary
        while self.size_bytes + value_size > self.capacity_bytes and self.cache:
            self._evict_one()
        
        # Add to cache
        if key not in self.cache:
            self.cache[key] = value
            self.size_bytes += value_size
            self._update_access(key)
            
            # Update monitoring
            self.monitor.update_cache_size(
                self.cache_type,
                self.size_bytes,
                self.capacity_bytes,
                len(self.cache)
            )
    
    def _update_access(self, key: str):
        """Update access tracking for eviction policy."""
        if self.eviction_policy == EvictionPolicy.LRU:
            self.access_order.pop(key, None)
            self.access_order[key] = None
        elif self.eviction_policy == EvictionPolicy.LFU:
            self.access_count[key] += 1
    
    def _evict_one(self):
        """Evict one entry based on eviction policy."""
        if not self.cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            key, _ = self.access_order.popitem(last=False)
        elif self.eviction_policy == EvictionPolicy.LFU:
            key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.access_count[key]
        else:  # FIFO
            key = next(iter(self.cache))
        
        value = self.cache.pop(key)
        self.size_bytes -= len(value)
        
        self.monitor.record_cache_eviction(self.cache_type)
        self.monitor.update_cache_size(
            self.cache_type,
            self.size_bytes,
            self.capacity_bytes,
            len(self.cache)
        )


def _run_demo(serve_metrics: bool = False, duration: float = 5.0):
    """
    Lightweight demo workload with optional Prometheus exporter.
    
    Keeps runtime short by default; enable --serve-metrics to expose gauges on :8001.
    """
    monitor = CacheMonitor(export_interval=2.0)
    cache = SimpleCache(
        capacity_bytes=8 * 1024 * 1024,  # 8 MB
        cache_type=CacheType.KV_CACHE,
        monitor=monitor,
        eviction_policy=EvictionPolicy.LRU,
    )
    
    if serve_metrics:
        start_http_server(8001)
        print("Prometheus exporter listening on http://localhost:8001/metrics")
    
    rng = random.Random(0)
    keys = [f"key_{i}" for i in range(256)]
    payload_sizes = [512, 1024, 2048, 4096]
    
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        key = rng.choice(keys)
        cached_value = cache.get(key)
        if cached_value is None:
            payload = bytes(rng.choice(payload_sizes))
            cache.put(key, payload)
        elif rng.random() < 0.1:
            # Occasional overwrite to exercise evictions/updates
            payload = bytes(rng.choice(payload_sizes))
            cache.put(key, payload)
        
        # Simulate prefix merge events intermittently
        if rng.random() < 0.2:
            monitor.record_prefix_merge(
                tokens_deduplicated=rng.randint(16, 256),
                compute_saved_ms=rng.random() * 2.0,
            )
        time.sleep(0.005)
    
    monitor.export_metrics()
    monitor.print_summary()
    if serve_metrics:
        print("Demo complete; leave the process running to keep metrics available.")


# Example usage and testing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cache monitoring demo (Chapter 16)")
    parser.add_argument(
        "--serve-metrics",
        action="store_true",
        help="Expose Prometheus exporter on :8001/metrics during the short demo run.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="How long to run the demo workload (seconds).",
    )
    args = parser.parse_args()
    _run_demo(serve_metrics=args.serve_metrics, duration=args.duration)
