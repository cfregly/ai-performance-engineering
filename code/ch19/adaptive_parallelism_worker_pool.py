#!/usr/bin/env python3

"""
Adaptive Parallelism Worker Pool System (Chapter 19)

Implements dynamic parallelism switching as described in Chapter 19.
Maintains multiple model instances with different sharding strategies (TP, PP, hybrid)
and routes requests to the optimal instance based on input characteristics.

Key features:
- Multiple worker pools with different parallelism strategies
- Dynamic request routing based on sequence length, memory, concurrency
- Real-time GPU utilization monitoring
- SLA-aware dispatching

Usage:
    pool_manager = AdaptiveParallelismManager(num_gpus=4, model_name="deepseek-r1")
    result = pool_manager.inference(prompt="...", max_tokens=100)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time


class ParallelismStrategy(Enum):
    """Parallelism strategies for model sharding"""
    TENSOR_PARALLEL = "tp"  # Tensor parallelism only
    PIPELINE_PARALLEL = "pp"  # Pipeline parallelism only
    HYBRID_TP_PP = "hybrid"  # Both TP and PP
    DATA_PARALLEL = "dp"  # Data parallel (replicas)


@dataclass
class WorkerPoolConfig:
    """Configuration for a worker pool"""
    strategy: ParallelismStrategy
    tp_degree: int  # Tensor parallel degree
    pp_degree: int  # Pipeline parallel degree
    gpu_ids: List[int]  # GPU IDs assigned to this pool
    max_seq_len: int  # Maximum sequence length this pool can handle
    max_batch_size: int  # Maximum batch size
    target_latency_ms: float  # Target latency in milliseconds
    

@dataclass
class InferenceRequest:
    """Container for an inference request"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float = 1.0
    seq_len: int = 0  # Will be computed
    priority: int = 0  # Higher = more important
    sla_latency_ms: Optional[float] = None  # Required latency SLA
    arrival_time: float = 0.0


@dataclass
class GPUMetrics:
    """Real-time GPU metrics for decision making"""
    gpu_id: int
    utilization: float  # 0-100%
    memory_used_gb: float
    memory_total_gb: float
    memory_util: float  # 0-100%
    temperature: float
    

def _count_whitespace_separated_tokens(text: str) -> int:
    """Count split-like tokens without allocating a list of substrings."""
    count = 0
    in_token = False
    for char in text:
        if char.isspace():
            in_token = False
        elif not in_token:
            count += 1
            in_token = True
    return count


class WorkerPool:
    """
    A worker pool managing a specific parallelism configuration.
    
    Each pool maintains model instances with a fixed sharding strategy
    and processes requests assigned to it.
    """
    
    def __init__(self, config: WorkerPoolConfig, model_name: str):
        """
        Initialize worker pool.
        
        Args:
            config: Worker pool configuration
            model_name: Name of the model to load
        """
        self.config = config
        self.model_name = model_name
        self.request_queue = queue.Queue()
        self.active_requests = 0
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.lock = threading.Lock()
        
        # Model instance (placeholder - would load actual model in production)
        self.model = None
        self.is_loaded = False
        
        print(f"Initialized worker pool: {config.strategy.value}, "
              f"TP={config.tp_degree}, PP={config.pp_degree}, "
              f"GPUs={config.gpu_ids}")
    
    def can_handle(self, request: InferenceRequest) -> bool:
        """
        Check if this pool can handle the request.
        
        Args:
            request: Inference request
            
        Returns:
            True if pool can handle this request
        """
        # Check sequence length
        if request.seq_len > self.config.max_seq_len:
            return False
        
        # Check if queue is not too full
        if self.request_queue.qsize() >= self.config.max_batch_size * 2:
            return False
        
        return True
    
    def get_estimated_latency(self, request: InferenceRequest) -> float:
        """
        Estimate latency for this request on this pool.
        
        Args:
            request: Inference request
            
        Returns:
            Estimated latency in milliseconds
        """
        # Base latency depends on strategy
        if self.config.strategy == ParallelismStrategy.TENSOR_PARALLEL:
            # TP: Low latency, all-reduce overhead
            base_latency = 10.0
            allreduce_overhead = 2.0 * self.config.tp_degree
        elif self.config.strategy == ParallelismStrategy.PIPELINE_PARALLEL:
            # PP: Higher latency due to pipeline bubbles
            base_latency = 20.0
            allreduce_overhead = 5.0 * self.config.pp_degree
        else:  # HYBRID
            base_latency = 15.0
            allreduce_overhead = 2.0 * self.config.tp_degree + 3.0 * self.config.pp_degree
        
        # Scale by sequence length
        seq_factor = request.seq_len / 1024.0
        
        # Add queuing delay based on active requests
        queue_delay = (self.active_requests + self.request_queue.qsize()) * 5.0
        
        return base_latency + allreduce_overhead + (seq_factor * 10.0) + queue_delay
    
    def submit_request(self, request: InferenceRequest):
        """Submit a request to this pool."""
        self.request_queue.put(request)
        with self.lock:
            self.active_requests += 1
    
    def process_request(self, request: InferenceRequest) -> Dict[str, Any]:
        """
        Process an inference request.
        
        Args:
            request: Inference request
            
        Returns:
            Response dictionary
        """
        try:
            self.request_queue.get_nowait()
        except queue.Empty:
            pass

        start_time = time.perf_counter()
        
        # Simulate inference (in production, would call actual model)
        # For demonstration, we just sleep based on estimated latency
        estimated_latency = self.get_estimated_latency(request)
        time.sleep(estimated_latency / 1000.0)
        
        actual_latency = (time.perf_counter() - start_time) * 1000.0
        
        with self.lock:
            self.active_requests -= 1
            self.total_requests += 1
            self.total_latency_ms += actual_latency
        
        return {
            "request_id": request.request_id,
            "output": f"Generated {request.max_tokens} tokens",
            "latency_ms": actual_latency,
            "worker_pool": self.config.strategy.value,
            "tp_degree": self.config.tp_degree,
            "pp_degree": self.config.pp_degree
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        with self.lock:
            avg_latency = (self.total_latency_ms / self.total_requests 
                          if self.total_requests > 0 else 0.0)
            return {
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "avg_latency_ms": avg_latency,
                "queue_size": self.request_queue.qsize()
            }


class AdaptiveParallelismManager:
    """
    Manages multiple worker pools with different parallelism strategies.
    
    Implements the adaptive parallelism approach from Chapter 19, routing
    requests to the optimal worker pool based on runtime characteristics.
    """
    
    def __init__(
        self,
        num_gpus: int = 8,
        model_name: str = "deepseek-r1",
        monitor_interval: float = 1.0
    ):
        """
        Initialize adaptive parallelism manager.
        
        Args:
            num_gpus: Total number of GPUs available
            model_name: Model to serve
            monitor_interval: How often to collect GPU metrics (seconds)
        """
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.monitor_interval = monitor_interval
        
        # Initialize worker pools with different strategies
        self.pools: List[WorkerPool] = []
        self._initialize_pools()
        
        # GPU metrics
        self.gpu_metrics: Dict[int, GPUMetrics] = {}
        self.metrics_lock = threading.Lock()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpus, daemon=True)
        self.monitor_thread.start()
    
    def _initialize_pools(self):
        """
        Initialize worker pools with different parallelism strategies.
        
        For a 4-GPU system:
        - Pool 1: TP=2, PP=1 (low latency, short sequences)
        - Pool 2: TP=1, PP=2 (hybrid, longer sequences)
        """
        if self.num_gpus == 4:
            # Pool 1: Tensor-parallel only (GPUs 0-1)
            # Best for: Short latency-sensitive requests
            pool1_config = WorkerPoolConfig(
                strategy=ParallelismStrategy.TENSOR_PARALLEL,
                tp_degree=2,
                pp_degree=1,
                gpu_ids=[0, 1],
                max_seq_len=4096,
                max_batch_size=32,
                target_latency_ms=50.0
            )
            self.pools.append(WorkerPool(pool1_config, self.model_name))
            
            # Pool 2: Pipeline parallel (GPUs 2-3)
            # Best for: Long sequences, high memory pressure
            pool2_config = WorkerPoolConfig(
                strategy=ParallelismStrategy.PIPELINE_PARALLEL,
                tp_degree=1,
                pp_degree=2,
                gpu_ids=[2, 3],
                max_seq_len=1000000,  # Support very long contexts
                max_batch_size=16,
                target_latency_ms=200.0
            )
            self.pools.append(WorkerPool(pool2_config, self.model_name))
            
        else:
            # Fallback: Single pool with all GPUs
            default_config = WorkerPoolConfig(
                strategy=ParallelismStrategy.TENSOR_PARALLEL,
                tp_degree=self.num_gpus,
                pp_degree=1,
                gpu_ids=list(range(self.num_gpus)),
                max_seq_len=8192,
                max_batch_size=32,
                target_latency_ms=100.0
            )
            self.pools.append(WorkerPool(default_config, self.model_name))
    
    def _monitor_gpus(self):
        """Background thread to monitor GPU metrics."""
        has_nvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            has_nvml = True
        except ImportError:
            print("Warning: pynvml not available, GPU monitoring disabled")
        except Exception as e:
            print(f"Warning: NVML initialization failed: {e}")
        
        while self.monitoring:
            if has_nvml:
                try:
                    for gpu_id in range(self.num_gpus):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        
                        metrics = GPUMetrics(
                            gpu_id=gpu_id,
                            utilization=util.gpu,
                            memory_used_gb=mem_info.used / (1024**3),
                            memory_total_gb=mem_info.total / (1024**3),
                            memory_util=util.memory,
                            temperature=temp
                        )
                        
                        with self.metrics_lock:
                            self.gpu_metrics[gpu_id] = metrics
                            
                except Exception as e:
                    print(f"Error monitoring GPUs: {e}")
            
            time.sleep(self.monitor_interval)
    
    def choose_worker_pool(
        self,
        request: InferenceRequest,
        concurrent_reqs: int
    ) -> WorkerPool:
        """
        Choose the optimal worker pool for a request.
        
        Implements and extends the decision logic from Chapter 19:
        - Long contexts or high memory pressure -> hybrid TP+PP
        - SLA-sensitive requests prefer pools that can meet the target latency
        - Many simultaneous small requests -> tensor parallel
        
        Args:
            request: Inference request with sequence length and SLA
            concurrent_reqs: Number of concurrent active requests
            
        Returns:
            Selected worker pool
        """
        seq_len = request.seq_len
        sla_latency = request.sla_latency_ms

        with self.metrics_lock:
            if self.gpu_metrics:
                avg_mem_util = sum(
                    m.memory_util for m in self.gpu_metrics.values()
                ) / len(self.gpu_metrics)
            else:
                avg_mem_util = 0.0
        
        if seq_len > 4096 or avg_mem_util > 80.0:
            preferred_strategy = ParallelismStrategy.HYBRID_TP_PP
        elif concurrent_reqs > 4:
            preferred_strategy = ParallelismStrategy.TENSOR_PARALLEL
        else:
            preferred_strategy = ParallelismStrategy.TENSOR_PARALLEL
        
        best_pool: Optional[WorkerPool] = None
        best_key: Optional[Tuple[int, int, float, float]] = None
        best_sla_pool: Optional[WorkerPool] = None
        best_sla_key: Optional[Tuple[int, float, int, float]] = None

        def consider_pool(pool: WorkerPool) -> None:
            nonlocal best_pool, best_key, best_sla_pool, best_sla_key
            estimated = pool.get_estimated_latency(request)
            strategy_rank = 0 if pool.config.strategy == preferred_strategy else 1
            queue_size = pool.request_queue.qsize()
            key = (
                strategy_rank,
                queue_size,
                abs(estimated - pool.config.target_latency_ms),
                estimated
            )
            if best_key is None or key < best_key:
                best_pool = pool
                best_key = key

            if sla_latency is not None and estimated <= sla_latency:
                sla_key = (
                    strategy_rank,
                    abs(estimated - min(sla_latency, pool.config.target_latency_ms)),
                    queue_size,
                    estimated
                )
                if best_sla_key is None or sla_key < best_sla_key:
                    best_sla_pool = pool
                    best_sla_key = sla_key

        candidate_found = False
        for pool in self.pools:
            if pool.can_handle(request):
                candidate_found = True
                consider_pool(pool)

        if not candidate_found:
            for pool in self.pools:
                consider_pool(pool)

        if best_sla_pool is not None:
            return best_sla_pool
        if best_pool is not None:
            return best_pool
        
        return self.pools[0]
    
    def inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        sla_latency_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run inference with automatic worker pool selection.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            sla_latency_ms: Optional latency SLA
            
        Returns:
            Inference result
        """
        # Create request
        now = time.time()
        request = InferenceRequest(
            request_id=f"req_{now}",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seq_len=_count_whitespace_separated_tokens(prompt),  # Rough approximation
            sla_latency_ms=sla_latency_ms,
            arrival_time=now
        )
        
        # Choose worker pool
        concurrent_reqs = sum(pool.active_requests for pool in self.pools)
        selected_pool = self.choose_worker_pool(request, concurrent_reqs)
        
        # Submit and process
        result = selected_pool.process_request(request)
        
        return result
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics for all worker pools."""
        stats = {
            "pools": []
        }
        
        for pool in self.pools:
            pool_stats = pool.get_stats()
            pool_stats["strategy"] = pool.config.strategy.value
            pool_stats["tp_degree"] = pool.config.tp_degree
            pool_stats["pp_degree"] = pool.config.pp_degree
            pool_stats["gpu_ids"] = pool.config.gpu_ids
            stats["pools"].append(pool_stats)
        
        # Add GPU metrics
        with self.metrics_lock:
            stats["gpu_metrics"] = {
                gpu_id: {
                    "utilization": m.utilization,
                    "memory_used_gb": m.memory_used_gb,
                    "memory_util": m.memory_util,
                    "temperature": m.temperature
                }
                for gpu_id, m in self.gpu_metrics.items()
            }
        
        return stats
    
    def shutdown(self):
        """Shutdown the manager and all worker pools."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)


# Add method to WorkerPool
def can_handle_seq_len(self, seq_len: int) -> bool:
    """Check if pool can handle sequence length."""
    return seq_len <= self.config.max_seq_len

WorkerPool.can_handle_seq_len = can_handle_seq_len


# Example usage
if __name__ == '__main__':
    print("=" * 70)
    print("Adaptive Parallelism Worker Pool Demo (Chapter 19)")
    print("=" * 70)
    
    # Initialize manager
    manager = AdaptiveParallelismManager(num_gpus=4, model_name="deepseek-r1")
    
    print("\nInitialized worker pools:")
    for i, pool in enumerate(manager.pools):
        print(f"  Pool {i+1}: {pool.config.strategy.value} "
              f"(TP={pool.config.tp_degree}, PP={pool.config.pp_degree}, "
              f"GPUs={pool.config.gpu_ids})")
    
    # Simulate various workloads
    print("\n" + "=" * 70)
    print("Testing different workload patterns")
    print("=" * 70)
    
    # Test 1: Short latency-sensitive request
    print("\n1. Short request (256 tokens):")
    result = manager.inference(
        prompt="Write a short story" * 50,  # ~100 tokens
        max_tokens=256,
        sla_latency_ms=50.0
    )
    print(f"   Routed to: {result['worker_pool']} (TP={result['tp_degree']}, PP={result['pp_degree']})")
    print(f"   Latency: {result['latency_ms']:.1f} ms")
    
    # Test 2: Long context request
    print("\n2. Long context request (100k tokens):")
    long_prompt = "Context: " * 20000  # Simulate very long prompt
    result = manager.inference(
        prompt=long_prompt,
        max_tokens=100,
        sla_latency_ms=200.0
    )
    print(f"   Routed to: {result['worker_pool']} (TP={result['tp_degree']}, PP={result['pp_degree']})")
    print(f"   Latency: {result['latency_ms']:.1f} ms")
    
    # Test 3: Multiple concurrent short requests
    print("\n3. Multiple concurrent requests (simulating high QPS):")
    import concurrent.futures
    completed = 0
    latency_total_ms = 0.0
    strategies_used = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = (
            executor.submit(
                manager.inference,
                prompt=f"Query {i}: " * 10,
                max_tokens=50
            )
            for i in range(8)
        )
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += 1
            latency_total_ms += result['latency_ms']
            strategies_used.add(result['worker_pool'])
    
    average_latency_ms = latency_total_ms / completed if completed else 0.0
    print(f"   Completed {completed} requests")
    print(f"   Average latency: {average_latency_ms:.1f} ms")
    print(f"   Strategies used: {strategies_used}")
    
    # Print cluster statistics
    print("\n" + "=" * 70)
    print("Cluster Statistics")
    print("=" * 70)
    stats = manager.get_cluster_stats()
    for pool_stats in stats["pools"]:
        print(f"\n{pool_stats['strategy'].upper()} Pool:")
        print(f"  Total requests: {pool_stats['total_requests']}")
        print(f"  Avg latency: {pool_stats['avg_latency_ms']:.1f} ms")
        print(f"  Active requests: {pool_stats['active_requests']}")
    
    # Cleanup
    manager.shutdown()
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
