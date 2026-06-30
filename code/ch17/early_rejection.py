import torch
from core.utils.architecture_runtime import get_arch_config

_ARCH_CFG = get_arch_config()

"""early_rejection.py
Chapter 17: Early Rejection Policies

Early rejection policies for disaggregated inference inspired by the Chapter 17
QoS mechanisms for ultra-scale inference systems."""

import time
import random
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from bisect import bisect_left, bisect_right, insort
import threading

class Priority(Enum):
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"


def _exclusive_quantile_from_sorted(sorted_values: List[float], n: int, cut_index: int) -> float:
    """Return one cut point using statistics.quantiles' exclusive method."""
    count = len(sorted_values)
    if count < 2:
        raise ValueError("exclusive quantile requires at least two samples")
    scale = count + 1
    j = cut_index * scale // n
    j = 1 if j < 1 else count - 1 if j > count - 1 else j
    delta = cut_index * scale - j * n
    return (sorted_values[j - 1] * (n - delta) + sorted_values[j] * delta) / n


def _ttft_p95_p99_from_ordered(ordered_samples: List[float]) -> Tuple[float, float]:
    count = len(ordered_samples)
    if count == 0:
        return 0.0, 0.0
    if count >= 100:
        return (
            _exclusive_quantile_from_sorted(ordered_samples, 100, 95),
            _exclusive_quantile_from_sorted(ordered_samples, 100, 99),
        )
    p95 = _exclusive_quantile_from_sorted(ordered_samples, 20, 19) if count >= 20 else ordered_samples[-1]
    return p95, ordered_samples[-1]


def _ttft_p95_p99(samples: List[float]) -> Tuple[float, float]:
    samples.sort()
    return _ttft_p95_p99_from_ordered(samples)


def _count_ttft_violations_from_ordered(ordered_samples: List[float], slo_limit: float) -> int:
    return len(ordered_samples) - bisect_right(ordered_samples, slo_limit)

@dataclass
class Request:
    id: str
    prompt_length: int
    expected_output_length: int
    priority: Priority
    arrival_time: float
    deadline: Optional[float] = None
    
    def __post_init__(self):
        # Set deadline based on priority
        if self.deadline is None:
            if self.priority == Priority.PREMIUM:
                self.deadline = self.arrival_time + 0.2  # 200ms for premium
            elif self.priority == Priority.STANDARD:
                self.deadline = self.arrival_time + 0.5  # 500ms for standard
            else:
                self.deadline = self.arrival_time + 1.0  # 1000ms for free

@dataclass
class SystemMetrics:
    """Real-time system metrics for admission control."""
    prefill_queue_length: int = 0
    decode_queue_length: int = 0
    avg_prefill_time_per_req: float = 50.0  # ms
    avg_decode_time_per_req: float = 10.0   # ms
    current_load: float = 0.0  # 0-1
    recent_ttft_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_tpot_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_ttft_ordered: List[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


def _ordered_ttft_samples(metrics: SystemMetrics) -> List[float]:
    if len(metrics.recent_ttft_ordered) != len(metrics.recent_ttft_samples):
        metrics.recent_ttft_ordered = sorted(metrics.recent_ttft_samples)
    return metrics.recent_ttft_ordered


def _append_ttft_sample(metrics: SystemMetrics, sample: float) -> None:
    evicted = None
    if (
        metrics.recent_ttft_samples.maxlen is not None
        and len(metrics.recent_ttft_samples) == metrics.recent_ttft_samples.maxlen
    ):
        evicted = metrics.recent_ttft_samples[0]
    metrics.recent_ttft_samples.append(sample)
    if evicted is not None:
        evict_idx = bisect_left(metrics.recent_ttft_ordered, evicted)
        if (
            evict_idx < len(metrics.recent_ttft_ordered)
            and metrics.recent_ttft_ordered[evict_idx] == evicted
        ):
            del metrics.recent_ttft_ordered[evict_idx]
        else:
            metrics.recent_ttft_ordered = sorted(metrics.recent_ttft_samples)
            return
    insort(metrics.recent_ttft_ordered, sample)

class QoSController:
    """
    Quality of Service controller implementing early rejection policies.
    Based on Chapter 17's admission control examples.
    """
    
    def __init__(self):
        # SLO thresholds (milliseconds)
        self.TTFT_SLO_MAX = {
            Priority.PREMIUM: 200,   # 200ms for premium
            Priority.STANDARD: 500,  # 500ms for standard  
            Priority.FREE: 1000      # 1000ms for free
        }
        
        self.TPOT_SLO_MAX = {
            Priority.PREMIUM: 30,    # 30ms per token
            Priority.STANDARD: 50,   # 50ms per token
            Priority.FREE: 100       # 100ms per token
        }
        
        # Capacity limits
        self.MAX_CONCURRENT_REQUESTS = {
            Priority.PREMIUM: 50,    # Reserve capacity for premium
            Priority.STANDARD: 100,  # Standard tier capacity
            Priority.FREE: 200       # Best effort for free
        }
        
        # Current system state
        self.metrics = SystemMetrics()
        self.active_requests: Dict[Priority, int] = {
            Priority.PREMIUM: 0,
            Priority.STANDARD: 0,
            Priority.FREE: 0
        }
        
        # Request tracking
        self.admitted_requests: List[Request] = []
        self.rejected_requests: List[Request] = []
        
        # Performance tracking
        self.rejection_stats = {
            Priority.PREMIUM: {"total": 0, "rejected": 0},
            Priority.STANDARD: {"total": 0, "rejected": 0},
            Priority.FREE: {"total": 0, "rejected": 0}
        }
        
        self.lock = threading.Lock()
    
    def admit_request(self, request: Request) -> bool:
        """
        Core admission control function from Chapter 17.
        
        Early rejection based on estimated latency and priority.
        """
        with self.lock:
            self.rejection_stats[request.priority]["total"] += 1
            
            # Step 1: Check capacity limits
            if not self._check_capacity_limits(request):
                self._reject_request(request, "capacity_limit")
                return False
            
            # Step 2: Estimate TTFT for this request
            estimated_ttft = self._estimate_ttft(request)
            
            # Step 3: Check SLO compliance
            slo_limit = self.TTFT_SLO_MAX[request.priority]
            
            if estimated_ttft > slo_limit:
                if request.priority == Priority.FREE:
                    # Always reject free tier if SLO would be violated
                    self._reject_request(request, "slo_violation")
                    return False
                elif request.priority == Priority.STANDARD:
                    # Reject standard if load is very high
                    if self.metrics.current_load > 0.8:
                        self._reject_request(request, "high_load")
                        return False
                # Premium requests are rarely rejected
            
            # Step 4: Additional checks for system health
            if not self._system_health_check(request):
                self._reject_request(request, "system_health")
                return False
            
            # Request is admitted
            self._admit_request(request)
            return True
    
    def _check_capacity_limits(self, request: Request) -> bool:
        """Check if we have capacity for this priority level."""
        current_count = self.active_requests[request.priority]
        max_allowed = self.MAX_CONCURRENT_REQUESTS[request.priority]
        
        if current_count >= max_allowed:
            print(f"Capacity limit reached for {request.priority.value}: "
                  f"{current_count}/{max_allowed}")
            return False
        
        return True
    
    def _estimate_ttft(self, request: Request) -> float:
        """
        Estimate Time-To-First-Token based on current system state.
        Implementation from Chapter 17.
        """
        # Base estimation from queue lengths
        est_ttft = (self.metrics.prefill_queue_length * 
                   self.metrics.avg_prefill_time_per_req)
        
        # Consider decode backlog as well
        est_ttft += (self.metrics.decode_queue_length * 
                    self.metrics.avg_decode_time_per_req)
        
        # Adjust for request size (capped to prevent extreme values)
        # Larger prompts take longer, but cap the factor
        size_factor = min(5.0, max(1.0, request.prompt_length / 100.0))
        est_ttft *= size_factor
        
        # Adjust for system load (capped to prevent extreme values)
        load_factor = min(3.0, 1.0 + self.metrics.current_load)
        est_ttft *= load_factor
        
        # Priority gets better estimates (more accurate prediction)
        if request.priority == Priority.PREMIUM:
            est_ttft *= 0.9  # Premium gets 10% better estimates
        elif request.priority == Priority.FREE:
            est_ttft *= 1.1  # Free tier gets 10% worse estimates
        
        # Cap the final estimate to prevent unreasonable values
        max_reasonable_ttft = 10000.0  # 10 seconds max
        return min(max_reasonable_ttft, max(1.0, est_ttft))
    
    def _system_health_check(self, request: Request) -> bool:
        """Additional system health checks."""
        # Check recent performance
        if len(self.metrics.recent_ttft_samples) > 10:
            recent_p95_ttft = _exclusive_quantile_from_sorted(
                _ordered_ttft_samples(self.metrics), 20, 19
            )
            
            # If recent performance is bad, be more conservative
            if recent_p95_ttft > self.TTFT_SLO_MAX[Priority.STANDARD] * 1.5:
                if request.priority == Priority.FREE:
                    return False
        
        # Check memory pressure (simulated)
        memory_pressure = random.uniform(0, 1)
        if memory_pressure > 0.9:
            if request.priority != Priority.PREMIUM:
                print(f"High memory pressure, rejecting {request.priority.value} request")
                return False
        
        return True
    
    def _admit_request(self, request: Request):
        """Admit the request and update tracking."""
        self.active_requests[request.priority] += 1
        self.admitted_requests.append(request)
        
        print(f"ADMITTED {request.id} ({request.priority.value}) - "
              f"estimated TTFT: {self._estimate_ttft(request):.1f}ms")
    
    def _reject_request(self, request: Request, reason: str):
        """Reject the request and update statistics."""
        self.rejection_stats[request.priority]["rejected"] += 1
        self.rejected_requests.append(request)
        
        estimated_ttft = self._estimate_ttft(request)
        slo_limit = self.TTFT_SLO_MAX[request.priority]
        
        print(f"REJECTED {request.id} ({request.priority.value}) - "
              f"reason: {reason}, estimated TTFT: {estimated_ttft:.1f}ms "
              f"(limit: {slo_limit}ms)")
    
    def complete_request(self, request: Request, actual_ttft: float, actual_tpot: float):
        """Mark request as completed and update metrics."""
        with self.lock:
            self.active_requests[request.priority] -= 1
            
            # Update performance metrics
            _append_ttft_sample(self.metrics, actual_ttft)
            self.metrics.recent_tpot_samples.append(actual_tpot)
            
            # Update exponential moving averages
            alpha = 0.1
            self.metrics.avg_prefill_time_per_req = (
                alpha * actual_ttft + 
                (1 - alpha) * self.metrics.avg_prefill_time_per_req
            )
            
            self.metrics.avg_decode_time_per_req = (
                alpha * actual_tpot + 
                (1 - alpha) * self.metrics.avg_decode_time_per_req
            )
    
    def update_system_metrics(self, prefill_queue: int, decode_queue: int, load: float):
        """Update system metrics for admission control."""
        with self.lock:
            self.metrics.prefill_queue_length = prefill_queue
            self.metrics.decode_queue_length = decode_queue
            self.metrics.current_load = load
            self.metrics.last_updated = time.time()
    
    def get_rejection_rate(self, priority: Priority) -> float:
        """Get rejection rate for a priority level."""
        stats = self.rejection_stats[priority]
        if stats["total"] == 0:
            return 0.0
        return stats["rejected"] / stats["total"]
    
    def print_stats(self):
        """Print current QoS statistics."""
        print("\n=== QoS Statistics ===")
        
        total_requests = 0
        total_rejected = 0
        for stats in self.rejection_stats.values():
            total_requests += stats["total"]
            total_rejected += stats["rejected"]
        overall_rejection_rate = (total_rejected / total_requests * 100) if total_requests else 0.0
        
        print(f"Total requests: {total_requests}")
        print(f"Total rejected: {total_rejected}")
        print(f"Overall rejection rate: {overall_rejection_rate:.1f}%")
        
        print(f"\nBy priority:")
        for priority in Priority:
            stats = self.rejection_stats[priority]
            rate = self.get_rejection_rate(priority)
            active = self.active_requests[priority]
            capacity = self.MAX_CONCURRENT_REQUESTS[priority]
            
            print(f"  {priority.value:8}: {stats['rejected']:3}/{stats['total']:3} rejected "
                  f"({rate*100:5.1f}%), active: {active:3}/{capacity:3}")
        
        print(f"\nCurrent system state:")
        print(f"  Prefill queue: {self.metrics.prefill_queue_length}")
        print(f"  Decode queue: {self.metrics.decode_queue_length}")
        print(f"  System load: {self.metrics.current_load:.1f}")
        print(f"  Avg TTFT: {self.metrics.avg_prefill_time_per_req:.1f}ms")
        print(f"  Avg TPOT: {self.metrics.avg_decode_time_per_req:.1f}ms")

def simulate_load_spike():
    """Simulate a realistic load spike scenario."""
    qos = QoSController()
    
    print("Chapter 17: Early Rejection and Quality of Service")
    print("=" * 50)
    
    # Simulate different traffic patterns
    scenarios = [
        {
            "name": "Normal Load",
            "duration": 30,
            "request_rate": 2.0,  # requests per second
            "load_factor": 0.3,
            "priority_distribution": {Priority.PREMIUM: 0.1, Priority.STANDARD: 0.6, Priority.FREE: 0.3}
        },
        {
            "name": "Traffic Spike",
            "duration": 20, 
            "request_rate": 8.0,
            "load_factor": 0.8,
            "priority_distribution": {Priority.PREMIUM: 0.05, Priority.STANDARD: 0.3, Priority.FREE: 0.65}
        },
        {
            "name": "Heavy Premium Load",
            "duration": 15,
            "request_rate": 5.0,
            "load_factor": 0.9,
            "priority_distribution": {Priority.PREMIUM: 0.4, Priority.STANDARD: 0.4, Priority.FREE: 0.2}
        }
    ]
    
    request_id = 0
    
    for scenario in scenarios:
        print(f"\n=== {scenario['name']} ===")
        
        # Update system state
        base_queue = int(scenario['load_factor'] * 10)
        qos.update_system_metrics(
            prefill_queue=base_queue,
            decode_queue=base_queue // 2,
            load=scenario['load_factor']
        )
        
        # Generate requests for this scenario
        scenario_start = time.perf_counter()
        while time.perf_counter() - scenario_start < scenario['duration']:
            # Randomly generate a request
            priority = random.choices(
                list(scenario['priority_distribution'].keys()),
                weights=list(scenario['priority_distribution'].values())
            )[0]
            
            request = Request(
                id=f"req-{request_id:04d}",
                prompt_length=random.randint(50, 500),
                expected_output_length=random.randint(20, 200),
                priority=priority,
                arrival_time=time.time()
            )
            request_id += 1
            
            # Try to admit the request
            admitted = qos.admit_request(request)
            
            if admitted:
                # Simulate request processing
                actual_ttft = qos._estimate_ttft(request) + random.uniform(-10, 20)
                actual_tpot = qos.metrics.avg_decode_time_per_req + random.uniform(-5, 10)
                
                # Complete request after a short delay
                def complete_later():
                    time.sleep(0.1)  # Simulate processing time
                    qos.complete_request(request, actual_ttft, actual_tpot)
                
                threading.Thread(target=complete_later, daemon=True).start()
            
            # Wait between requests based on rate
            time.sleep(1.0 / scenario['request_rate'])
            
            # Occasionally update system metrics during the scenario
            if random.random() < 0.1:
                load_variance = random.uniform(-0.1, 0.1)
                new_load = max(0, min(1, scenario['load_factor'] + load_variance))
                new_prefill_queue = max(0, base_queue + random.randint(-2, 3))
                new_decode_queue = max(0, base_queue // 2 + random.randint(-1, 2))
                
                qos.update_system_metrics(new_prefill_queue, new_decode_queue, new_load)
        
        # Print stats for this scenario
        qos.print_stats()
        
        # Brief pause between scenarios
        time.sleep(1)
    
    print(f"\n=== Final Summary ===")
    qos.print_stats()
    
    # Analyze SLO compliance
    print(f"\n=== SLO Analysis ===")
    if len(qos.metrics.recent_ttft_samples) > 0:
        ttft_ordered = _ordered_ttft_samples(qos.metrics)
        ttft_p95, ttft_p99 = _ttft_p95_p99_from_ordered(ttft_ordered)
        ttft_count = len(ttft_ordered)
        
        print(f"TTFT P95: {ttft_p95:.1f}ms")
        print(f"TTFT P99: {ttft_p99:.1f}ms")
        
        # Check SLO compliance by priority
        for priority in Priority:
            slo_limit = qos.TTFT_SLO_MAX[priority]
            violations = _count_ttft_violations_from_ordered(ttft_ordered, slo_limit)
            violation_rate = violations / ttft_count if ttft_count else 0
            
            print(f"{priority.value} SLO ({slo_limit}ms): {violation_rate*100:.1f}% violations")

def demonstrate_qos_configuration():
    """Demonstrate QoS configuration similar to Chapter 17's YAML example."""
    
    # This simulates the configuration from Chapter 17
    qos_config = {
        "scheduler": {
            "qos_classes": [
                {
                    "name": "premium",
                    "reserved_fraction": 0.10,
                    "priority": 100,
                    "slo_ttft_ms": 200,
                    "slo_tpot_ms": 30
                },
                {
                    "name": "standard", 
                    "reserved_fraction": 0.30,
                    "priority": 50,
                    "slo_ttft_ms": 500,
                    "slo_tpot_ms": 50
                },
                {
                    "name": "free",
                    "reserved_fraction": 0.60,
                    "priority": 10,
                    "slo_ttft_ms": 1000,
                    "slo_tpot_ms": 100
                }
            ],
            "request_router": {
                "routes": [
                    {"match": {"header": "x-customer-tier: premium"}, "qos": "premium"},
                    {"match": {"header": "x-customer-tier: standard"}, "qos": "standard"},
                    {"match": {"header": "x-customer-tier: free"}, "qos": "free"},
                    {"match": {}, "qos": "free"}  # default fallback
                ]
            }
        }
    }
    
    print("=== QoS Configuration Example ===")
    print("This configuration reserves capacity for different tiers:")
    
    for qos_class in qos_config["scheduler"]["qos_classes"]:
        print(f"  {qos_class['name']:8}: {qos_class['reserved_fraction']*100:4.0f}% capacity, "
              f"TTFT≤{qos_class['slo_ttft_ms']}ms, TPOT≤{qos_class['slo_tpot_ms']}ms")
    
    return qos_config

def main():
    """Main demonstration of early rejection and QoS policies."""
    
    # First show the configuration
    demonstrate_qos_configuration()
    
    print(f"\n" + "="*60)
    
    # Run the load spike simulation
    simulate_load_spike()
    
    print(f"\n=== Key Benefits of Early Rejection ===")
    print("- Prevents system overload and cascade failures")
    print("- Maintains SLO compliance for admitted requests")
    print("- Provides fair resource allocation across priority tiers") 
    print("- Enables graceful degradation under extreme load")
    print("- Protects premium users from free tier traffic spikes")

if __name__ == "__main__":
    main()

# Architecture-specific optimizations
if torch.cuda.is_available():
    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if _ARCH_CFG.arch in {"blackwell", "grace_blackwell"} and triton_cfg is not None:
        try:
            if hasattr(triton_cfg, "use_blackwell_optimizations"):
                triton_cfg.use_blackwell_optimizations = True
            if hasattr(triton_cfg, "hbm3e_optimizations"):
                triton_cfg.hbm3e_optimizations = True
            if hasattr(triton_cfg, "tma_support"):
                triton_cfg.tma_support = True
            if hasattr(triton_cfg, "stream_ordered_memory"):
                triton_cfg.stream_ordered_memory = True
        except AttributeError:
            print("Blackwell optimizations not available in this PyTorch build")

    if triton_cfg is not None and hasattr(triton_cfg, "unique_kernel_names"):
        triton_cfg.unique_kernel_names = True
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.automatic_dynamic_shapes = True
