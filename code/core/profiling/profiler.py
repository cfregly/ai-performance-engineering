"""
UnifiedProfiler - Comprehensive GPU profiling.

Integrates:
- torch.profiler for kernel-level timing
- CUDA events for accurate GPU timing
- Memory tracking and snapshots
- Chrome trace export
- Flame graph data generation
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.cuda

from core.profiling.torch_profiler_compat import prepare_torch_cuda_profiler


@dataclass
class KernelInfo:
    """Information about a GPU kernel."""
    name: str
    duration_us: float
    cuda_time_us: float
    cpu_time_us: float
    device_index: int = 0
    stream: int = 0
    grid: Tuple[int, ...] = ()
    block: Tuple[int, ...] = ()
    registers_per_thread: int = 0
    shared_memory_bytes: int = 0
    call_count: int = 1


@dataclass
class ProfileSession:
    """Complete profiling session results."""
    
    # Timing
    total_time_ms: float = 0.0
    cuda_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0
    reserved_memory_mb: float = 0.0
    
    # Kernel breakdown
    kernels: List[KernelInfo] = field(default_factory=list)
    kernel_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Bottleneck analysis
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Export paths
    trace_path: Optional[Path] = None
    flame_graph_path: Optional[Path] = None
    memory_snapshot_path: Optional[Path] = None
    
    # Raw data for visualizations
    timeline_data: Dict[str, Any] = field(default_factory=dict)
    flame_graph_data: Dict[str, Any] = field(default_factory=dict)
    memory_timeline: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    device_name: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "timing": {
                "total_ms": self.total_time_ms,
                "cuda_ms": self.cuda_time_ms,
                "cpu_ms": self.cpu_time_ms,
            },
            "memory": {
                "peak_mb": self.peak_memory_mb,
                "allocated_mb": self.allocated_memory_mb,
                "reserved_mb": self.reserved_memory_mb,
            },
            "kernels": [
                {
                    "name": k.name,
                    "duration_us": k.duration_us,
                    "call_count": k.call_count,
                }
                for k in self.kernels[:50]  # Top 50
            ],
            "kernel_summary": self.kernel_summary,
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
            "device": self.device_name,
            "timestamp": self.timestamp,
        }


class UnifiedProfiler:
    """
    Unified GPU profiler combining multiple profiling techniques.
    
    Usage:
        profiler = UnifiedProfiler(output_dir="./profiles")
        
        with profiler.profile("my_benchmark") as session:
            # Your GPU code here
            model(inputs)
        
        print(session.total_time_ms)
        print(session.bottlenecks)
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enable_trace: bool = True,
        enable_memory: bool = True,
        enable_flame_graph: bool = True,
        warmup_iterations: int = 3,
        profile_iterations: int = 10,
        record_shapes: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        with_modules: bool = True,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("./profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_trace = enable_trace
        self.enable_memory = enable_memory
        self.enable_flame_graph = enable_flame_graph
        self.warmup_iterations = warmup_iterations
        self.profile_iterations = profile_iterations
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        
        self._current_session: Optional[ProfileSession] = None
    
    @contextmanager
    def profile(self, name: str = "profile"):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name for this profiling session
            
        Yields:
            ProfileSession that will be populated with results
        """
        session = ProfileSession(
            timestamp=datetime.now().isoformat(),
            device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        )
        
        self._current_session = session
        
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            # Enable memory recording if requested
            if self.enable_memory and torch.cuda.is_available():
                try:
                    torch.cuda.memory._record_memory_history(
                        max_entries=100000,
                        context="all",
                    )
                except Exception:
                    pass  # Memory history not available in all PyTorch versions
            
            # Set up torch.profiler
            activities = [torch.profiler.ProfilerActivity.CPU]
            cuda_profile_requested = torch.cuda.is_available()
            if cuda_profile_requested:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            prepare_torch_cuda_profiler(
                cuda_profile_requested=cuda_profile_requested
            )
            
            trace_path = self.output_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with torch.profiler.profile(
                activities=activities,
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                with_flops=self.with_flops,
                with_modules=self.with_modules,
                profile_memory=True,
                on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)) if self.enable_trace else None,
            ) as prof:
                
                # Start timing
                if torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                
                cpu_start = time.perf_counter()
                
                yield session
                
                cpu_end = time.perf_counter()
                
                if torch.cuda.is_available():
                    end_event.record()
                    torch.cuda.synchronize()
                    session.cuda_time_ms = start_event.elapsed_time(end_event)
                
                session.cpu_time_ms = (cpu_end - cpu_start) * 1000
                session.total_time_ms = session.cuda_time_ms if torch.cuda.is_available() else session.cpu_time_ms
            
            # Extract kernel information
            self._extract_kernel_info(prof, session)
            
            # Memory stats
            if torch.cuda.is_available():
                session.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                session.allocated_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                session.reserved_memory_mb = torch.cuda.memory_reserved() / 1024 / 1024
            
            # Save memory snapshot
            if self.enable_memory and torch.cuda.is_available():
                try:
                    snapshot = torch.cuda.memory._snapshot()
                    if snapshot:
                        snapshot_path = self.output_dir / f"{name}_memory.pickle"
                        torch.save(snapshot, snapshot_path)
                        session.memory_snapshot_path = snapshot_path
                except Exception:
                    pass
                finally:
                    try:
                        torch.cuda.memory._record_memory_history(enabled=None)
                    except Exception:
                        pass
            
            # Set trace path
            if self.enable_trace and trace_path.exists():
                session.trace_path = trace_path
            
            # Generate flame graph data
            if self.enable_flame_graph:
                session.flame_graph_data = self._generate_flame_graph_data(prof)
            
            # Analyze bottlenecks
            self._analyze_bottlenecks(session)
            
        finally:
            self._current_session = None
    
    def profile_function(
        self,
        fn: Callable,
        *args,
        name: str = "function",
        warmup: int = None,
        iterations: int = None,
        **kwargs,
    ) -> ProfileSession:
        """
        Profile a function with warmup and multiple iterations.
        
        Args:
            fn: Function to profile
            *args: Arguments to pass to function
            name: Name for profiling session
            warmup: Warmup iterations (default: self.warmup_iterations)
            iterations: Profile iterations (default: self.profile_iterations)
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            ProfileSession with results
        """
        warmup = warmup or self.warmup_iterations
        iterations = iterations or self.profile_iterations
        
        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile
        with self.profile(name) as session:
            for _ in range(iterations):
                fn(*args, **kwargs)
        
        # Adjust times to per-iteration
        session.total_time_ms /= iterations
        session.cuda_time_ms /= iterations
        session.cpu_time_ms /= iterations
        
        return session
    
    def _extract_kernel_info(self, prof, session: ProfileSession):
        """Extract kernel timing information from profiler."""
        
        kernel_times = {}
        
        try:
            events = prof.key_averages()
            
            for event in events:
                if event.key and event.cuda_time_total > 0:
                    name = event.key
                    
                    # Clean up kernel names
                    if '::' in name:
                        name = name.split('::')[-1]
                    
                    if name not in kernel_times:
                        kernel_times[name] = {
                            'cuda_time_us': 0,
                            'cpu_time_us': 0,
                            'count': 0,
                        }
                    
                    kernel_times[name]['cuda_time_us'] += event.cuda_time_total
                    kernel_times[name]['cpu_time_us'] += event.cpu_time_total
                    kernel_times[name]['count'] += event.count
            
            # Sort by CUDA time
            sorted_kernels = sorted(
                kernel_times.items(),
                key=lambda x: x[1]['cuda_time_us'],
                reverse=True
            )
            
            # Create KernelInfo objects
            for name, data in sorted_kernels[:100]:  # Top 100 kernels
                session.kernels.append(KernelInfo(
                    name=name,
                    duration_us=data['cuda_time_us'],
                    cuda_time_us=data['cuda_time_us'],
                    cpu_time_us=data['cpu_time_us'],
                    call_count=data['count'],
                ))
            
            # Summary by kernel type
            session.kernel_summary = {
                name: {
                    'total_us': data['cuda_time_us'],
                    'count': data['count'],
                    'avg_us': data['cuda_time_us'] / max(data['count'], 1),
                }
                for name, data in sorted_kernels[:20]
            }
            
        except Exception as e:
            session.bottlenecks.append(f"Kernel extraction error: {e}")
    
    def _generate_flame_graph_data(self, prof) -> Dict[str, Any]:
        """Generate flame graph data structure."""
        
        # Hierarchical structure for flame graph
        root = {
            "name": "root",
            "value": 0,
            "children": []
        }
        
        try:
            events = prof.key_averages(group_by_stack_n=5)
            
            for event in events:
                if event.cuda_time_total > 0:
                    current = root
                    
                    # Build hierarchy from stack
                    if hasattr(event, 'stack') and event.stack:
                        for frame in event.stack[::-1]:  # Reverse to root-first
                            frame_name = str(frame).split('\n')[0][:50]
                            
                            # Find or create child
                            child = next(
                                (c for c in current.get('children', []) if c['name'] == frame_name),
                                None
                            )
                            
                            if child is None:
                                child = {"name": frame_name, "value": 0, "children": []}
                                current.setdefault('children', []).append(child)
                            
                            current = child
                    
                    # Add kernel at leaf
                    kernel_name = event.key.split('::')[-1] if event.key else "unknown"
                    leaf = {
                        "name": kernel_name,
                        "value": event.cuda_time_total,
                        "children": []
                    }
                    current.setdefault('children', []).append(leaf)
            
            # Calculate totals
            def sum_values(node):
                node['value'] = sum(sum_values(c) for c in node.get('children', [])) or node['value']
                return node['value']
            
            sum_values(root)
            
        except Exception:
            pass
        
        return root
    
    def _analyze_bottlenecks(self, session: ProfileSession):
        """Analyze profiling data to identify bottlenecks."""
        
        bottlenecks = []
        recommendations = []
        
        # Check kernel distribution
        if session.kernels:
            total_cuda_time = sum(k.cuda_time_us for k in session.kernels)
            top_kernel = session.kernels[0]
            
            if total_cuda_time > 0:
                top_pct = (top_kernel.cuda_time_us / total_cuda_time) * 100
                
                if top_pct > 50:
                    bottlenecks.append(
                        f"Single kernel dominates: {top_kernel.name} ({top_pct:.1f}%)"
                    )
                    
                    if 'matmul' in top_kernel.name.lower() or 'gemm' in top_kernel.name.lower():
                        recommendations.append("Consider using torch.compile or Tensor Cores")
                    elif 'elementwise' in top_kernel.name.lower():
                        recommendations.append("Multiple elementwise ops - consider kernel fusion")
        
        # Check memory usage
        if session.peak_memory_mb > 0:
            if torch.cuda.is_available():
                total_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                memory_pct = (session.peak_memory_mb / total_memory_mb) * 100
                
                if memory_pct > 80:
                    bottlenecks.append(f"High memory usage: {memory_pct:.1f}% of GPU memory")
                    recommendations.append("Consider gradient checkpointing or mixed precision")
        
        # Check CPU/GPU balance
        if session.cpu_time_ms > 0 and session.cuda_time_ms > 0:
            cpu_pct = session.cpu_time_ms / (session.cpu_time_ms + session.cuda_time_ms) * 100
            if cpu_pct > 30:
                bottlenecks.append(f"CPU overhead: {cpu_pct:.1f}% of total time")
                recommendations.append("Consider CUDA Graphs or reducing Python overhead")
        
        # Check kernel launch overhead
        total_kernels = sum(k.call_count for k in session.kernels)
        if total_kernels > 1000 and session.total_time_ms > 0:
            launches_per_ms = total_kernels / session.total_time_ms
            if launches_per_ms > 100:
                bottlenecks.append(f"High kernel launch rate: {launches_per_ms:.0f} launches/ms")
                recommendations.append("Consider CUDA Graphs or torch.compile for kernel fusion")
        
        session.bottlenecks = bottlenecks
        session.recommendations = recommendations
    
    def export_summary(self, session: ProfileSession, path: Path):
        """Export profiling summary to JSON."""
        path = Path(path)
        path.write_text(json.dumps(session.to_dict(), indent=2))
    
    def compare_sessions(
        self,
        baseline: ProfileSession,
        optimized: ProfileSession,
    ) -> Dict[str, Any]:
        """Compare two profiling sessions."""
        
        speedup = baseline.total_time_ms / optimized.total_time_ms if optimized.total_time_ms > 0 else 0
        
        return {
            "speedup": speedup,
            "timing_diff": {
                "total_ms": baseline.total_time_ms - optimized.total_time_ms,
                "cuda_ms": baseline.cuda_time_ms - optimized.cuda_time_ms,
                "cpu_ms": baseline.cpu_time_ms - optimized.cpu_time_ms,
            },
            "memory_diff": {
                "peak_mb": baseline.peak_memory_mb - optimized.peak_memory_mb,
            },
            "kernel_changes": self._compare_kernels(baseline, optimized),
        }
    
    def _compare_kernels(
        self,
        baseline: ProfileSession,
        optimized: ProfileSession,
    ) -> Dict[str, Any]:
        """Compare kernel breakdowns between sessions."""
        
        baseline_kernels = {k.name: k for k in baseline.kernels}
        optimized_kernels = {k.name: k for k in optimized.kernels}
        
        all_names = set(baseline_kernels.keys()) | set(optimized_kernels.keys())
        
        changes = []
        for name in all_names:
            b = baseline_kernels.get(name)
            o = optimized_kernels.get(name)
            
            if b and o:
                diff = b.cuda_time_us - o.cuda_time_us
                if abs(diff) > 100:  # > 0.1ms difference
                    changes.append({
                        "name": name,
                        "baseline_us": b.cuda_time_us,
                        "optimized_us": o.cuda_time_us,
                        "diff_us": diff,
                    })
            elif b:
                changes.append({
                    "name": name,
                    "status": "removed",
                    "baseline_us": b.cuda_time_us,
                })
            elif o:
                changes.append({
                    "name": name,
                    "status": "added",
                    "optimized_us": o.cuda_time_us,
                })
        
        return sorted(changes, key=lambda x: abs(x.get('diff_us', x.get('baseline_us', 0))), reverse=True)[:20]


