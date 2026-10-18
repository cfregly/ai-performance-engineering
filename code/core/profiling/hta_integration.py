"""
HTA (Holistic Trace Analysis) Integration.

Integrates with Meta's HTA tool for comprehensive trace analysis:
- Temporal breakdown analysis
- Idle time analysis
- Kernel breakdown
- Communication analysis
- Memory analysis
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from core.profiling.torch_profiler_compat import prepare_torch_cuda_profiler


@dataclass
class HTAReport:
    """HTA analysis report."""
    
    # Temporal breakdown
    gpu_idle_time_pct: float = 0
    compute_time_pct: float = 0
    communication_time_pct: float = 0
    memory_time_pct: float = 0
    
    # Kernel analysis
    top_kernels: List[Dict[str, Any]] = field(default_factory=list)
    kernel_launch_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Communication analysis
    comm_breakdown: Dict[str, float] = field(default_factory=dict)
    collective_ops: List[Dict[str, Any]] = field(default_factory=list)
    
    # Memory analysis
    memory_bandwidth_gbps: float = 0
    memory_efficiency_pct: float = 0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    
    # Raw data
    raw_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temporal_breakdown": {
                "gpu_idle_pct": self.gpu_idle_time_pct,
                "compute_pct": self.compute_time_pct,
                "communication_pct": self.communication_time_pct,
                "memory_pct": self.memory_time_pct,
            },
            "kernel_analysis": {
                "top_kernels": self.top_kernels[:10],
                "launch_stats": self.kernel_launch_stats,
            },
            "communication": {
                "breakdown": self.comm_breakdown,
                "collective_ops": self.collective_ops[:10],
            },
            "memory": {
                "bandwidth_gbps": self.memory_bandwidth_gbps,
                "efficiency_pct": self.memory_efficiency_pct,
            },
            "recommendations": self.recommendations,
            "bottlenecks": self.bottlenecks,
        }


class HTAAnalyzer:
    """
    Integrate with HTA (Holistic Trace Analysis) for deep trace analysis.
    
    HTA provides comprehensive analysis of PyTorch Profiler traces including:
    - Temporal breakdown (compute vs idle vs communication)
    - Kernel-level analysis
    - Communication overhead analysis
    - Memory bandwidth utilization
    
    Usage:
        analyzer = HTAAnalyzer()
        
        # From existing trace
        report = analyzer.analyze_trace("trace.json")
        
        # With profiler
        with analyzer.profile("my_model") as session:
            model(input)
        report = session.report
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        hta_path: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/runs/analysis/hta")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hta_path = hta_path
        self._hta_available = self._check_hta_available()
    
    def _check_hta_available(self) -> bool:
        """Check if HTA is installed."""
        try:
            import hta
            return True
        except ImportError:
            return False
    
    def analyze_trace(
        self,
        trace_path: Path,
        analysis_types: Optional[List[str]] = None,
    ) -> HTAReport:
        """
        Analyze a trace file using HTA.
        
        Args:
            trace_path: Path to Chrome trace JSON
            analysis_types: Types of analysis to run
            
        Returns:
            HTAReport with analysis results
        """
        trace_path = Path(trace_path)
        report = HTAReport()
        
        if self._hta_available:
            report = self._run_hta_analysis(trace_path, analysis_types)
        else:
            # Fallback to manual trace analysis
            report = self._manual_trace_analysis(trace_path)
        
        # Add recommendations based on analysis
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def _run_hta_analysis(
        self,
        trace_path: Path,
        analysis_types: Optional[List[str]] = None,
    ) -> HTAReport:
        """Run HTA analysis on trace."""
        report = HTAReport()
        
        try:
            from hta.trace_analysis import TraceAnalysis
            
            # Create trace directory structure HTA expects
            trace_dir = trace_path.parent
            
            analyzer = TraceAnalysis(trace_dir=str(trace_dir))
            
            # Temporal breakdown
            try:
                temporal = analyzer.get_temporal_breakdown()
                if temporal is not None and not temporal.empty:
                    total_time = temporal.sum().sum()
                    if total_time > 0:
                        report.compute_time_pct = temporal.get('compute', temporal.sum(axis=1)).sum() / total_time * 100
                        report.gpu_idle_time_pct = temporal.get('idle', 0).sum() / total_time * 100 if 'idle' in temporal else 0
                        report.communication_time_pct = temporal.get('communication', 0).sum() / total_time * 100 if 'communication' in temporal else 0
            except Exception:
                pass
            
            # Kernel breakdown
            try:
                kernels = analyzer.get_gpu_kernel_breakdown()
                if kernels is not None and not kernels.empty:
                    for _, row in kernels.head(20).iterrows():
                        report.top_kernels.append({
                            "name": str(row.get('name', 'unknown')),
                            "duration_us": float(row.get('sum', 0)),
                            "count": int(row.get('count', 0)),
                        })
            except Exception:
                pass
            
            # Idle time analysis
            try:
                idle_time = analyzer.get_idle_time_breakdown()
                if idle_time is not None:
                    report.bottlenecks.append(f"GPU idle time: {report.gpu_idle_time_pct:.1f}%")
            except Exception:
                pass
            
            # Communication analysis
            try:
                comm = analyzer.get_comm_comp_overlap()
                if comm is not None:
                    report.raw_analysis['comm_overlap'] = comm.to_dict() if hasattr(comm, 'to_dict') else str(comm)
            except Exception:
                pass
            
        except Exception as e:
            report.bottlenecks.append(f"HTA analysis error: {e}")
        
        return report
    
    def _manual_trace_analysis(self, trace_path: Path) -> HTAReport:
        """Fallback manual trace analysis when HTA is not available."""
        report = HTAReport()
        
        try:
            with open(trace_path) as f:
                trace = json.load(f)
            
            events = trace if isinstance(trace, list) else trace.get('traceEvents', [])
            
            # Analyze events
            kernel_times = {}
            total_time = 0
            cuda_time = 0
            comm_time = 0
            
            for event in events:
                if event.get('ph') != 'X':
                    continue
                
                name = event.get('name', '')
                dur = event.get('dur', 0)
                cat = event.get('cat', '')
                
                total_time += dur
                
                # Categorize
                name_lower = name.lower()
                cat_lower = cat.lower()
                
                if 'cuda' in cat_lower or 'kernel' in cat_lower:
                    cuda_time += dur
                    
                    if name not in kernel_times:
                        kernel_times[name] = {'total': 0, 'count': 0}
                    kernel_times[name]['total'] += dur
                    kernel_times[name]['count'] += 1
                
                if 'nccl' in name_lower or 'allreduce' in name_lower or 'allgather' in name_lower:
                    comm_time += dur
                    report.collective_ops.append({
                        'name': name,
                        'duration_us': dur,
                    })
            
            # Calculate percentages
            if total_time > 0:
                report.compute_time_pct = (cuda_time / total_time) * 100
                report.communication_time_pct = (comm_time / total_time) * 100
                report.gpu_idle_time_pct = max(0, 100 - report.compute_time_pct - report.communication_time_pct)
            
            # Top kernels
            sorted_kernels = sorted(
                kernel_times.items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )
            
            for name, data in sorted_kernels[:20]:
                report.top_kernels.append({
                    'name': name,
                    'duration_us': data['total'],
                    'count': data['count'],
                })
            
            # Kernel launch stats
            if kernel_times:
                report.kernel_launch_stats = {
                    'total_kernels': sum(k['count'] for k in kernel_times.values()),
                    'unique_kernels': len(kernel_times),
                    'avg_kernel_time_us': cuda_time / max(1, sum(k['count'] for k in kernel_times.values())),
                }
            
        except Exception as e:
            report.bottlenecks.append(f"Manual analysis error: {e}")
        
        return report
    
    def _generate_recommendations(self, report: HTAReport) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Idle time recommendations
        if report.gpu_idle_time_pct > 30:
            recommendations.append(
                f"High GPU idle time ({report.gpu_idle_time_pct:.1f}%). "
                "Consider: CUDA Graphs, reducing Python overhead, or overlapping compute with data loading."
            )
        
        # Communication recommendations
        if report.communication_time_pct > 20:
            recommendations.append(
                f"Significant communication overhead ({report.communication_time_pct:.1f}%). "
                "Consider: overlapping communication with compute, gradient compression, or reducing synchronization."
            )
        
        # Kernel launch recommendations
        if report.kernel_launch_stats:
            total_kernels = report.kernel_launch_stats.get('total_kernels', 0)
            avg_time = report.kernel_launch_stats.get('avg_kernel_time_us', 0)
            
            if total_kernels > 10000 and avg_time < 10:
                recommendations.append(
                    f"Many small kernels ({total_kernels} launches, avg {avg_time:.1f}μs). "
                    "Consider: torch.compile for kernel fusion, or CUDA Graphs."
                )
        
        # Top kernel recommendations
        if report.top_kernels:
            top = report.top_kernels[0]
            total_kernel_time = sum(k['duration_us'] for k in report.top_kernels)
            
            if total_kernel_time > 0:
                top_pct = (top['duration_us'] / total_kernel_time) * 100
                
                if top_pct > 50:
                    recommendations.append(
                        f"Kernel '{top['name'][:40]}' dominates ({top_pct:.1f}% of GPU time). "
                        "Consider optimizing this specific kernel or finding alternatives."
                    )
        
        if not recommendations:
            recommendations.append("Profile looks well-optimized. Consider micro-optimizations if needed.")
        
        return recommendations
    
    def profile_and_analyze(
        self,
        fn,
        *args,
        name: str = "profile",
        warmup: int = 3,
        iterations: int = 10,
        **kwargs,
    ) -> HTAReport:
        """
        Profile a function and run HTA analysis.
        
        Args:
            fn: Function to profile
            name: Name for the profile
            warmup: Warmup iterations
            iterations: Profile iterations
            
        Returns:
            HTAReport with analysis
        """
        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile
        trace_path = self.output_dir / f"{name}_trace.json"
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        cuda_profile_requested = torch.cuda.is_available()
        if cuda_profile_requested:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prepare_torch_cuda_profiler(
            cuda_profile_requested=cuda_profile_requested
        )
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            profile_memory=True,
        ) as prof:
            for _ in range(iterations):
                fn(*args, **kwargs)
        
        # Export trace
        prof.export_chrome_trace(str(trace_path))
        
        # Analyze
        return self.analyze_trace(trace_path)
    
    def export_report(self, report: HTAReport, path: Path, format: str = "json"):
        """Export HTA report to file."""
        path = Path(path)
        
        if format == "json":
            path.write_text(json.dumps(report.to_dict(), indent=2))
        
        elif format == "html":
            html = self._generate_html_report(report)
            path.write_text(html)
    
    def _generate_html_report(self, report: HTAReport) -> str:
        """Generate HTML report."""
        top_kernels_rows = "".join(
            f"<tr><td>{k['name'][:50]}</td><td>{k['duration_us']:.0f}</td><td>{k['count']}</td></tr>"
            for k in report.top_kernels[:10]
        )
        recommendations_html = "".join(
            f'<div class="recommendation">{r}</div>' for r in report.recommendations
        )
        bottlenecks_html = (
            "<div class=\"section\">"
            "<h2>Bottlenecks</h2>"
            + "".join(f'<div class="bottleneck">{b}</div>' for b in report.bottlenecks)
            + "</div>"
        ) if report.bottlenecks else ""

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HTA Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0f172a;
            color: #e2e8f0;
        }}
        h1, h2 {{ margin-bottom: 15px; }}
        .section {{
            background: #1e293b;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .chart-container {{
            max-width: 400px;
            margin: 20px auto;
        }}
        .kernels-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .kernels-table th, .kernels-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}
        .kernels-table th {{
            color: #94a3b8;
        }}
        .recommendation {{
            background: #1e3a5f;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }}
        .bottleneck {{
            background: #3f1f1f;
            border-left: 4px solid #ef4444;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }}
    </style>
</head>
<body>
    <h1>📊 HTA Analysis Report</h1>
    
    <div class="section">
        <h2>Temporal Breakdown</h2>
        <div class="chart-container">
            <canvas id="temporalChart"></canvas>
        </div>
    </div>
    
    <div class="section">
        <h2>Top Kernels</h2>
        <table class="kernels-table">
            <tr>
                <th>Kernel</th>
                <th>Time (μs)</th>
                <th>Count</th>
            </tr>
            {top_kernels_rows}
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {recommendations_html}
    </div>
    
    {bottlenecks_html}
    
    <script>
        new Chart(document.getElementById('temporalChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Compute', 'Idle', 'Communication', 'Other'],
                datasets: [{{
                    data: [
                        {report.compute_time_pct:.1f},
                        {report.gpu_idle_time_pct:.1f},
                        {report.communication_time_pct:.1f},
                        {max(0, 100 - report.compute_time_pct - report.gpu_idle_time_pct - report.communication_time_pct):.1f}
                    ],
                    backgroundColor: ['#22c55e', '#64748b', '#f59e0b', '#8b5cf6']
                }}]
            }},
            options: {{
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e2e8f0' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>'''
