"""
Report Templates - Pre-defined report formats.

Provides specialized report types for different use cases:
- BenchmarkReport: Focus on benchmark results
- ProfileReport: Focus on profiling data
- OptimizationReport: Focus on LLM optimization results
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .generator import ReportGenerator, ReportConfig, BenchmarkData


@dataclass
class BenchmarkReport:
    """
    Generate a benchmark-focused report.
    
    Usage:
        report = BenchmarkReport(benchmarks)
        report.generate("benchmark_report.pdf")
    """
    
    benchmarks: List[BenchmarkData]
    title: str = "Benchmark Performance Report"
    include_charts: bool = True
    include_memory: bool = True
    
    def generate(self, output_path: str, format: str = "pdf") -> Path:
        """Generate the report."""
        config = ReportConfig(
            title=self.title,
            subtitle="Detailed Benchmark Analysis",
            include_charts=self.include_charts,
        )
        
        generator = ReportGenerator(config)
        
        if format == "pdf":
            return generator.generate_pdf(self.benchmarks, output_path)
        else:
            return generator.generate_html(self.benchmarks, output_path)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = len(self.benchmarks)
        speedups = [b.speedup for b in self.benchmarks if b.qualified]
        
        return {
            "total_benchmarks": total,
            "avg_speedup": sum(speedups) / len(speedups) if speedups else 0.0,
            "max_speedup": max(speedups, default=0.0),
            "min_speedup": min(speedups, default=0.0),
            "successful": sum(1 for s in speedups if s > 1.0),
            "regressed": sum(1 for s in speedups if s < 1.0),
        }


@dataclass
class ProfileReport:
    """
    Generate a profiling-focused report.
    
    Usage:
        report = ProfileReport(profile_data)
        report.generate("profile_report.pdf")
    """
    
    profile_data: Dict[str, Any]
    title: str = "GPU Profiling Report"
    include_flame_graph: bool = True
    include_timeline: bool = True
    include_memory: bool = True
    
    def generate(self, output_path: str, format: str = "pdf") -> Path:
        """Generate the report."""
        output_path = Path(output_path)
        
        html = self._generate_html()
        
        if format == "html":
            output_path.write_text(html)
            return output_path
        
        # For PDF, we need a generator
        config = ReportConfig(title=self.title)
        generator = ReportGenerator(config)
        
        if generator.pdf_method:
            if generator.pdf_method == "weasyprint":
                generator._generate_pdf_weasyprint(html, output_path)
            elif generator.pdf_method == "wkhtmltopdf":
                generator._generate_pdf_wkhtmltopdf(html, output_path)
            else:
                # Fall back to HTML
                output_path = output_path.with_suffix(".html")
                output_path.write_text(html)
        else:
            output_path = output_path.with_suffix(".html")
            output_path.write_text(html)
        
        return output_path
    
    def _generate_html(self) -> str:
        """Generate HTML content for profiling report."""
        
        # Extract data
        timing = self.profile_data.get("timing", {})
        memory = self.profile_data.get("memory", {})
        kernels = self.profile_data.get("kernels", [])
        bottlenecks = self.profile_data.get("bottlenecks", [])
        recommendations = self.profile_data.get("recommendations", [])
        
        # Generate kernel rows
        kernel_rows = ""
        for k in kernels[:20]:
            kernel_rows += f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #334155;">{k.get('name', 'unknown')[:50]}</td>
                <td style="padding: 10px; border-bottom: 1px solid #334155; text-align: right;">{k.get('duration_us', 0):.0f}</td>
                <td style="padding: 10px; border-bottom: 1px solid #334155; text-align: right;">{k.get('call_count', 0)}</td>
            </tr>
            """
        
        # Generate bottleneck list
        bottleneck_html = ""
        for b in bottlenecks:
            bottleneck_html += f'<div style="background: rgba(239,68,68,0.1); border-left: 3px solid #ef4444; padding: 10px; margin-bottom: 8px; border-radius: 0 6px 6px 0;">⚠️ {b}</div>'
        
        # Generate recommendations
        rec_html = ""
        for r in recommendations:
            rec_html += f'<div style="background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; padding: 10px; margin-bottom: 8px; border-radius: 0 6px 6px 0;">💡 {r}</div>'
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
            background: #0f172a;
            color: #ffffff;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ font-size: 28px; margin-bottom: 10px; color: #00f5d4; }}
        h2 {{ font-size: 18px; margin-top: 30px; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 1px solid #334155; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat {{ background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: 700; color: #00f5d4; }}
        .stat-label {{ font-size: 13px; color: #94a3b8; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: rgba(255,255,255,0.08); padding: 12px; text-align: left; font-size: 12px; color: #94a3b8; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #334155; text-align: center; font-size: 12px; color: #64748b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {self.title}</h1>
        <p style="color: #94a3b8;">Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{timing.get('total_ms', 0):.2f}ms</div>
                <div class="stat-label">Total Time</div>
            </div>
            <div class="stat">
                <div class="stat-value">{timing.get('cuda_ms', 0):.2f}ms</div>
                <div class="stat-label">GPU Time</div>
            </div>
            <div class="stat">
                <div class="stat-value">{memory.get('peak_mb', 0):.0f}MB</div>
                <div class="stat-label">Peak Memory</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(kernels)}</div>
                <div class="stat-label">Kernels</div>
            </div>
        </div>
        
        <h2>🔥 Top Kernels</h2>
        <table>
            <tr><th>Kernel</th><th style="text-align: right;">Time (μs)</th><th style="text-align: right;">Calls</th></tr>
            {kernel_rows if kernel_rows else '<tr><td colspan="3" style="padding: 20px; color: #64748b;">No kernel data available</td></tr>'}
        </table>
        
        <h2>⚠️ Bottlenecks</h2>
        {bottleneck_html if bottleneck_html else '<p style="color: #64748b;">No bottlenecks identified</p>'}
        
        <h2>💡 Recommendations</h2>
        {rec_html if rec_html else '<p style="color: #64748b;">No recommendations</p>'}
        
        <div class="footer">
            AI Performance Engineering • GPU Profiling Report
        </div>
    </div>
</body>
</html>"""


@dataclass
class OptimizationReport:
    """
    Generate an LLM optimization-focused report.
    
    Usage:
        report = OptimizationReport(optimization_results)
        report.generate("optimization_report.pdf")
    """
    
    results: List[Dict[str, Any]]
    title: str = "LLM Optimization Report"
    include_code: bool = True
    include_explanations: bool = True
    
    def generate(self, output_path: str, format: str = "pdf") -> Path:
        """Generate the report."""
        output_path = Path(output_path)
        
        html = self._generate_html()
        
        if format == "html":
            output_path.write_text(html)
            return output_path
        
        config = ReportConfig(title=self.title)
        generator = ReportGenerator(config)
        
        if generator.pdf_method:
            if generator.pdf_method == "weasyprint":
                generator._generate_pdf_weasyprint(html, output_path)
            elif generator.pdf_method == "wkhtmltopdf":
                generator._generate_pdf_wkhtmltopdf(html, output_path)
            else:
                output_path = output_path.with_suffix(".html")
                output_path.write_text(html)
        else:
            output_path = output_path.with_suffix(".html")
            output_path.write_text(html)
        
        return output_path
    
    def _generate_html(self) -> str:
        """Generate HTML content for optimization report."""
        
        # Summary stats
        total = len(self.results)
        successful = sum(1 for r in self.results if r.get("success", False))
        avg_speedup = sum(r.get("speedup", 1.0) for r in self.results) / max(total, 1)
        
        # All techniques used
        all_techniques = set()
        for r in self.results:
            all_techniques.update(r.get("techniques", []))
        
        # Generate result rows
        result_rows = ""
        for r in self.results:
            status = "✅" if r.get("success", False) else "❌"
            speedup = r.get("speedup", 1.0)
            speedup_color = "#22c55e" if speedup > 1.1 else "#f59e0b" if speedup > 1.0 else "#ef4444"
            techniques = ", ".join(r.get("techniques", [])[:3]) or "-"
            
            result_rows += f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 20px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-weight: 600;">{status} {r.get('name', 'Unknown')}</div>
                    <div style="font-size: 24px; font-weight: 700; color: {speedup_color};">{speedup:.2f}x</div>
                </div>
                <div style="font-size: 13px; color: #94a3b8; margin-bottom: 10px;">
                    {r.get('original_time_ms', 0):.2f}ms → {r.get('optimized_time_ms', 0):.2f}ms
                </div>
                <div style="font-size: 13px;">
                    <strong>Techniques:</strong> {techniques}
                </div>
                {f'<div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.02); border-radius: 6px; font-size: 13px; color: #94a3b8;">{r.get("explanation", "")[:200]}</div>' if r.get("explanation") and self.include_explanations else ''}
            </div>
            """
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
            background: #0f172a;
            color: #ffffff;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ font-size: 28px; margin-bottom: 10px; color: #9d4edd; }}
        h2 {{ font-size: 18px; margin-top: 30px; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 1px solid #334155; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat {{ background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: 700; color: #9d4edd; }}
        .stat-label {{ font-size: 13px; color: #94a3b8; margin-top: 5px; }}
        .techniques {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 15px 0; }}
        .technique {{ background: rgba(157,78,221,0.2); color: #c084fc; padding: 4px 10px; border-radius: 20px; font-size: 12px; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #334155; text-align: center; font-size: 12px; color: #64748b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ {self.title}</h1>
        <p style="color: #94a3b8;">Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Optimizations</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color: #22c55e;">{successful}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat">
                <div class="stat-value">{avg_speedup:.2f}x</div>
                <div class="stat-label">Avg Speedup</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(all_techniques)}</div>
                <div class="stat-label">Techniques Used</div>
            </div>
        </div>
        
        <h2>🔧 Techniques Applied</h2>
        <div class="techniques">
            {''.join(f'<span class="technique">{t}</span>' for t in sorted(all_techniques))}
        </div>
        
        <h2>📋 Optimization Results</h2>
        {result_rows if result_rows else '<p style="color: #64748b;">No optimization results available</p>'}
        
        <div class="footer">
            AI Performance Engineering • LLM Optimization Report
        </div>
    </div>
</body>
</html>"""


def generate_summary_report(
    benchmark_data: Dict[str, Any],
    profile_data: Optional[Dict[str, Any]] = None,
    optimization_data: Optional[List[Dict]] = None,
    output_path: str = "summary_report.pdf",
) -> Path:
    """
    Generate a comprehensive summary report combining all data types.
    
    Args:
        benchmark_data: Benchmark results
        profile_data: Optional profiling data
        optimization_data: Optional LLM optimization results
        output_path: Output file path
        
    Returns:
        Path to generated report
    """
    config = ReportConfig(
        title="AI Performance Engineering Report",
        subtitle="Comprehensive Benchmark, Profiling, and Optimization Summary",
    )
    
    generator = ReportGenerator(config)
    return generator.generate_pdf(benchmark_data, output_path)



