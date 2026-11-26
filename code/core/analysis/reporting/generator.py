"""
Report Generator - Create PDF and HTML reports from benchmark data.

Supports multiple output formats and customizable templates.
"""

import base64
import io
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.analysis.performance_analyzer import (
    has_success_status,
    is_finite_measurement,
    is_qualified_benchmark,
    load_benchmark_data,
    read_benchmark_json,
)

# Try to import PDF libraries
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "GPU Performance Report"
    subtitle: str = "Benchmark Analysis and Optimization Results"
    author: str = "AI Performance Engineering"
    logo_path: Optional[str] = None
    
    # Styling
    primary_color: str = "#00f5d4"
    secondary_color: str = "#9d4edd"
    accent_color: str = "#f72585"
    background_color: str = "#0f172a"
    text_color: str = "#ffffff"
    
    # Content options
    include_summary: bool = True
    include_charts: bool = True
    include_details: bool = True
    include_recommendations: bool = True
    include_code_snippets: bool = False
    
    # Page settings
    page_size: str = "A4"  # "A4" or "letter"
    orientation: str = "portrait"


@dataclass
class BenchmarkData:
    """Benchmark data for report generation."""
    name: str
    chapter: str = ""
    baseline_time_ms: Optional[float] = 0
    optimized_time_ms: Optional[float] = 0
    speedup: Optional[float] = 1.0
    techniques: List[str] = field(default_factory=list)
    memory_baseline_mb: float = 0
    memory_optimized_mb: float = 0
    status: str = "success"
    notes: str = ""
    source_file: str = ""
    run_id: str = ""
    pair_role: str = ""
    chapter_alignment: str = ""
    chapter_native_exemplar: Optional[bool] = None

    @property
    def qualified(self) -> bool:
        return is_qualified_benchmark({"status": self.status, "speedup": self.speedup})


def _speedup_label(speedup: Optional[float], precision: int = 2) -> str:
    return f"{speedup:.{precision}f}x" if speedup is not None else "n/a"


def _time_label(time_ms: Optional[float]) -> str:
    return f"{time_ms:.2f}" if time_ms is not None else "n/a"


def _report_summary(benchmarks: List[BenchmarkData]) -> tuple:
    qualified = sorted((b for b in benchmarks if b.qualified), key=lambda b: b.speedup, reverse=True)
    diagnostics = [b for b in benchmarks if not b.qualified]
    average = sum(b.speedup for b in qualified) / len(qualified) if qualified else None
    best = qualified[0].speedup if qualified else None
    successful = sum(b.speedup > 1.0 for b in qualified)
    return qualified, diagnostics, average, best, successful


class ReportGenerator:
    """
    Generate PDF and HTML reports from benchmark data.
    
    Usage:
        generator = ReportGenerator(config=ReportConfig(title="My Report"))
        
        # From benchmark data
        generator.generate_pdf(benchmarks, "report.pdf")
        
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check available PDF libraries."""
        self.pdf_method = None
        
        if WEASYPRINT_AVAILABLE:
            self.pdf_method = "weasyprint"
        elif REPORTLAB_AVAILABLE:
            self.pdf_method = "reportlab"
        else:
            # Check for wkhtmltopdf
            try:
                subprocess.run(["wkhtmltopdf", "--version"], capture_output=True)
                self.pdf_method = "wkhtmltopdf"
            except FileNotFoundError:
                pass
    
    def generate_pdf(
        self,
        data: Union[Dict[str, Any], List[BenchmarkData]],
        output_path: Union[str, Path],
        template: str = "default",
    ) -> Path:
        """
        Generate a PDF report.
        
        Args:
            data: Benchmark data (dict or list of BenchmarkData)
            output_path: Output PDF path
            template: Template name ("default", "minimal", "detailed")
            
        Returns:
            Path to generated PDF
        """
        output_path = Path(output_path)
        
        # Normalize data
        if isinstance(data, dict):
            benchmarks = self._parse_dict_data(data)
        else:
            benchmarks = data
        
        # Generate HTML first
        html_content = self._generate_html(benchmarks, template)
        
        # Convert to PDF
        if self.pdf_method == "weasyprint":
            self._generate_pdf_weasyprint(html_content, output_path)
        elif self.pdf_method == "reportlab":
            self._generate_pdf_reportlab(benchmarks, output_path)
        elif self.pdf_method == "wkhtmltopdf":
            self._generate_pdf_wkhtmltopdf(html_content, output_path)
        else:
            # Fallback: save as HTML
            html_path = output_path.with_suffix(".html")
            html_path.write_text(html_content)
            print(f"⚠️ No PDF library available. Saved as HTML: {html_path}")
            print("Install weasyprint: pip install weasyprint")
            return html_path
        
        return output_path
    
    def generate_html(
        self,
        data: Union[Dict[str, Any], List[BenchmarkData]],
        output_path: Union[str, Path],
        template: str = "default",
    ) -> Path:
        """Generate an HTML report."""
        output_path = Path(output_path)
        
        if isinstance(data, dict):
            benchmarks = self._parse_dict_data(data)
        else:
            benchmarks = data
        
        html_content = self._generate_html(benchmarks, template)
        output_path.write_text(html_content)
        
        return output_path
    
    def _parse_dict_data(self, data: Dict[str, Any]) -> List[BenchmarkData]:
        """Parse dictionary data into BenchmarkData objects."""
        benchmarks = []
        rows = data.get("benchmarks")
        diagnostics = data.get("diagnostics", [])
        if not isinstance(rows, list) or not isinstance(diagnostics, list):
            raise ValueError("Expected 'benchmarks' and optional 'diagnostics' lists in report input")

        for index, item in enumerate(rows + diagnostics):
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ValueError(f"Invalid benchmark at report row {index}")
            for key in ("baseline_time_ms", "optimized_time_ms", "speedup"):
                value = item.get(key)
                if value is not None and not is_finite_measurement(value):
                    raise ValueError(f"Invalid {key} for report row {item['name']}")
            status = item.get("status", "success")
            chapter = item.get("chapter", "")
            techniques = item.get("techniques", [])
            if not isinstance(status, str) or not isinstance(chapter, str):
                raise ValueError(f"Invalid status or chapter for report row {item['name']}")
            if not isinstance(techniques, list) or any(not isinstance(technique, str) for technique in techniques):
                raise ValueError(f"Expected techniques list for report row {item['name']}")
            if has_success_status(status) and (
                index >= len(rows) or not is_qualified_benchmark(item)
            ):
                status = "unqualified"
            benchmarks.append(BenchmarkData(
                name=item["name"],
                chapter=chapter,
                baseline_time_ms=item.get("baseline_time_ms"),
                optimized_time_ms=item.get("optimized_time_ms"),
                speedup=item.get("speedup"),
                techniques=techniques,
                memory_baseline_mb=item.get("baseline_memory_mb", 0),
                memory_optimized_mb=item.get("optimized_memory_mb", 0),
                status=status,
                notes=" ".join(str(value) for value in (item.get("story_note") or item.get("notes"), item.get("error")) if value),
                source_file=str(item.get("source_file") or ""),
                run_id=str(item.get("run_id") or ""),
                pair_role=item.get("pair_role", ""),
                chapter_alignment=item.get("chapter_alignment", ""),
                chapter_native_exemplar=item.get("chapter_native_exemplar"),
            ))

        return benchmarks
    
    def _generate_html(self, benchmarks: List[BenchmarkData], template: str) -> str:
        """Generate HTML content for the report."""
        
        total = len(benchmarks)
        qualified, diagnostics, avg_speedup, best_speedup, successful = _report_summary(benchmarks)

        chapters: Dict[str, List[BenchmarkData]] = {}
        for b in qualified:
            chapters.setdefault(b.chapter or "Other", []).append(b)

        benchmark_rows = ""
        for b in qualified[:50]:
            status_color = "#22c55e" if b.speedup > 1.1 else "#f59e0b" if b.speedup > 1.0 else "#ef4444"
            techniques_str = ", ".join(b.techniques[:3]) if b.techniques else "-"
            note_html = ""
            if b.notes:
                note_html = (
                    f'<div style="margin-top: 6px; font-size: 11px; color: #94a3b8;">{escape(b.notes)}</div>'
                )
            benchmark_rows += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #334155;">{escape(b.name)}{note_html}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155;">{escape(b.chapter)}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: right;">{_time_label(b.baseline_time_ms)}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: right;">{_time_label(b.optimized_time_ms)}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; text-align: right; color: {status_color}; font-weight: bold;">{_speedup_label(b.speedup)}</td>
                <td style="padding: 12px; border-bottom: 1px solid #334155; font-size: 12px;">{escape(techniques_str)}</td>
            </tr>
            """

        chapter_summary = ""
        for ch, ch_benchmarks in sorted(chapters.items()):
            ch_avg = sum(b.speedup for b in ch_benchmarks) / len(ch_benchmarks)
            ch_best = max(b.speedup for b in ch_benchmarks)
            chapter_summary += f"""
            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                <div style="font-weight: 600; margin-bottom: 5px;">{escape(ch)}</div>
                <div style="display: flex; gap: 20px; font-size: 14px; color: #94a3b8;">
                    <span>{len(ch_benchmarks)} benchmarks</span>
                    <span>Avg: {ch_avg:.2f}x</span>
                    <span>Best: {ch_best:.2f}x</span>
                </div>
            </div>
            """

        top_performers = ""
        for b in qualified[:5]:
            top_performers += f"""
            <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #334155;">
                <span>{escape(b.name)}</span>
                <span style="color: #22c55e; font-weight: bold;">{_speedup_label(b.speedup)}</span>
            </div>
            """

        diagnostics_section = ""
        if diagnostics:
            diagnostic_rows = []
            for b in diagnostics:
                details = " · ".join(part for part in (b.notes, b.run_id, b.source_file) if part)
                diagnostic_rows.append(
                    f"<tr><td>{escape(b.chapter)}:{escape(b.name)}</td>"
                    f"<td>{escape(b.status)}</td><td>{_speedup_label(b.speedup)}</td>"
                    f"<td>{escape(details)}</td></tr>"
                )
            diagnostics_section = f"""
            <div class="section">
                <h2 class="section-title">Run Diagnostics</h2>
                <table>
                    <thead><tr><th>Benchmark</th><th>Status</th><th>Recorded Speedup</th><th>Details</th></tr></thead>
                    <tbody>{"".join(diagnostic_rows)}</tbody>
                </table>
            </div>
            """

        # Generate full HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.config.title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: {self.config.background_color};
            color: {self.config.text_color};
            line-height: 1.6;
            padding: 40px;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 2px solid {self.config.primary_color};
        }}
        
        .header h1 {{
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(135deg, {self.config.primary_color}, {self.config.secondary_color});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header .subtitle {{
            font-size: 16px;
            color: #94a3b8;
        }}
        
        .header .date {{
            font-size: 14px;
            color: #64748b;
            margin-top: 10px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .summary-card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        
        .summary-value {{
            font-size: 36px;
            font-weight: 700;
            color: {self.config.primary_color};
        }}
        
        .summary-label {{
            font-size: 14px;
            color: #94a3b8;
            margin-top: 5px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #334155;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.02);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th {{
            background: rgba(255,255,255,0.05);
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
        }}
        
        .card-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #334155;
            text-align: center;
            font-size: 12px;
            color: #64748b;
        }}
        
        @media print {{
            body {{ 
                background: white; 
                color: #1e293b;
                padding: 20px;
            }}
            .summary-card, .card {{
                background: #f8fafc;
                border-color: #e2e8f0;
            }}
            .summary-value {{
                color: #0f172a;
            }}
            th {{
                background: #f1f5f9;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.config.title}</h1>
            <div class="subtitle">{self.config.subtitle}</div>
            <div class="date">Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}</div>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-value">{total}</div>
                <div class="summary-label">Result Rows</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" style="color: #22c55e;">{_speedup_label(best_speedup, 1)}</div>
                <div class="summary-label">Best Speedup</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{_speedup_label(avg_speedup)}</div>
                <div class="summary-label">Average Speedup</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{successful}/{len(qualified)}</div>
                <div class="summary-label">Optimized</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 Qualified Benchmark Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Chapter</th>
                        <th style="text-align: right;">Baseline (ms)</th>
                        <th style="text-align: right;">Optimized (ms)</th>
                        <th style="text-align: right;">Speedup</th>
                        <th>Techniques</th>
                    </tr>
                </thead>
                <tbody>
                    {benchmark_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">🏆 Top Performers</div>
                    {top_performers}
                </div>
                <div class="card">
                    <div class="card-title">📁 By Chapter</div>
                    {chapter_summary}
                </div>
            </div>
        </div>
        
        {diagnostics_section}

        <div class="footer">
            <p>{self.config.author} • Generated with AI Performance Engineering Tools</p>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_pdf_weasyprint(self, html_content: str, output_path: Path):
        """Generate PDF using WeasyPrint."""
        html = HTML(string=html_content)
        html.write_pdf(str(output_path))
    
    def _generate_pdf_wkhtmltopdf(self, html_content: str, output_path: Path):
        """Generate PDF using wkhtmltopdf."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            html_path = f.name
        
        try:
            subprocess.run(
                ["wkhtmltopdf", "--enable-local-file-access", html_path, str(output_path)],
                capture_output=True,
                check=True,
            )
        finally:
            Path(html_path).unlink()
    
    def _generate_pdf_reportlab(self, benchmarks: List[BenchmarkData], output_path: Path):
        """Generate PDF using ReportLab."""
        
        page_size = A4 if self.config.page_size == "A4" else letter
        doc = SimpleDocTemplate(str(output_path), pagesize=page_size)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            textColor=colors.HexColor("#0f172a"),
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=30,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor("#1e293b"),
        )
        
        story = []
        
        # Title
        story.append(Paragraph(self.config.title, title_style))
        story.append(Paragraph(
            f"{self.config.subtitle}<br/>Generated on {datetime.now().strftime('%B %d, %Y')}",
            subtitle_style
        ))
        
        # Summary stats
        total = len(benchmarks)
        qualified, diagnostics, avg_speedup, best_speedup, successful = _report_summary(benchmarks)
        
        summary_data = [
            ['Result Rows', 'Best Speedup', 'Average Speedup', 'Optimized'],
            [str(total), f'{_speedup_label(best_speedup, 1)}', f'{_speedup_label(avg_speedup)}', f'{successful}/{len(qualified)}'],
        ]
        
        summary_table = Table(summary_data, colWidths=[120, 100, 100, 100])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#64748b")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, 1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 30))
        
        # Benchmark table
        story.append(Paragraph("Qualified Benchmark Results", heading_style))
        
        table_data = [['Benchmark', 'Baseline (ms)', 'Optimized (ms)', 'Speedup']]
        
        for b in qualified[:30]:
            display_name = b.name
            if b.pair_role == "comparison" and b.chapter_alignment == "supplementary":
                display_name = f"{display_name} [supplementary comparison]"
            table_data.append([
                display_name[:56],
                _time_label(b.baseline_time_ms),
                _time_label(b.optimized_time_ms),
                _speedup_label(b.speedup),
            ])
        
        bench_table = Table(table_data, colWidths=[200, 80, 80, 70])
        bench_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        
        story.append(bench_table)
        
        if diagnostics:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Run Diagnostics", heading_style))
            for b in diagnostics:
                details = " | ".join(part for part in (b.notes, b.run_id, b.source_file) if part)
                story.append(Paragraph(
                    escape(f"{b.chapter}:{b.name} | {b.status} | Recorded speedup {_speedup_label(b.speedup)} | {details}"),
                    styles["Normal"],
                ))

        # Build PDF
        doc.build(story)


def generate_report(
    data: Union[Dict, str, Path],
    output_path: Union[str, Path],
    format: str = "pdf",
    config: Optional[ReportConfig] = None,
) -> Path:
    """
    Convenience function to generate a report.
    
    Args:
        data: Benchmark data (dict or JSON file path)
        output_path: Output file path
        format: "pdf" or "html"
        config: Report configuration
        
    Returns:
        Path to generated report
    """
    generator = ReportGenerator(config=config)
    
    if isinstance(data, (str, Path)):
        data_path = Path(data)
        data = read_benchmark_json(data_path)
        if "results" in data and "benchmarks" not in data:
            data = load_benchmark_data(data_file=data_path)
        if not isinstance(data.get("benchmarks"), list):
            raise ValueError(f"Expected 'benchmarks' list in report input {data_path}")

    if format == "pdf":
        return generator.generate_pdf(data, output_path)
    else:
        return generator.generate_html(data, output_path)


# CLI support
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PDF/HTML reports from benchmark data")
    parser.add_argument("input", help="Input data (JSON file)")
    parser.add_argument("-o", "--output", default="report.pdf", help="Output file path")
    parser.add_argument("-f", "--format", choices=["pdf", "html"], default="pdf", help="Output format")
    parser.add_argument("--title", default="GPU Performance Report", help="Report title")
    parser.add_argument("--author", default="AI Performance Engineering", help="Report author")
    
    args = parser.parse_args()
    
    config = ReportConfig(title=args.title, author=args.author)
    
    output_path = generate_report(
        args.input,
        args.output,
        format=args.format,
        config=config,
    )
    
    print(f"✅ Report generated: {output_path}")


if __name__ == "__main__":
    main()
