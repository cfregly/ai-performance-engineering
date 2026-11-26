# 📕 Report Generation

Generate polished **PDF and HTML reports** from benchmark data, profiling results, and optimization summaries.

## Features

- **Multiple formats**: PDF, HTML, CSV
- **Beautiful styling**: Dark theme, professional layout
- **Customizable**: Titles, colors, content sections
- **Multiple report types**: Benchmark, Profiling, Optimization
- **Python API**: Generate reports from benchmark data
- **CLI support**: Command-line report generation

## Quick Start

### From CLI

```bash
# Generate PDF from JSON data
python -m cli.aisp bench report --data-file benchmark_test_results.json --output report.pdf

# Generate HTML report
python -m cli.aisp bench report --data-file benchmark_test_results.json --output report.html --format html

# Custom title
python -m cli.aisp bench report --data-file benchmark_test_results.json --output report.pdf --title "Q4 Performance Report"
```

### From Python

```python
from core.analysis.reporting import ReportGenerator, ReportConfig

# Basic usage
generator = ReportGenerator()
generator.generate_pdf(benchmark_data, "report.pdf")

# With custom config
config = ReportConfig(
    title="My Performance Report",
    subtitle="GPU Optimization Results",
    author="Engineering Team",
    include_charts=True,
)

generator = ReportGenerator(config)
generator.generate_pdf(data, "custom_report.pdf")
```

Benchmark rankings and summary statistics use successful baseline and optimization
pairs. Historical results that contain only a recorded speedup can also qualify,
with unrecorded times shown as unavailable. Discovered scans select the latest
qualified result per target and show any later rejection beside it. Rejected rows
and baseline-only runs appear in **Run Diagnostics**. Missing or malformed
explicit inputs produce an error. In discovered files, valid rows remain
available and malformed records appear under **Input Errors**.

## Report Types

### BenchmarkReport

Focus on benchmark results and speedups:

```python
from core.analysis.reporting import BenchmarkReport, BenchmarkData

benchmarks = [
    BenchmarkData(
        name="matmul_tiling",
        chapter="ch05",
        baseline_time_ms=100.0,
        optimized_time_ms=25.0,
        speedup=4.0,
        techniques=["torch.compile", "tiling"],
    ),
    # ... more benchmarks
]

report = BenchmarkReport(benchmarks, title="Benchmark Results")
report.generate("benchmark_report.pdf")

# Get summary stats
print(report.summary())
# {'total_benchmarks': 10, 'avg_speedup': 2.5, 'max_speedup': 4.0, ...}
```

### ProfileReport

Focus on GPU profiling data:

```python
from core.analysis.reporting import ProfileReport

profile_data = {
    "timing": {"total_ms": 100, "cuda_ms": 90, "cpu_ms": 10},
    "memory": {"peak_mb": 1024, "allocated_mb": 800},
    "kernels": [
        {"name": "matmul", "duration_us": 5000, "call_count": 100},
        {"name": "softmax", "duration_us": 1000, "call_count": 100},
    ],
    "bottlenecks": ["High memory bandwidth usage"],
    "recommendations": ["Consider using torch.compile"],
}

report = ProfileReport(profile_data)
report.generate("profile_report.pdf")
```

### OptimizationReport

Focus on LLM optimization results:

```python
from core.analysis.reporting import OptimizationReport

results = [
    {
        "name": "model_forward",
        "success": True,
        "speedup": 2.5,
        "original_time_ms": 100,
        "optimized_time_ms": 40,
        "techniques": ["torch.compile", "mixed_precision"],
        "explanation": "Applied kernel fusion and BF16 precision...",
    },
]

report = OptimizationReport(results)
report.generate("optimization_report.pdf")
```

## Configuration

```python
from core.analysis.reporting import ReportConfig

config = ReportConfig(
    # Content
    title="GPU Performance Report",
    subtitle="Benchmark Analysis",
    author="AI Performance Engineering",
    
    # Styling
    primary_color="#00f5d4",
    secondary_color="#9d4edd",
    background_color="#0f172a",
    
    # Sections
    include_summary=True,
    include_charts=True,
    include_details=True,
    include_recommendations=True,
    include_code_snippets=False,
    
    # Page
    page_size="A4",  # or "letter"
    orientation="portrait",
)
```

## PDF Libraries

The report generator supports multiple PDF backends:

| Library | Install | Notes |
|---------|---------|-------|
| **WeasyPrint** (recommended) | `pip install weasyprint` | Best quality, CSS support |
| **ReportLab** | `pip install reportlab` | Pure Python, no dependencies |
| **wkhtmltopdf** | System install | Requires external binary |

If no PDF library is available, reports are saved as HTML.

```bash
# Install WeasyPrint (recommended)
pip install weasyprint

# Or ReportLab
pip install reportlab
```

## API Reference

### ReportGenerator

```python
class ReportGenerator:
    def __init__(self, config: ReportConfig = None)
    
    def generate_pdf(
        self,
        data: Union[Dict, List[BenchmarkData]],
        output_path: Union[str, Path],
        template: str = "default",
    ) -> Path
    
    def generate_html(
        self,
        data: Union[Dict, List[BenchmarkData]],
        output_path: Union[str, Path],
        template: str = "default",
    ) -> Path
    
```

### BenchmarkData

```python
@dataclass
class BenchmarkData:
    name: str
    chapter: str = ""
    baseline_time_ms: float = 0
    optimized_time_ms: float = 0
    speedup: float = 1.0
    techniques: List[str] = field(default_factory=list)
    memory_baseline_mb: float = 0
    memory_optimized_mb: float = 0
    status: str = "success"
    notes: str = ""
```

## Examples

### Generate Report from JSON File

```python
from core.analysis.reporting import generate_report

generate_report(
    "benchmark_test_results.json",
    "ch05_report.pdf"
)
```

### Batch Report Generation

```python
from pathlib import Path
from core.analysis.reporting import ReportGenerator, ReportConfig
import json

generator = ReportGenerator()

for results_file in Path("artifacts").rglob("benchmark_test_results.json"):
    chapter = results_file.parent.parent.name
    
    with open(results_file) as f:
        data = json.load(f)
    
    generator.generate_pdf(
        data,
        f"reports/{chapter}_report.pdf"
    )
```

## Architecture

```
analysis/reporting/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── generator.py         # ReportGenerator, ReportConfig
├── templates.py         # BenchmarkReport, ProfileReport, OptimizationReport
└── README.md            # This file
```
