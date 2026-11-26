from __future__ import annotations

import typer

from .generator import generate_report, ReportConfig

app = typer.Typer(help="Generate PDF/HTML reports from benchmark data.")


@app.command()
def generate(
    input: str = typer.Argument(..., help="Input data (JSON file)"),
    output: str = typer.Option("report.pdf", "-o", "--output", help="Output file path"),
    format: str = typer.Option("pdf", "-f", "--format", help="pdf or html"),
    title: str = typer.Option("GPU Performance Report", "--title"),
    author: str = typer.Option("AI Performance Engineering", "--author"),
) -> None:
    config = ReportConfig(title=title, author=author)
    try:
        path = generate_report(input, output, format=format, config=config)
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"✅ Report generated: {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

