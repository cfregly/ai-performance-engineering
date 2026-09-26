"""Compare fixed-fleet vLLM and SGLang monolithic and P/D serving endpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import run_comparison
from .schema import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="Strict endpoint profile JSON")
    parser.add_argument(
        "--trace", type=Path, required=True, help="Deterministic request trace JSONL"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New artifact directory")
    parser.add_argument("--repeats", type=int, default=4, help="Measured matrix repeats, minimum 2")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup trace replays per arm")
    parser.add_argument("--ttft-ms", type=float, required=True, help="Maximum time to first token")
    parser.add_argument(
        "--tpot-ms", type=float, required=True, help="Maximum time per output token"
    )
    parser.add_argument("--max-itl-ms", type=float, help="Optional maximum inter-token latency")
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=10.0,
        help="Serving endpoint connection timeout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_comparison(
            profile_path=args.profile,
            trace_path=args.trace,
            output_dir=args.output_dir,
            repeats=args.repeats,
            warmups=args.warmups,
            ttft_ms=args.ttft_ms,
            tpot_ms=args.tpot_ms,
            max_itl_ms=args.max_itl_ms,
            connect_timeout_s=args.connect_timeout_s,
        )
    except (ConfigError, OSError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "comparison_valid": result["comparison_valid"],
                "valid_for_performance_claim": result["valid_for_performance_claim"],
                "summary": str(args.output_dir / "summary.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if result["comparison_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
