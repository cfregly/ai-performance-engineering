"""Compute serving latency and SLO goodput from actual request timestamps.

JSONL rows: request_id, arrival_s, token_timestamps_s, finish_s, status.
Status is completed, failed, or cancelled. Failed/partial requests remain in the
denominator and retain their emitted tokens. No duration is synthesized.
"""

import argparse
import json
import math
from pathlib import Path


def summarize(records, *, max_ttft_s, max_tpot_s, max_itl_s=None):
    limits = [max_ttft_s, max_tpot_s] + ([] if max_itl_s is None else [max_itl_s])
    if any(not math.isfinite(v) or v < 0 for v in limits):
        raise ValueError("SLO limits must be finite and nonnegative")
    if not records:
        raise ValueError("trace contains no request records")
    seen = set()
    requests = []
    for record in records:
        request_id = record["request_id"]
        if not isinstance(request_id, (str, int)) or request_id in seen:
            raise ValueError("request IDs must be unique strings or integers")
        seen.add(request_id)
        status = record["status"]
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"invalid status for request {request_id}")
        arrival, finish = record["arrival_s"], record["finish_s"]
        tokens = record["token_timestamps_s"]
        if not isinstance(tokens, list):
            raise ValueError("token_timestamps_s must be a list")
        timestamps = [arrival, *tokens, finish]
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in timestamps):
            raise ValueError("timestamps must be finite numbers")
        if any(a > b for a, b in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError(f"timestamps are not ordered for request {request_id}")
        if status == "completed" and not tokens:
            raise ValueError(f"completed request {request_id} has no output tokens")
        ttft = tokens[0] - arrival if tokens else None
        intervals = [b - a for a, b in zip(tokens, tokens[1:], strict=False)]
        tpot = sum(intervals) / len(intervals) if intervals else None
        max_itl = max(intervals) if intervals else None
        eligible = (
            status == "completed"
            and ttft <= max_ttft_s
            and (tpot is None or tpot <= max_tpot_s)
            and (max_itl_s is None or max_itl is None or max_itl <= max_itl_s)
        )
        requests.append(
            {
                **record,
                "token_timestamps_s": list(tokens),
                "request_id": request_id,
                "status": status,
                "arrival_s": arrival,
                "finish_s": finish,
                "emitted_tokens": len(tokens),
                "ttft_s": ttft,
                "tpot_s": tpot,
                "max_itl_s": max_itl,
                "meets_slo": eligible,
            }
        )
    elapsed = max(r["finish_s"] for r in requests) - min(r["arrival_s"] for r in requests)
    if elapsed <= 0:
        raise ValueError("trace observation interval must be positive")
    completed = [r for r in requests if r["status"] == "completed"]
    good = [r for r in requests if r["meets_slo"]]
    return {
        "observation_seconds": elapsed,
        "total_requests": len(requests),
        "completed_requests": len(completed),
        "failed_requests": sum(r["status"] == "failed" for r in requests),
        "cancelled_requests": sum(r["status"] == "cancelled" for r in requests),
        "slo_attainment_fraction": len(good) / len(requests),
        "goodput_requests_per_second": len(good) / elapsed,
        "goodput_tokens_per_second": sum(r["emitted_tokens"] for r in good) / elapsed,
        "emitted_tokens_per_second": sum(r["emitted_tokens"] for r in requests) / elapsed,
        "completed_tokens_per_second": sum(r["emitted_tokens"] for r in completed) / elapsed,
        "slos": {"max_ttft_s": max_ttft_s, "max_tpot_s": max_tpot_s, "max_itl_s": max_itl_s},
        "requests": requests,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="JSONL request timestamp records")
    parser.add_argument("--ttft-ms", type=float, required=True)
    parser.add_argument("--tpot-ms", type=float, required=True)
    parser.add_argument("--max-itl-ms", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    records = []
    for line_number, line in enumerate(args.trace.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("each record must be an object")
            records.append(record)
        except (json.JSONDecodeError, ValueError) as exc:
            parser.error(f"{args.trace}:{line_number}: {exc}")
    try:
        result = summarize(
            records,
            max_ttft_s=args.ttft_ms / 1000,
            max_tpot_s=args.tpot_ms / 1000,
            max_itl_s=None if args.max_itl_ms is None else args.max_itl_ms / 1000,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
