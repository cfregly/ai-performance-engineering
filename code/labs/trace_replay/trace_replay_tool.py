"""Capture caller tensor values and replay fixed native operator pairs."""

import argparse
import json
from pathlib import Path

from labs.trace_replay.runner import capture_plan, replay_bundle
from labs.trace_replay.schema import load_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Retain actual tensors from an explicit generated-input recipe")
    capture.add_argument("--plan", required=True, type=Path)
    capture.add_argument("--output-dir", required=True, type=Path)
    capture.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    validate = commands.add_parser("validate", help="Validate the immutable manifest and every retained tensor")
    validate.add_argument("--bundle", required=True, type=Path)
    replay = commands.add_parser("replay", help="Run the existing native paired harness for each event")
    replay.add_argument("--bundle", required=True, type=Path)
    replay.add_argument("--output-dir", required=True, type=Path)
    replay.add_argument("--validity-profile", choices=["strict", "portable"], default="strict")
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            bundle = capture_plan(args.plan, args.output_dir, args.device)
            result = {"bundle": str(bundle.root), "bundle_sha256": bundle.sha256, "capture_backend": bundle.manifest["capture_backend"], "events": [{"event_id": event["event_id"], "status": event["status"]} for event in bundle.events]}
            success = all(event["status"] == "completed" for event in bundle.events)
        elif args.command == "validate":
            bundle = load_bundle(args.bundle)
            result = {"status": "valid", "bundle_sha256": bundle.sha256, "events": len(bundle.events), "capture_backend": bundle.manifest["capture_backend"]}
            success = True
        else:
            result = replay_bundle(args.bundle, args.output_dir, args.validity_profile)
            success = result["status"] == "completed"
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
