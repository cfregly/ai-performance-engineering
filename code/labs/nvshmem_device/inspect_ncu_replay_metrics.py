"""Read replay metadata from a real NCU report without treating missing as zero."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path)
parser.add_argument("--ncu-python", type=Path, required=True,
                    help="The installed Nsight Compute extras/python directory")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
sys.path.insert(0, str(args.ncu_python.resolve()))
import ncu_report


def metric_record(action, name):
    metric = action.metric_by_name(name)
    if metric is None:
        return {"present": False}
    if not metric.has_value():
        return {"present": True, "has_value": False}
    return {"present": True, "has_value": True, "value": metric.value(),
            "unit": metric.unit(), "rollup_operation": metric.rollup_operation()}


context = ncu_report.load_report(str(args.report.resolve()))
actions = []
for range_index in range(context.num_ranges()):
    result_range = context.range_by_idx(range_index)
    for action_index in range(result_range.num_actions()):
        action = result_range.action_by_idx(action_index)
        names = sorted(action.metric_names())
        selected = sorted({name for name in names if name.startswith("profiler__replayer")
                           or name.startswith("gpu__time_duration")})
        explicit = ("profiler__replayer_passes",
                    "profiler__replayer_passes_type_warmup",
                    "profiler__replayer_bytes_mem_backed_up.sum",
                    "profiler__replayer_bytes_mem_backed_up")
        actions.append({"range_index": range_index, "action_index": action_index,
                        "name": action.name(), "metric_names": names,
                        "replay_and_duration": {name: metric_record(action, name)
                                                for name in sorted(set(selected) | set(explicit))}})

result = {"report": str(args.report.resolve()),
          "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
          "ncu_report_module": ncu_report.__file__, "actions": actions}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"report": result["report"], "actions": len(actions),
                  "output": str(args.output.resolve())}))
