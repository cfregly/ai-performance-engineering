"""vLLM monitoring bundle capturing the full v1 metric surface.

This is a chapter tool (not a comparable baseline/optimized benchmark).
Run via `python -m cli.aisp tools vllm-monitoring -- --outdir ...`.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import List

from ch16.monitoring_bundle import MonitoringBundle, write_bundle  # noqa: E402
from ch16.monitoring_config import MetricNames, AlertThresholds, load_monitoring_overrides  # noqa: E402


def build_vllm_monitoring_bundle(metrics: MetricNames, thresholds: AlertThresholds) -> MonitoringBundle:
    """
    Full-fidelity bundle: per-model TTFT/prefill/decode, queue churn, KV cache,
    CUDA graph mode, and EngineCore/scheduler error hooks.
    """
    scrape_config = textwrap.dedent(
        """
        scrape_configs:
          - job_name: vllm
            metrics_path: /metrics
            scrape_interval: 5s
            static_configs:
              - targets: ['localhost:8001']
                labels:
                  service: vllm
                  cluster: prod-a
        """
    ).strip()

    m = metrics
    recording_rules = textwrap.dedent(
        f"""
        groups:
          - name: vllm-v1-recording-rules
            interval: 15s
            rules:
              # Latency histograms (TTFT, prefill, decode, end-to-end, inter-token)
              - record: vllm:ttft_seconds:p50
                expr: |
                  histogram_quantile(
                    0.50,
                    sum by (le, instance, model_name) (
                      rate({m.ttft_hist}[5m])
                    )
                  )
              - record: vllm:ttft_seconds:p90
                expr: |
                  histogram_quantile(
                    0.90,
                    sum by (le, instance, model_name) (
                      rate({m.ttft_hist}[5m])
                    )
                  )
              - record: vllm:ttft_seconds:p99
                expr: |
                  histogram_quantile(
                    0.99,
                    sum by (le, instance, model_name) (
                      rate({m.ttft_hist}[5m])
                    )
                  )

              - record: vllm:prefill_seconds:p50
                expr: |
                  histogram_quantile(
                    0.50,
                    sum by (le, instance, model_name) (
                      rate({m.prefill_hist}[5m])
                    )
                  )
              - record: vllm:prefill_seconds:p90
                expr: |
                  histogram_quantile(
                    0.90,
                    sum by (le, instance, model_name) (
                      rate({m.prefill_hist}[5m])
                    )
                  )

              - record: vllm:decode_seconds:p50
                expr: |
                  histogram_quantile(
                    0.50,
                    sum by (le, instance, model_name) (
                      rate({m.decode_hist}[5m])
                    )
                  )
              - record: vllm:decode_seconds:p90
                expr: |
                  histogram_quantile(
                    0.90,
                    sum by (le, instance, model_name) (
                      rate({m.decode_hist}[5m])
                    )
                  )

              - record: vllm:e2e_seconds:p90
                expr: |
                  histogram_quantile(
                    0.90,
                    sum by (le, instance, model_name) (
                      rate({m.e2e_hist}[5m])
                    )
                  )

              - record: vllm:time_per_output_token_seconds:p90
                expr: |
                  histogram_quantile(
                    0.90,
                    sum by (le, instance, model_name) (
                      rate({m.inter_token_hist}[5m])
                    )
                  )

              # Throughput and queue health
              - record: vllm:req_rate:rps
                expr: |
                  sum by (instance, model_name) (
                    rate({m.request_success}[1m])
                  )

              - record: vllm:active_requests
                expr: |
                  sum by (instance, model_name) ({m.num_running})

              - record: vllm:waiting_requests
                expr: |
                  sum by (instance, model_name) ({m.num_waiting})

              - record: vllm:finished_requests_rate
                expr: |
                  sum by (instance, model_name) (
                    rate({m.request_success}[1m])
                  )

              - record: vllm:finished_to_active_ratio
                expr: |
                  vllm:finished_requests_rate
                  /
                  (vllm:active_requests + 1)

              # KV cache occupancy (0..1 exposer converted to percent)
              - record: vllm:kv_cache_usage_percent
                expr: |
                  100 * avg by (instance, model_name) (
                    {m.gpu_cache_usage}
                  )

              # CUDA graph mode info (if exported)
              - record: vllm:cuda_graph_mode
                expr: |
                  max by (instance, model_name, mode) ({m.cudagraph_mode_info})
        """
    ).strip()

    t = thresholds
    alert_rules = textwrap.dedent(
        f"""
        groups:
          - name: vllm-v1-alerts
            rules:
              - alert: VLLMHighTTFTP90
                expr: vllm:ttft_seconds:p90 > {t.ttft_p90_warn}
                for: 5m
                labels:
                  severity: warning
                annotations:
                  summary: "TTFT p90 high ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    Time to first token p90 is {{{{ $value }}}}s.
                    Check scheduler load, prefill batch sizes, and KV cache usage.

              - alert: VLLMHighTTFTP99
                expr: vllm:ttft_seconds:p99 > {t.ttft_p99_crit}
                for: 5m
                labels:
                  severity: critical
                annotations:
                  summary: "TTFT p99 degraded ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    TTFT p99 > {t.ttft_p99_crit}s. Often caused by GPU saturation, long prompts, or cache fragmentation.

              - alert: VLLMHighPrefillLatency
                expr: vllm:prefill_seconds:p90 > {t.prefill_p90_warn}
                for: 10m
                labels:
                  severity: warning
                annotations:
                  summary: "Prefill latency p90 elevated ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    Prefill p90 exceeds {t.prefill_p90_warn}s. Prompts may be too long or batches oversized.

              - alert: VLLMHighDecodeLatency
                expr: vllm:decode_seconds:p90 > {t.decode_p90_warn}
                for: 10m
                labels:
                  severity: warning
                annotations:
                  summary: "Decode latency p90 elevated ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    Decode p90 above {t.decode_p90_warn}s suggests slow token throughput at current concurrency.

              - alert: VLLMHighInterTokenLatency
                expr: vllm:time_per_output_token_seconds:p90 > {t.inter_token_p90_warn}
                for: 10m
                labels:
                  severity: warning
                annotations:
                  summary: "Inter-token latency p90 high ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    Time per output token p90 exceeds {t.inter_token_p90_warn}s. Check GPU utilization, batching, or attention backend.

              - alert: VLLMHighKVCacheUsageWarning
                expr: vllm:kv_cache_usage_percent > {t.kv_warn}
                for: 5m
                labels:
                  severity: warning
                annotations:
                  summary: "KV cache usage > {t.kv_warn}% ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    KV cache sustained above {t.kv_warn}%. Expect elevated TTFT and possible stalls soon.

              - alert: VLLMHighKVCacheUsageCritical
                expr: vllm:kv_cache_usage_percent > {t.kv_crit}
                for: 1m
                labels:
                  severity: critical
                annotations:
                  summary: "KV cache nearly full ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    KV cache above {t.kv_crit}%. New requests may hang or fail. Shed load or lower max concurrency.

              - alert: VLLMStalledRequestQueue
                expr: |
                  (vllm:active_requests > 0)
                  and
                  (vllm:finished_requests_rate < {t.stalled_finished_rate})
                for: 5m
                labels:
                  severity: critical
                annotations:
                  summary: "Active requests but almost no completions ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    Active work with almost zero completions suggests deadlocks, cache starvation, or CUDA stalls.

              - alert: VLLMLowFinishedToActiveRatio
                expr: |
                  vllm:finished_to_active_ratio < {t.stalled_finished_rate}
                  and vllm:active_requests > {t.stalled_active_floor}
                for: 10m
                labels:
                  severity: warning
                annotations:
                  summary: "Queue draining slowly ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    Finished/active ratio is very low with many active requests.
                    Scheduler may be overloaded or batch policy misconfigured.

              - alert: VLLMCUDAGraphModeChanged
                expr: vllm:cuda_graph_mode{{mode!="FULL_AND_PIECEWISE"}} == 1
                for: 5m
                labels:
                  severity: info
                annotations:
                  summary: "CUDA graph mode changed ({{{{ $labels.model_name }}}} @ {{{{ $labels.instance }}}})"
                  description: |
                    Mode deviated from FULL_AND_PIECEWISE. Could indicate config drift or eager fallback.
        """
    ).strip()

    grafana_dashboard = {
        "title": "vLLM v1 – Inference Health (optimized)",
        "uid": "vllm-v1-health",
        "schemaVersion": 38,
        "version": 1,
        "timezone": "browser",
        "panels": [
            {
                "type": "row",
                "title": "Latency",
                "collapsed": False,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0},
                "panels": [],
            },
            {
                "id": 1,
                "type": "timeseries",
                "title": "TTFT p50 / p90 / p99",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "expr": "vllm:ttft_seconds:p50{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "p50 {{instance}} {{model_name}}",
                        "refId": "A",
                    },
                    {
                        "expr": "vllm:ttft_seconds:p90{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "p90 {{instance}} {{model_name}}",
                        "refId": "B",
                    },
                    {
                        "expr": "vllm:ttft_seconds:p99{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "p99 {{instance}} {{model_name}}",
                        "refId": "C",
                    },
                ],
                "gridPos": {"h": 7, "w": 12, "x": 0, "y": 1},
            },
            {
                "id": 2,
                "type": "timeseries",
                "title": "Prefill latency p50 / p90",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "expr": "vllm:prefill_seconds:p50{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "p50 {{instance}} {{model_name}}",
                        "refId": "A",
                    },
                    {
                        "expr": "vllm:prefill_seconds:p90{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "p90 {{instance}} {{model_name}}",
                        "refId": "B",
                    },
                ],
                "gridPos": {"h": 7, "w": 12, "x": 12, "y": 1},
            },
            {
                "id": 3,
                "type": "timeseries",
                "title": "Decode latency p50 / p90",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "expr": "vllm:decode_seconds:p50{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "p50 {{instance}} {{model_name}}",
                        "refId": "A",
                    },
                    {
                        "expr": "vllm:decode_seconds:p90{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "p90 {{instance}} {{model_name}}",
                        "refId": "B",
                    },
                ],
                "gridPos": {"h": 7, "w": 24, "x": 0, "y": 8},
            },
            {
                "type": "row",
                "title": "Throughput & Queue",
                "collapsed": False,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": 15},
                "panels": [],
            },
            {
                "id": 4,
                "type": "timeseries",
                "title": "Request rate (RPS)",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "expr": "vllm:req_rate:rps{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "rps {{instance}} {{model_name}}",
                        "refId": "A",
                    }
                ],
                "gridPos": {"h": 6, "w": 8, "x": 0, "y": 16},
            },
            {
                "id": 5,
                "type": "timeseries",
                "title": "Active / waiting / finished rate",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "expr": "vllm:active_requests{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "active {{instance}} {{model_name}}",
                        "refId": "A",
                    },
                    {
                        "expr": "vllm:waiting_requests{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "waiting {{instance}} {{model_name}}",
                        "refId": "B",
                    },
                    {
                        "expr": "vllm:finished_requests_rate{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "finished/s {{instance}} {{model_name}}",
                        "refId": "C",
                    },
                ],
                "gridPos": {"h": 6, "w": 8, "x": 8, "y": 16},
            },
            {
                "id": 6,
                "type": "timeseries",
                "title": "Finished / active ratio",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "expr": "vllm:finished_to_active_ratio{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "ratio {{instance}} {{model_name}}",
                        "refId": "A",
                    }
                ],
                "gridPos": {"h": 6, "w": 8, "x": 16, "y": 16},
            },
            {
                "type": "row",
                "title": "KV cache & CUDA graphs",
                "collapsed": False,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": 22},
                "panels": [],
            },
            {
                "id": 7,
                "type": "timeseries",
                "title": "KV cache usage (%)",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "expr": "vllm:kv_cache_usage_percent{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} {{model_name}}",
                        "refId": "A",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percent",
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "orange", "value": t.kv_warn},
                                {"color": "red", "value": t.kv_crit},
                            ],
                        },
                    },
                    "overrides": [],
                },
                "gridPos": {"h": 6, "w": 12, "x": 0, "y": 23},
            },
            {
                "id": 8,
                "type": "stat",
                "title": "Current CUDA graph mode",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "orientation": "horizontal",
                    "textMode": "value",
                },
                "targets": [
                    {
                        "expr": "max by (mode) (vllm:cuda_graph_mode{model_name=~\"$model\",instance=~\"$instance\"})",
                        "legendFormat": "{{mode}}",
                        "refId": "A",
                    }
                ],
                "gridPos": {"h": 4, "w": 12, "x": 12, "y": 23},
            },
            {
                "type": "row",
                "title": "Error hooks (fill if you export counters)",
                "collapsed": False,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": 29},
                "panels": [],
            },
            {
                "id": 9,
                "type": "timeseries",
                "title": "EngineCore errors (per min)",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "expr": f'sum by (instance, model_name, error_type) (rate({m.enginecore_errors_total}{{model_name=~"$model",instance=~"$instance"}}[1m]))',
                        "legendFormat": "{{error_type}} {{instance}} {{model_name}}",
                        "refId": "A",
                    }
                ],
                "gridPos": {"h": 6, "w": 12, "x": 0, "y": 30},
            },
            {
                "id": 10,
                "type": "timeseries",
                "title": "Scheduler errors (per min)",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "expr": f'sum by (instance, model_name, error_type) (rate({m.scheduler_errors_total}{{model_name=~"$model",instance=~"$instance"}}[1m]))',
                        "legendFormat": "{{error_type}} {{instance}} {{model_name}}",
                        "refId": "A",
                    }
                ],
                "gridPos": {"h": 6, "w": 12, "x": 12, "y": 30},
            },
            {
                "type": "row",
                "title": "Hotspots",
                "collapsed": False,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": 36},
                "panels": [],
            },
            {
                "id": 11,
                "type": "scatter",
                "title": "TTFT p90 vs KV cache usage",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "gridPos": {"h": 10, "w": 24, "x": 0, "y": 37},
                "targets": [
                    {
                        "refId": "A",
                        "expr": "vllm:ttft_seconds:p90{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "ttft_p90 {{instance}} {{model_name}}",
                        "instant": True,
                    },
                    {
                        "refId": "B",
                        "expr": "vllm:kv_cache_usage_percent{model_name=~\"$model\",instance=~\"$instance\"}",
                        "legendFormat": "kv_cache {{instance}} {{model_name}}",
                        "instant": True,
                    },
                ],
                "transformations": [
                    {
                        "id": "joinByField",
                        "options": {"byField": "instance", "mode": "inner"},
                    }
                ],
                "options": {
                    "xAxis": {"displayName": "KV cache usage (%)", "show": True},
                    "yAxis": {"displayName": "TTFT p90 (s)", "show": True},
                    "legend": {"showLegend": True, "displayMode": "list"},
                },
            },
        ],
        "templating": {
            "list": [
                {
                    "name": "instance",
                    "type": "query",
                    "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                    "query": f"label_values({m.gpu_cache_usage}, instance)",
                    "refresh": 1,
                    "multi": True,
                    "includeAll": True,
                },
                {
                    "name": "model",
                    "type": "query",
                    "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                    "query": f"label_values({m.gpu_cache_usage}, model_name)",
                    "refresh": 1,
                    "multi": True,
                    "includeAll": True,
                },
            ]
        },
    }

    return MonitoringBundle(
        name="vllm_monitoring",
        scrape_config=scrape_config,
        recording_rules=recording_rules,
        alerting_rules=alert_rules,
        grafana_dashboard=grafana_dashboard,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit vLLM monitoring bundle (full v1 surface).")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd() / "artifacts" / "vllm_monitoring",
        help="Directory to write Prometheus and Grafana assets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "vllm_monitoring.yaml",
        help="YAML with metric name and alert-threshold overrides.",
    )
    args = parser.parse_args()

    metrics, thresholds = load_monitoring_overrides(args.config)
    bundle = build_vllm_monitoring_bundle(metrics, thresholds)
    paths = write_bundle(bundle, args.outdir)
    print(f"Wrote monitoring bundle to {args.outdir}")
    print(f"Metric names: {asdict(metrics)}")
    print(f"Thresholds: {asdict(thresholds)}")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
