"""Focused CPU tests for the read-only data-loading analysis provider."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest import mock

from core.diagnostics import data_loading
from core.perf_core_base import PerformanceCoreBase


def _cpu_observation(label: str = "fixture") -> data_loading.CpuObservation:
    return data_loading.CpuObservation(
        architecture="fixture-arch",
        logical_cpu_count=16,
        model=label,
        is_grace=None,
        grace_status="not_observed",
        provenance=f"cpu:{label}",
    )


def _gpu_observation(
    *,
    status: data_loading.GpuStatus = "available",
    gpu_id: int | None = 2,
    pci_bus_id: str | None = "0000:17:00.0",
) -> data_loading.GpuObservation:
    return data_loading.GpuObservation(
        status=status,
        gpu_id=gpu_id,
        device_count=4 if status == "available" else 0,
        pci_bus_id=pci_bus_id,
        provenance="gpu:fixture",
    )


class DataLoadingAnalysisProviderTests(unittest.TestCase):
    def test_recommendation_config_rejects_non_contract_types(self) -> None:
        cases = (
            ({"batch_size": math.nan}, TypeError),
            ({"batch_size": True}, TypeError),
            ({"num_workers": 1.5}, TypeError),
            ({"num_workers": False}, TypeError),
            ({"prefetch_factor": True}, TypeError),
            ({"prefetch_factor": 1.5}, TypeError),
            ({"pin_memory": "yes"}, TypeError),
            ({"persistent_workers": 1}, TypeError),
        )
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(expected):
                data_loading.DataLoaderRecommendationConfig(**kwargs)

    def test_cpu_model_prefers_model_name_over_numeric_processor_index(self) -> None:
        cpuinfo_path = mock.Mock()
        cpuinfo_path.read_text.return_value = (
            "processor : 0\n"
            "model name : NVIDIA Grace CPU Superchip\n"
            "hardware : fallback platform\n"
        )

        with mock.patch.object(data_loading, "Path", return_value=cpuinfo_path):
            observation = data_loading.observe_cpu()

        self.assertEqual(observation.model, "NVIDIA Grace CPU Superchip")
        self.assertTrue(observation.is_grace)
        self.assertEqual(observation.grace_status, "observed")

    def test_generic_arm_or_nvidia_text_does_not_invent_grace(self) -> None:
        cpuinfo_path = mock.Mock()
        cpuinfo_path.read_text.return_value = (
            "processor : ARM Neoverse V2\n" "Hardware : NVIDIA reference platform\n"
        )

        with (
            mock.patch.object(data_loading, "Path", return_value=cpuinfo_path),
            mock.patch.object(data_loading.platform, "machine", return_value="aarch64"),
            mock.patch.object(data_loading.os, "cpu_count", return_value=144),
        ):
            observation = data_loading.observe_cpu()

        self.assertIsNone(observation.is_grace)
        self.assertEqual(observation.grace_status, "not_observed")
        self.assertEqual(observation.architecture, "aarch64")

    def test_endpoint_reports_observed_affinity_without_mutating_it(self) -> None:
        provider = data_loading.ReadOnlyDataLoadingAnalysisProvider(
            cpu_observer=lambda: _cpu_observation(),
            gpu_observer=lambda: _gpu_observation(),
            affinity_observer=lambda: data_loading.AffinityObservation(
                cpu_ids=(1, 3, 5),
                status="observed",
                provenance="affinity:fixture",
            ),
            numa_observer=lambda _gpu: data_loading.NumaMappingObservation(
                status="known",
                numa_node=6,
                provenance="numa:fixture",
            ),
        )
        core = PerformanceCoreBase.__new__(PerformanceCoreBase)
        core._data_loading_provider = provider

        with mock.patch.object(
            data_loading.os,
            "sched_setaffinity",
            create=True,
        ) as affinity_setter:
            result = core.get_data_loading_analysis()

        affinity_setter.assert_not_called()
        self.assertTrue(result["success"])
        self.assertFalse(result["affinity_applied"])
        self.assertEqual(result["cpu_affinity"], [1, 3, 5])
        self.assertEqual(result["current_cpu_affinity"], [1, 3, 5])
        self.assertEqual(result["current_affinity_status"], "observed")
        self.assertEqual(result["numa_node"], 6)
        self.assertEqual(result["gpu_numa_mapping"]["status"], "known")

    def test_available_gpu_with_no_observed_mapping_stays_unknown(self) -> None:
        provider = data_loading.ReadOnlyDataLoadingAnalysisProvider(
            cpu_observer=lambda: _cpu_observation(),
            gpu_observer=lambda: _gpu_observation(gpu_id=7, pci_bus_id=None),
            affinity_observer=lambda: data_loading.AffinityObservation(
                cpu_ids=None,
                status="unknown",
                provenance="affinity unavailable",
            ),
            numa_observer=data_loading.observe_gpu_numa_mapping,
        )

        result = provider.analyze().to_dict()

        self.assertEqual(result["gpu_id"], 7)
        self.assertIsNone(result["numa_node"])
        self.assertEqual(result["gpu_numa_mapping"]["status"], "unknown")
        self.assertIsNone(result["gpu_numa_mapping"]["numa_node"])
        self.assertIn("PCI bus ID unavailable", result["gpu_numa_mapping"]["provenance"])
        self.assertIsNone(result["current_cpu_affinity"])
        self.assertEqual(result["current_affinity_status"], "unknown")

    def test_configured_recommendations_are_deterministic(self) -> None:
        recommendations = data_loading.DataLoaderRecommendationConfig(
            batch_size=64,
            num_workers=3,
            pin_memory=False,
            prefetch_factor=5,
            persistent_workers=True,
        )
        affinity = data_loading.AffinityObservation(
            cpu_ids=(0,),
            status="observed",
            provenance="affinity:fixture",
        )
        known_provider = data_loading.ReadOnlyDataLoadingAnalysisProvider(
            recommendations,
            cpu_observer=lambda: _cpu_observation("known"),
            gpu_observer=lambda: _gpu_observation(),
            affinity_observer=lambda: affinity,
            numa_observer=lambda _gpu: data_loading.NumaMappingObservation(
                status="known",
                numa_node=4,
                provenance="numa:known",
            ),
        )
        unknown_provider = data_loading.ReadOnlyDataLoadingAnalysisProvider(
            recommendations,
            cpu_observer=lambda: _cpu_observation("unknown"),
            gpu_observer=lambda: _gpu_observation(
                status="unavailable",
                gpu_id=None,
                pci_bus_id=None,
            ),
            affinity_observer=lambda: affinity,
            numa_observer=data_loading.observe_gpu_numa_mapping,
        )

        known = known_provider.analyze().to_dict()
        unknown = unknown_provider.analyze().to_dict()
        expected = {
            "batch_size": 64,
            "num_workers": 3,
            "pin_memory": False,
            "persistent_workers": True,
            "prefetch_factor": 5,
        }

        self.assertEqual(known["dataloader_kwargs"], expected)
        self.assertEqual(unknown["dataloader_kwargs"], expected)
        self.assertEqual(known["recommendation_provenance"], unknown["recommendation_provenance"])
        self.assertEqual(
            known["recommendation_provenance"]["source"],
            "explicit_provider_config",
        )
        self.assertFalse(known["recommendation_provenance"]["hardware_adaptive"])

    def test_default_recommendations_are_labeled_unmeasured_static_policy(self) -> None:
        result = data_loading.ReadOnlyDataLoadingAnalysisProvider(
            cpu_observer=lambda: _cpu_observation(),
            gpu_observer=lambda: _gpu_observation(
                status="unavailable", gpu_id=None, pci_bus_id=None
            ),
            affinity_observer=lambda: data_loading.AffinityObservation(
                cpu_ids=None,
                status="unknown",
                provenance="affinity unavailable",
            ),
            numa_observer=data_loading.observe_gpu_numa_mapping,
        ).analyze().to_dict()

        self.assertEqual(
            result["recommendation_provenance"]["source"],
            "default_static_policy",
        )
        self.assertIn("unmeasured static defaults", result["notes"][0])

    def test_provider_failure_preserves_versioned_response_envelope(self) -> None:
        class FailingProvider:
            def analyze(self):
                raise RuntimeError("fixture provider failure")

        core = PerformanceCoreBase.__new__(PerformanceCoreBase)
        core._data_loading_provider = FailingProvider()

        result = core.get_data_loading_analysis()

        self.assertFalse(result["success"])
        self.assertEqual(
            result["schema_version"],
            data_loading.DATA_LOADING_ANALYSIS_SCHEMA_VERSION,
        )
        self.assertEqual(result["analysis_mode"], "read_only")
        self.assertIsNone(result["dataloader_kwargs"])
        self.assertEqual(result["cpu"]["grace_status"], "unknown")
        self.assertEqual(result["current_affinity_status"], "unknown")
        self.assertEqual(
            result["recommendation_provenance"]["source"],
            "unavailable_provider_error",
        )
        self.assertFalse(result["affinity_applied"])

    def test_schema_version_and_read_only_mode_are_explicit(self) -> None:
        provider = data_loading.ReadOnlyDataLoadingAnalysisProvider(
            cpu_observer=lambda: _cpu_observation(),
            gpu_observer=lambda: _gpu_observation(
                status="unavailable",
                gpu_id=None,
                pci_bus_id=None,
            ),
            affinity_observer=lambda: data_loading.AffinityObservation(
                cpu_ids=None,
                status="unknown",
                provenance="affinity unavailable",
            ),
            numa_observer=data_loading.observe_gpu_numa_mapping,
        )

        result = provider.analyze().to_dict()

        self.assertEqual(
            result["schema_version"],
            data_loading.DATA_LOADING_ANALYSIS_SCHEMA_VERSION,
        )
        self.assertEqual(result["analysis_mode"], "read_only")
        self.assertFalse(result["affinity_applied"])
        self.assertNotIn("throughput", result)
        self.assertNotIn("worker_efficiency", result)

    def test_falsy_injected_provider_is_not_replaced(self) -> None:
        class FalsyProvider:
            def __bool__(self) -> bool:
                return False

            def analyze(self) -> data_loading.DataLoadingAnalysis:
                raise AssertionError("not called by this constructor test")

        provider = FalsyProvider()
        with (
            mock.patch(
                "core.perf_core_base.get_bench_roots",
                return_value=[Path("fixture-bench-root")],
            ),
            mock.patch.object(PerformanceCoreBase, "_make_analyzer"),
            mock.patch(
                "core.perf_core_base.ReadOnlyDataLoadingAnalysisProvider"
            ) as default_provider,
        ):
            core = PerformanceCoreBase(data_loading_provider=provider)

        self.assertIs(core._data_loading_provider, provider)
        default_provider.assert_not_called()

    def test_actual_cli_renders_read_only_observation_contract(self) -> None:
        from typer.testing import CliRunner

        from cli.aisp import app

        payload = (
            data_loading.ReadOnlyDataLoadingAnalysisProvider(
                cpu_observer=lambda: _cpu_observation("CLI fixture"),
                gpu_observer=lambda: _gpu_observation(gpu_id=3, pci_bus_id=None),
                affinity_observer=lambda: data_loading.AffinityObservation(
                    cpu_ids=(2, 4),
                    status="observed",
                    provenance="affinity:cli",
                ),
                numa_observer=data_loading.observe_gpu_numa_mapping,
            )
            .analyze()
            .to_dict()
        )

        with mock.patch.object(
            PerformanceCoreBase,
            "get_data_loading_analysis",
            return_value=payload,
        ):
            result = CliRunner().invoke(app, ["profile", "data-loading"])

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("Data Loading Analysis (read-only)", result.stdout)
        self.assertIn("Schema: data_loading_analysis.v1", result.stdout)
        self.assertIn("Recommended DataLoader Settings", result.stdout)
        self.assertIn("Observed CPU", result.stdout)
        self.assertIn("Observed GPU/NUMA Locality", result.stdout)
        self.assertIn("mapping status: unknown", result.stdout)
        self.assertIn("Observed Process Affinity", result.stdout)
        self.assertIn("affinity applied: False", result.stdout)
        self.assertNotIn("Bottlenecks Detected", result.stdout)
        self.assertNotIn("Worker Efficiency", result.stdout)

    def test_actual_cli_returns_nonzero_when_provider_fails(self) -> None:
        from typer.testing import CliRunner

        from cli.aisp import app

        with mock.patch.object(
            PerformanceCoreBase,
            "get_data_loading_analysis",
            return_value={"success": False, "error": "fixture provider failure"},
        ):
            result = CliRunner().invoke(app, ["profile", "data-loading"])

        self.assertEqual(result.exit_code, 1, result.stdout)
        self.assertIn("fixture provider failure", result.stdout)

    def test_actual_mcp_tool_and_description_match_read_only_contract(self) -> None:
        from mcp import mcp_server

        payload = (
            data_loading.ReadOnlyDataLoadingAnalysisProvider(
                cpu_observer=lambda: _cpu_observation("MCP fixture"),
                gpu_observer=lambda: _gpu_observation(
                    status="unavailable",
                    gpu_id=None,
                    pci_bus_id=None,
                ),
                affinity_observer=lambda: data_loading.AffinityObservation(
                    cpu_ids=None,
                    status="unknown",
                    provenance="affinity:mcp",
                ),
                numa_observer=data_loading.observe_gpu_numa_mapping,
            )
            .analyze()
            .to_dict()
        )

        with mock.patch.object(
            PerformanceCoreBase,
            "get_data_loading_analysis",
            return_value=payload,
        ):
            result = mcp_server.tool_analyze_dataloader({})

        self.assertEqual(result["schema_version"], "data_loading_analysis.v1")
        self.assertEqual(result["analysis_mode"], "read_only")
        self.assertEqual(result["gpu_numa_mapping"]["status"], "not_applicable")
        self.assertFalse(result["affinity_applied"])
        self.assertNotIn("throughput", result)
        self.assertNotIn("worker_efficiency", result)

        description = mcp_server.TOOLS["analyze_dataloader"].description
        self.assertIn("data_loading_analysis.v1", description)
        self.assertIn("Read-only DataLoader", description)
        self.assertNotIn("Returns: {throughput", description)
        self.assertNotIn("worker_efficiency", description)


if __name__ == "__main__":
    unittest.main()
