"""Actual CPU sizing-model and expert-math regressions; no GPU throughput claims."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from labs.moe_parallelism.plan import PlanEvaluator, SPEC_PRESETS, _bytes_to_ms
from labs.moe_parallelism.scenarios import get_scenario_pairs
from labs.moe_cuda.optimized_router import AdaptiveTopKMoE


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.mark.parametrize("name", ["memory_budget", "moe_grouping", "network_affinity", "parallelism_breakdown", "pipeline_schedule"])
def test_dgx_scenarios_use_matching_cluster_and_model(name):
    pair = get_scenario_pairs()[name]
    assert (pair.cluster, pair.model) == SPEC_PRESETS["dgx_a100_175b"]
    report = PlanEvaluator(pair.cluster, pair.model).analyze(pair.optimized)
    assert report.world_size_matches
    assert report.world_size == 128
    assert sum(report.stage_layers) == pair.model.layers
    assert report.tp_ep_product == report.gpus_per_stage
    if name == "parallelism_breakdown":
        baseline = PlanEvaluator(pair.cluster, pair.model).analyze(pair.baseline)
        assert not baseline.world_size_matches  # intentionally underspecified negative scenario
        assert baseline.world_size == 64


@pytest.mark.parametrize("name", ["gpt_gb200", "deepseek_gb200"])
def test_gb200_scenarios_keep_their_separate_576_gpu_scope(name):
    pair = get_scenario_pairs()[name]
    assert pair.cluster.gpus_total == 576
    assert all(PlanEvaluator(pair.cluster, pair.model).analyze(plan).world_size_matches for plan in (pair.baseline, pair.optimized))


def test_no_expert_collective_or_network_hotspot_for_ep_one():
    pair = get_scenario_pairs()["gpt_gb200"]
    plan = replace(pair.optimized, ep=1, cross_node_ep=True)
    report = PlanEvaluator(pair.cluster, pair.model).analyze(plan)
    assert report.ep_time_ms == 0
    assert not report.ep_cross_node
    assert not any("all-to-all" in msg for msg in report.hotspots)
    assert any("EP=1" in msg for msg in report.affinity)


def test_network_messages_use_configured_interconnect_and_rates():
    cluster, model = SPEC_PRESETS["gpt_oss_120b_gb200_ethernet"]
    plan = replace(get_scenario_pairs()["gpt_gb200"].optimized, cross_node_ep=True)
    report = PlanEvaluator(cluster, model).analyze(plan)
    text = "\n".join(report.hotspots + report.affinity)
    assert "Ethernet" in text and "400" in text
    assert "HDR100" not in text
    assert report.ep_time_ms > 0


def test_decimal_network_units_and_zero_payload_without_link():
    assert _bytes_to_ms(1e9, 100) == pytest.approx(10.0)
    assert _bytes_to_ms(0, 0) == 0


def test_no_collective_cost_on_single_rank_without_network_links():
    pair = get_scenario_pairs()["gpt_gb200"]
    cluster = replace(pair.cluster, nodes=1, gpus_per_node=1, nics_per_node=0, nvlink_bandwidth_tbps=0)
    plan = replace(pair.optimized, dp=1, pp=1, tp=1, ep=1, stage_layers=None)
    report = PlanEvaluator(cluster, pair.model).analyze(plan)
    assert report.world_size_matches
    assert report.dp_time_ms == report.pipeline_time_ms == report.ep_time_ms == 0
    assert report.estimated_step_ms == report.compute_ms


def reference(model, tokens):
    scores, ids = torch.topk(model.router(tokens) + model.gate_bias, model.top_k, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    all_outputs = []
    for expert in range(model.num_experts):
        first = model.expert_fc1.weight[expert * 2 * model.hidden_size:(expert + 1) * 2 * model.hidden_size]
        second = model.expert_fc2.weight[expert * model.hidden_size:(expert + 1) * model.hidden_size]
        all_outputs.append(F.linear(F.gelu(F.linear(tokens, first)), second))
    all_outputs = torch.stack(all_outputs, dim=1)
    selected = all_outputs.gather(1, ids.unsqueeze(-1).expand(-1, -1, model.hidden_size))
    return (selected * probabilities.unsqueeze(-1)).sum(1)


@pytest.mark.parametrize("batch,top_k,training", [(1, 1, False), (3, 2, False), (5, 4, False), (3, 2, True), (0, 2, False)])
def test_adaptive_router_uses_each_selected_experts_second_layer(batch, top_k, training):
    import copy

    torch.manual_seed(730)
    model = AdaptiveTopKMoE(hidden_size=3, num_experts=4, top_k=top_k).double()
    # Every expert has distinct nonzero second-layer weights. Force the top-1
    # case away from expert0 so this cannot pass by accidentally choosing it.
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.copy_(torch.tensor([-4.0, -1.0, 1.0, 2.0]))
        model.expert_fc2.weight.add_(torch.arange(4).repeat_interleave(3).unsqueeze(1))
    oracle = copy.deepcopy(model)
    backing = torch.randn(batch, 6, dtype=torch.float64)
    tokens = backing[:, ::2].requires_grad_(training)
    reference_tokens = tokens.detach().clone().requires_grad_(training)
    with torch.set_grad_enabled(training):
        actual = model(tokens)
        expected = reference(oracle, reference_tokens)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        if training:
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(tokens.grad, reference_tokens.grad, rtol=1e-10, atol=1e-10)
            for actual_parameter, reference_parameter in zip(model.parameters(), oracle.parameters()):
                torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, rtol=1e-10, atol=1e-10)
