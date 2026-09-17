"""Admission counters need separate smoothing from noisy latency telemetry."""
import pytest

from labs.dynamic_router.router_policy import Router


def router(**kwargs):
    result = Router(**kwargs)
    for name in ('gpu0', 'gpu1'):
        result.register_gpu(name, is_prefill=True, is_decode=True)
    return result


def test_exact_queue_counter_preserves_latency_smoothing():
    value = router(queue_depth_alpha=1.0)
    value.update_metrics('gpu0', {'queue_depth': 0.0, 'ttft_ms': 0.0})
    value.update_metrics('gpu0', {'queue_depth': 4.0, 'ttft_ms': 100.0})
    snapshot = value._gpus['gpu0'].snapshot()
    assert snapshot['queue_depth'] == 4.0
    assert snapshot['ttft_ms'] == pytest.approx(30.0)
    value.update_metrics('gpu0', {'queue_depth': 0.0})
    assert value._gpus['gpu0'].snapshot()['queue_depth'] == 0.0


def test_default_sampled_telemetry_keeps_its_smoothing():
    value = router()
    value.update_metrics('gpu0', {'queue_depth': 0.0})
    value.update_metrics('gpu0', {'queue_depth': 4.0})
    assert value._gpus['gpu0'].snapshot()['queue_depth'] == pytest.approx(1.2)


def test_burst_admission_balances_actual_pending_counts():
    value = router(queue_depth_alpha=1.0)
    pending = {'gpu0': 3, 'gpu1': 0}
    for name, depth in pending.items():
        value.update_metrics(name, {'queue_depth': float(depth)})
    chosen = []
    for _ in range(8):
        name = value.choose_prefill_gpu()
        chosen.append(name)
        pending[name] += 1
        value.update_metrics(name, {'queue_depth': float(pending[name])})
    assert chosen[0] == 'gpu1'
    assert abs(pending['gpu0'] - pending['gpu1']) <= 1
    assert [chosen.count('gpu0'), chosen.count('gpu1')] == [3, 5]
