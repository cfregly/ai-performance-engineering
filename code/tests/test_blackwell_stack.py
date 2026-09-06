#!/usr/bin/env python3
"""GPU stack regression checks for the pinned PyTorch/CUDA/Triton environment.

Missing hardware is an explicit pytest skip. Once a prerequisite is present,
compile, launch, profiler and numerical errors propagate as test failures.
These smoke measurements do not qualify hardware peaks or TMA instructions.
"""

import os
from datetime import timedelta
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

from core.harness.arch_config import ArchitectureConfig
from core.utils.compile_utils import configure_tf32, restore_tf32


def ensure_cuda(feature: str) -> bool:
    """Require actual CUDA; returning from a test must never impersonate a skip."""
    if not torch.cuda.is_available():
        pytest.skip(f"{feature} requires a real CUDA device")
    return True


@pytest.fixture(autouse=True)
def ieee_float32_reference_policy():
    """Keep these FP32 reference comparisons independent of prior TF32 tests."""
    previous = configure_tf32(enable_matmul=False, enable_cudnn=False)
    try:
        yield
    finally:
        restore_tf32(previous)


def test_architecture_detection():
    ensure_cuda("Blackwell architecture detection")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) not in {(10, 0), (10, 3)}:
        pytest.skip(f"Datacenter Blackwell check requires CC 10.0/10.3, found {props.major}.{props.minor}")
    config = ArchitectureConfig()
    expected = "blackwell_ultra" if props.minor == 3 else "blackwell"
    assert config.arch == expected
    assert config.config["compute_capability"] == f"{props.major}.{props.minor}"
    assert config.config["sm_version"] == f"sm_{props.major}{props.minor}"


def test_pytorch_29_features():
    """Compile and compare multiple batch shapes; configuration alone is not proof."""
    ensure_cuda("torch.compile")
    model = torch.nn.Linear(128, 96).cuda().eval()
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    with torch.no_grad():
        for batch in (3, 11, 5):
            inputs = torch.randn(batch, 128, device="cuda")
            expected = model(inputs)
            actual = compiled(inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    torch.cuda.synchronize()


def test_cuda_130_features():
    """Exercise stream dependencies with a CUDA-13 build, without claiming TMA."""
    ensure_cuda("CUDA 13 stream integration")
    version = torch.version.cuda
    assert version is not None, "CUDA available but PyTorch build has no CUDA version"
    if tuple(int(v) for v in version.split(".")[:2]) < (13, 0):
        pytest.skip(f"CUDA 13 stack check requires a CUDA >=13 build, found {version}")
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        inputs = torch.arange(8193, device="cuda", dtype=torch.float32)
        ready = producer.record_event()
    consumer.wait_event(ready)
    with torch.cuda.stream(consumer):
        actual = inputs * 3 + 1
        inputs.record_stream(consumer)
        finished = consumer.record_event()
    torch.cuda.current_stream().wait_event(finished)
    actual.record_stream(torch.cuda.current_stream())
    torch.testing.assert_close(actual.cpu(), torch.arange(8193, dtype=torch.float32) * 3 + 1)


def test_profiling_tools():
    """Require recorded CPU and CUDA work; an unopened active window is insufficient."""
    ensure_cuda("CUDA profiler and NVTX")
    from core.profiling.nvtx_helper import nvtx_range

    x = torch.randn(128, 128, device="cuda")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as trace:
        with record_function("audit_stack_mm"), nvtx_range("audit_stack_mm", enable=True):
            output = torch.mm(x, x)
        torch.cuda.synchronize()
    torch.testing.assert_close(output, x.double().mm(x.double()).float(), rtol=1e-4, atol=1e-4)
    events = trace.events()
    assert any(event.name == "audit_stack_mm" for event in events), "CPU range was not recorded"
    assert any(event.device_type == torch.autograd.DeviceType.CUDA for event in events), "No CUDA work recorded"


def test_triton_35():
    """Compile a masked, ragged add and compare every output, including the tail."""
    ensure_cuda("Triton kernel")
    # A supported CUDA environment must install its declared Triton dependency.
    # Missing or broken imports there are failures, not success-shaped fallbacks.
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0)
        tl.store(output_ptr + offsets, x + y, mask=mask)

    n, block = 1027, 128
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = torch.linspace(-3, 7, n, dtype=torch.float32, device="cuda")
    output = torch.full_like(x, float("nan"))
    add_kernel[(triton.cdiv(n, block),)](x, y, output, n, BLOCK_SIZE=block)
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)


def test_performance():
    """Record actual work and correct traffic/FLOPs; do not impose a speed threshold."""
    ensure_cuda("CUDA timing smoke check")
    size = 16 * 1024 * 1024
    x = torch.randn(size, dtype=torch.float32, device="cuda")
    y = torch.randn_like(x)
    for _ in range(2):
        output = x + y
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    output = x + y
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    assert elapsed_ms > 0
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)
    bytes_moved = (x.numel() + y.numel() + output.numel()) * x.element_size()
    print(f"Add traffic (two reads + one write): {bytes_moved / elapsed_ms / 1e6:.2f} GB/s")

    a = torch.randn(512, 512, device="cuda")
    b = torch.randn_like(a)
    for _ in range(2):
        output = a @ b
    start.record()
    output = a @ b
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    assert elapsed_ms > 0
    torch.testing.assert_close(output, (a.double() @ b.double()).float(), rtol=1e-4, atol=1e-4)
    print(f"FP32 matmul smoke: {2 * 512**3 / elapsed_ms / 1e9:.2f} TFLOP/s")


@pytest.fixture(scope="module")
def nccl_process_group():
    """Keep one NCCL group for the module; in-process reinitialization is unsupported."""
    ensure_cuda("distributed inference")
    if not dist.is_available() or not dist.is_nccl_available():
        pytest.skip("NCCL is unavailable")
    if not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Run the distributed cases with torchrun --nproc-per-node=2 -m pytest")
    owns_group = not dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert local_rank < torch.cuda.device_count(), "LOCAL_RANK exceeds actual CUDA device count"
    torch.cuda.set_device(local_rank)
    if owns_group:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=60),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    try:
        assert dist.get_world_size() == 2, "These cases require exactly two ranks"
        assert dist.get_backend() == "nccl", "Distributed GPU validation requires NCCL"
        yield dist.get_rank(), local_rank
    finally:
        if owns_group:
            dist.destroy_process_group()


@pytest.fixture
def nccl_world(nccl_process_group):
    """Isolate compiler state while reusing the module's live NCCL group."""
    # Compiled FlexAttention specializations are process-global.  Isolate each
    # distributed test so shape coverage in one test cannot exhaust Dynamo's
    # recompile limit in the next one.
    torch.compiler.reset()
    torch.manual_seed(20260830)
    return nccl_process_group


def test_kv_cache_batched_attention_2gpu(nccl_world):
    """
    Test batched attention with heterogeneous cache lengths on 2 GPUs.
    Validates padding/masking logic and head sharding.
    """
    print("\n=== KV Cache Batched Attention 2-GPU Test ===")
    
    from ch16.inference_serving_multigpu import DemoCausalLM, ShardedKVCacheManager

    rank, local_rank = nccl_world
    device = torch.device(f"cuda:{local_rank}")

    num_layers = 2
    num_heads = 8  # 4 heads per GPU
    d_model = 128
    head_dim = d_model // num_heads
    vocab_size = 128
    batch_size = 4
    
    # Create model and cache with 2 GPUs
    model = DemoCausalLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_gpus=2,
        max_batch_size=batch_size,
        max_seq_len=64,
    ).to(device)
    model.eval()
    
    kv_cache = ShardedKVCacheManager(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=batch_size,
        max_seq_len=64,
        num_gpus=2,
        dtype=torch.float32,
        page_size=4,
    )
    
    # Allocate slots for batch
    slots = [kv_cache.allocate_slot() for _ in range(batch_size)]
    assert all(s is not None for s in slots), "Failed to allocate slots"
    
    # Create heterogeneous initial sequences: [3, 5, 7, 9] tokens
    initial_lengths = [3, 5, 7, 9]
    initial_prompts = []
    for length in initial_lengths:
        prompt = torch.randint(1, vocab_size, (1, length), device=device, dtype=torch.long)
        initial_prompts.append(prompt)
    
    # Process initial prompts and cache KV
    for idx, prompt in enumerate(initial_prompts):
        logits, keys, values = model(prompt)
        
        # Store in cache
        key_stack = torch.stack(
            [keys[layer_idx, 0] for layer_idx in range(num_layers)], dim=0
        )
        value_stack = torch.stack(
            [values[layer_idx, 0] for layer_idx in range(num_layers)], dim=0
        )
        kv_cache.append_tokens(
            slot=slots[idx],
            key=key_stack,
            value=value_stack,
            num_tokens=prompt.shape[1],
        )
    
    if rank == 0:
        print(f" Initial cache lengths: {initial_lengths}")
        print(f" Heads per GPU: {kv_cache.heads_per_gpu}")
    
    # Generate next token using cached KV with batched attention
    new_tokens = torch.randint(1, vocab_size, (batch_size, 1), device=device, dtype=torch.long)
    
    # Build past_kv for batched forward
    past_kv = []
    for layer_idx in range(num_layers):
        layer_cache = []
        for slot in slots:
            cache_k, cache_v = kv_cache.get_cache(slot, layer_idx)
            layer_cache.append((cache_k, cache_v))
        past_kv.append(layer_cache)
    
    # Forward with cached KV (tests batched attention with padding/masking)
    logits_batched, keys_batched, values_batched = model(new_tokens, past_kv=past_kv)
    
    if rank == 0:
        print(f" Batched logits shape: {logits_batched.shape}")
        assert logits_batched.shape == (batch_size, vocab_size), "Incorrect batched logits shape"
    
    # Validate against full-context forward for each request
    for idx in range(batch_size):
        full_sequence = torch.cat([initial_prompts[idx], new_tokens[idx:idx+1]], dim=1)
        logits_full, _, _ = model(full_sequence)
        
        # Compare logits
        torch.testing.assert_close(
            logits_batched[idx:idx+1],
            logits_full,
            rtol=1e-2,
            atol=5e-2,
            msg=f"Logits mismatch for request {idx} (rank {rank})",
        )
    
    if rank == 0:
        print(" Batched attention matches full-context forward")
        print(" Padding/masking logic validated")
        print(" Head sharding across GPUs validated")
    
    # Cleanup
    for slot in slots:
        kv_cache.free_slot(slot)
    
    if dist.is_initialized():
        dist.barrier()


def test_inference_server_multigpu_distributed(nccl_world):
    """
    Full end-to-end test of InferenceServerMultiGPU with distributed execution.
    Tests continuous batching, cache management, and throughput.
    """
    print("\n=== Inference Server Multi-GPU Distributed Test ===")
    
    from ch16.inference_serving_multigpu import (
        DemoCausalLM, InferenceServerMultiGPU, InferenceRequest,
    )

    rank, _local_rank = nccl_world
    world_size = dist.get_world_size()

    # Create demo model (keep head sharding divisible by world_size)
    num_heads = 4 * world_size
    d_model = 64 * world_size
    model = DemoCausalLM(
        vocab_size=256,
        d_model=d_model,
        num_layers=4,
        num_heads=num_heads,
        num_gpus=world_size,
        max_batch_size=32,
        max_seq_len=256,
    )
    
    # Create server
    server = InferenceServerMultiGPU(
        model=model,
        num_layers=4,
        d_model=d_model,
        num_heads=num_heads,
        max_batch_size=32,
        max_seq_len=256,
    )
    assert (
        server._prefill_graph_available
    ), "Distributed prefill CUDA graph was not captured"

    # Verify every graph output against the eager path before timed serving.
    graph_input = torch.randint(
        0,
        model.vocab_size,
        (server.max_batch_size, server._prefill_graph_seq_len),
        device=server.device,
    )
    with torch.inference_mode():
        graph_outputs = server._run_prefill_graph(graph_input)
        eager_outputs = server.model(graph_input, past_kv=None)
    output_names = ("logits", "keys", "values")
    assert len(graph_outputs) == len(eager_outputs) == len(output_names)
    for output_name, graph_output, eager_output in zip(
        output_names,
        graph_outputs,
        eager_outputs,
    ):
        torch.testing.assert_close(
            graph_output,
            eager_output,
            rtol=1e-4,
            atol=1e-5,
            msg=f"CUDA graph {output_name} output differs from eager inference",
        )
    
    # Submit test requests (all ranks enqueue identical work to keep the scheduler in sync)
    if rank == 0:
        print(" Submitting 32 test requests...")
    
    for i in range(32):
        # Varying prompt lengths: 10-50 tokens
        prompt_length = 10 + (i * 2)
        request = InferenceRequest(
            request_id=f"test_req_{i}",
            prompt_tokens=list(range(1, prompt_length + 1)),
            max_new_tokens=10,
            temperature=1.0,
            priority=0,
        )
        # All ranks add the same requests to avoid deadlock in all_reduce
        server.scheduler.add_request(request)
    
    # Synchronize all ranks before starting serve loop
    dist.barrier()
    
    # Run serving loop for 2 seconds (all ranks execute together)
    server.serve_loop(duration_seconds=2.0)
    
    # Check statistics
    if rank == 0:
        stats = server.scheduler.get_stats()
        cache_stats = server.kv_cache.stats()
        
        print(f" Total requests: {stats['total_requests']}")
        print(f" Completed: {stats['completed_requests']}")
        print(f" Tokens generated: {stats['total_tokens_generated']}")
        print(f" Active cache slots: {cache_stats['active_slots']}")
        print(f" Resident pages: {cache_stats['resident_pages']}")
        
        assert stats['completed_requests'] > 0, "No requests completed"
        assert stats['total_tokens_generated'] > 0, "No tokens generated"
        print(" Distributed inference server validated")
    
    dist.barrier()


def main() -> int:
    """Run the actual test runner; do not print success when CUDA is missing."""
    if not torch.cuda.is_available():
        print("UNAVAILABLE: Blackwell validation requires a real CUDA device", file=sys.stderr)
        return 3
    return int(pytest.main([str(Path(__file__).resolve()), "-q", "-ra", *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
