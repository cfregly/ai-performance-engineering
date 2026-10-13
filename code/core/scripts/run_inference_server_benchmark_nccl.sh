#!/bin/bash
# Run inference server benchmark with correct NCCL settings
# Usage: ./core/scripts/run_inference_server_benchmark_nccl.sh [duration] [target_qps] [output_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DURATION="${1:-300}"  # Default 5 minutes
TARGET_QPS="${2:-100}"
OUTPUT_DIR="${3:-inference_server_nccl_results_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

echo "================================================================================"
echo "Inference Server Benchmark with Correct NCCL Settings"
echo "================================================================================"
echo ""
echo "Duration: ${DURATION}s"
echo "Target QPS: ${TARGET_QPS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Step 1: Configure NCCL for NVLink mesh
echo "Step 1: Configuring NCCL for NVLink mesh..."
echo ""

# Critical: Enable NVLink (was incorrectly disabled before)
# NCCL finds NVLink through topology discovery. It has no environment switch
# that turns NVLink or NVLink-C2C on.
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_NVLS_ENABLE=1
export NCCL_PROTO=Simple
export NCCL_ALGO=Tree,Ring,NVLS
export NCCL_NCHANNELS_PER_NET_PEER=8

echo "NCCL Configuration:"
echo "  NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL}"
echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "  NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE}"
echo "  NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE}"
echo "  NCCL_PROTO=${NCCL_PROTO}"
echo "  NCCL_ALGO=${NCCL_ALGO}"
echo ""

# Step 2: Verify NVLink configuration
echo "Step 2: Verifying NVLink configuration..."
if command -v python &> /dev/null; then
    python core/verification/verify_nvlink.py > "${OUTPUT_DIR}/nvlink_verification.txt" 2>&1 || true
    echo "NVLink verification saved to: ${OUTPUT_DIR}/nvlink_verification.txt"
fi
echo ""

# Step 3: Capture baseline system info
echo "Step 3: Capturing system information..."
{
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo ""
    echo "=== GPU Information ==="
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
    echo ""
    echo "=== NVLink Topology ==="
    nvidia-smi topo -m
    echo ""
    echo "=== NCCL Environment ==="
    env | grep NCCL | sort
} > "${OUTPUT_DIR}/system_info.txt"
echo "System info saved to: ${OUTPUT_DIR}/system_info.txt"
echo ""

# Step 4: Run inference server load test
echo "Step 4: Running inference server load test..."
echo ""

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RESULTS_JSON="${OUTPUT_DIR}/inference_results.json"
LOG_FILE="${OUTPUT_DIR}/inference_server.log"

echo "Starting load test..."
echo "  Duration: ${DURATION}s"
echo "  Target QPS: ${TARGET_QPS}"
echo "  GPUs: 8"
echo ""

if torchrun --nproc_per_node=8 ch16/inference_server_load_test.py \
    --duration ${DURATION} \
    --target-qps ${TARGET_QPS} \
    --output-json "${RESULTS_JSON}" \
    > "${LOG_FILE}" 2>&1; then
    echo "[OK] Load test completed successfully"
    SUCCESS=1
else
    echo "WARNING: Load test completed with errors (check log)"
    SUCCESS=0
fi

echo ""

# Step 5: Analyze results
if [ -f "${RESULTS_JSON}" ]; then
    echo "Step 5: Analyzing results..."
    echo ""
    
    # Extract key metrics using Python
    python3 << EOF > "${OUTPUT_DIR}/metrics_summary.txt"
import json
import sys

try:
    with open("${RESULTS_JSON}", "r") as f:
        data = json.load(f)
    
    print("=== Inference Server Metrics ===")
    print(f"Duration: {data.get('elapsed', 0):.1f}s")
    print(f"Throughput: {data.get('throughput_tok_per_s', 0):,.0f} tokens/s")
    print(f"Total Requests: {data.get('total_requests', 0)}")
    print(f"Completed: {data.get('completed_requests', 0)}")
    print(f"Rejected: {data.get('rejected_requests', 0)}")
    print("")
    print("Latency (ms):")
    print(f"  P50: {data.get('latency_p50_ms', 0):.2f}")
    print(f"  P90: {data.get('latency_p90_ms', 0):.2f}")
    print(f"  P99: {data.get('latency_p99_ms', 0):.2f}")
except Exception as e:
    print(f"Error parsing results: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    cat "${OUTPUT_DIR}/metrics_summary.txt"
    echo ""
fi

# Step 6: Create summary document
echo "Step 6: Creating summary document..."
cat > "${OUTPUT_DIR}/BENCHMARK_SUMMARY.md" << EOF
# Inference Server Benchmark with Correct NCCL Settings

**Generated:** $(date)
**Duration:** ${DURATION}s
**Target QPS:** ${TARGET_QPS}
**Status:** $(if [ ${SUCCESS} -eq 1 ]; then echo "[OK] Success"; else echo "WARNING: Completed with errors"; fi)

## Configuration

### NCCL Settings (Corrected)

\`\`\`bash
export NCCL_P2P_LEVEL=NVL         # Force NVLink level
export NCCL_P2P_DISABLE=0         # ENABLE P2P (was incorrectly 1!)
export NCCL_IB_DISABLE=1          # Disable InfiniBand
export NCCL_SHM_DISABLE=0         # Enable shared memory
export NCCL_NVLS_ENABLE=1         # Enable NVLS multicast
export NCCL_PROTO=Simple          # Simple protocol
export NCCL_ALGO=Tree,Ring,NVLS   # Use Tree, Ring, NVLS
\`\`\`

### Hardware

- Multi-GPU NVIDIA GPUs
- Full NVLink mesh (18 links @ 50 GB/s = 900 GB/s per GPU)
- NVLS multicast support (24 channels)

## Results

$(if [ -f "${OUTPUT_DIR}/metrics_summary.txt" ]; then
    echo "\`\`\`"
    cat "${OUTPUT_DIR}/metrics_summary.txt"
    echo "\`\`\`"
else
    echo "Results not available - check logs"
fi)

## Files

- \`inference_results.json\`: Detailed benchmark results
- \`inference_server.log\`: Full log output
- \`system_info.txt\`: System and GPU information
- \`nvlink_verification.txt\`: NVLink configuration verification
- \`metrics_summary.txt\`: Key metrics summary
- \`BENCHMARK_SUMMARY.md\`: This file

## Comparison with Previous Results

**Previous (Incorrect NCCL):**
- NCCL_P2P_DISABLE=1 (NVLink disabled)
- Fallback to PCIe (~64 GB/s)
- Lower throughput expected

**Current (Correct NCCL):**
- NCCL_P2P_DISABLE=0 (NVLink enabled)
- Full NVLink bandwidth (900 GB/s per GPU)
- Higher throughput expected

## Next Steps

1. Compare results with previous benchmarks
2. Update performance expectations in documentation
3. Document correct NCCL configuration
4. Run additional benchmarks at different QPS levels

## Related Documentation

- \`docs/playbooks/nvlink_pcie_playbook.md\` - NCCL configuration guide
- \`core/scripts/orchestrate_multigpu_load_test.sh\` - Load test orchestration
- \`ch16/inference_server_load_test.py\` - Inference server implementation
EOF

echo "Summary written to: ${OUTPUT_DIR}/BENCHMARK_SUMMARY.md"
echo ""

# Step 7: Final summary
echo "================================================================================"
echo "Benchmark Complete"
echo "================================================================================"
echo ""
echo "Results directory: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  - ${RESULTS_JSON}"
echo "  - ${OUTPUT_DIR}/BENCHMARK_SUMMARY.md"
echo "  - ${OUTPUT_DIR}/metrics_summary.txt"
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_DIR}/metrics_summary.txt"
echo "  cat ${OUTPUT_DIR}/BENCHMARK_SUMMARY.md"
