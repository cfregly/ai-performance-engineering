#!/bin/bash
# Orchestration script for multi-GPU B200 load testing
# This script coordinates load testing, power monitoring, and result collection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default parameters
DURATION="${1:-300}"  # 5 minutes default
TARGET_QPS="${2:-100}"
OUTPUT_DIR="${3:-load_test_multigpu_$(date +%Y%m%d_%H%M%S)}"

echo "================================================================================"
echo "Multi-GPU B200 Load Test Orchestration"
echo "================================================================================"
echo ""
echo "Duration: ${DURATION}s"
echo "Target QPS: ${TARGET_QPS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Step 0: Pre-flight checks
echo -e "${BLUE}Step 0: Pre-flight checks${NC}"
echo ""

# Check GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected GPUs: ${NUM_GPUS}"

if [ "${NUM_GPUS}" -lt 2 ]; then
    echo -e "${RED}ERROR: This script requires >=2 GPUs (found ${NUM_GPUS})${NC}"
    exit 1
fi

# Check for B200
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "B200"; then
    echo -e "${GREEN}B200 GPUs detected${NC}"
else
    echo -e "${YELLOW}⚠ B200 GPUs not detected - running on available hardware${NC}"
fi

# Check NCCL environment
echo ""
echo "NCCL Configuration:"
echo "  NCCL_NVLS_ENABLE: ${NCCL_NVLS_ENABLE:-not set}"
echo "  NCCL_P2P_LEVEL: ${NCCL_P2P_LEVEL:-not set}"

# Set optimal NCCL settings if not already set
# NCCL finds NVLink and NVLink-C2C through topology discovery. It has no
# environment switch that turns either link on.
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export NCCL_ALGO="${NCCL_ALGO:-Tree,Ring,NVLS}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

echo ""
echo -e "${GREEN}Environment configured${NC}"
echo ""

# Step 1: Start power monitoring in background
echo -e "${BLUE}Step 1: Starting power monitoring${NC}"
echo ""

POWER_OUTPUT="${OUTPUT_DIR}/power_metrics.json"
POWER_LOG="${OUTPUT_DIR}/power_monitor.log"

# Start power monitor in background
python3 core/scripts/power_monitor.py \
    --interval 0.5 \
    --output-json "${POWER_OUTPUT}" \
    -- sleep $((DURATION + 30)) > "${POWER_LOG}" 2>&1 &

POWER_PID=$!
echo "Power monitoring started (PID: ${POWER_PID})"
echo ""

# Give power monitor time to initialize
sleep 2

# Step 2: Run load test
echo -e "${BLUE}Step 2: Running load test${NC}"
echo ""

LOAD_TEST_OUTPUT="${OUTPUT_DIR}/load_test_results.json"
LOAD_TEST_LOG="${OUTPUT_DIR}/load_test.log"

echo "Starting distributed load test..."
echo "  Duration: ${DURATION}s"
echo "  Target QPS: ${TARGET_QPS}"
echo "  GPUs: ${NUM_GPUS}"
echo ""

# Run the load test
# Set PYTHONPATH to repo root for imports
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if torchrun --nproc_per_node="${NUM_GPUS}" \
    "${REPO_ROOT}/ch16/inference_server_load_test.py" \
    --duration "${DURATION}" \
    --target-qps "${TARGET_QPS}" \
    --output-json "${LOAD_TEST_OUTPUT}" \
    > "${LOAD_TEST_LOG}" 2>&1; then
    echo -e "${GREEN}Load test complete${NC}"
    LOAD_TEST_SUCCESS=1
else
    echo -e "${RED}✗ Load test failed${NC}"
    LOAD_TEST_SUCCESS=0
fi

echo ""

# Step 3: Stop power monitoring
echo -e "${BLUE}Step 3: Stopping power monitoring${NC}"
echo ""

# Kill the power monitor sleep process
kill ${POWER_PID} 2>/dev/null || true
wait ${POWER_PID} 2>/dev/null || true

# Wait for power monitor to finish writing
sleep 2

if [ -f "${POWER_OUTPUT}" ]; then
    echo -e "${GREEN}Power metrics collected${NC}"
else
    echo -e "${YELLOW}⚠ Power metrics not available${NC}"
fi

echo ""

# Step 4: Collect system metrics
echo -e "${BLUE}Step 4: Collecting system metrics${NC}"
echo ""

SYSTEM_INFO="${OUTPUT_DIR}/system_info.txt"

{
    echo "=== System Information ==="
    echo ""
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo ""
    echo "=== GPU Information ==="
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
    echo ""
    echo "=== CPU Information ==="
    lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket"
    echo ""
    echo "=== Memory Information ==="
    free -h
    echo ""
    echo "=== NVLink Topology ==="
    nvidia-smi nvlink --status
    echo ""
    echo "=== NCCL Environment ==="
    env | grep NCCL | sort
} > "${SYSTEM_INFO}" 2>&1

echo -e "${GREEN}System info collected${NC}"
echo ""

# Step 5: Calculate cost metrics
if [ -f "${POWER_OUTPUT}" ] && [ -f "${LOAD_TEST_OUTPUT}" ]; then
    echo -e "${BLUE}Step 5: Calculating cost per token${NC}"
    echo ""
    
    # Extract throughput from load test results
    THROUGHPUT=$(python3 -c "import json; data=json.load(open('${LOAD_TEST_OUTPUT}')); print(data.get('tokens_per_sec', 0))" 2>/dev/null || echo "0")
    
    if [ "$(echo "$THROUGHPUT > 0" | bc)" -eq 1 ]; then
        COST_REPORT="${OUTPUT_DIR}/cost_analysis.md"
        python3 core/scripts/calculate_cost_per_token.py \
            --power-json "${POWER_OUTPUT}" \
            --throughput "${THROUGHPUT}" \
            --workload-name "MultiGPU_LoadTest" \
            --output "${COST_REPORT}" \
            --output-json "${OUTPUT_DIR}/cost_metrics.json" \
            > "${OUTPUT_DIR}/cost_calculation.log" 2>&1 || true
        
        if [ -f "${COST_REPORT}" ]; then
            echo -e "${GREEN}Cost analysis complete${NC}"
        else
            echo -e "${YELLOW}⚠ Cost analysis unavailable${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Throughput not available, skipping cost analysis${NC}"
    fi
else
    echo -e "${YELLOW}Step 5: Skipping cost analysis (missing data)${NC}"
fi

echo ""

# Step 6: Generate summary report
echo -e "${BLUE}Step 6: Generating summary report${NC}"
echo ""

SUMMARY_REPORT="${OUTPUT_DIR}/SUMMARY.md"

{
    echo "# Multi-GPU Load Test Summary"
    echo ""
    echo "**Date**: $(date)"
    echo "**Duration**: ${DURATION}s"
    echo "**Target QPS**: ${TARGET_QPS}"
    echo ""
    echo "## Test Parameters"
    echo ""
    echo "- Duration: ${DURATION} seconds"
    echo "- Target QPS: ${TARGET_QPS}"
    echo "- Number of GPUs: ${NUM_GPUS}"
    echo ""
    
    if [ -f "${LOAD_TEST_OUTPUT}" ]; then
        echo "## Load Test Results"
        echo ""
        python3 -c "
import json
try:
    data = json.load(open('${LOAD_TEST_OUTPUT}'))
    print(f\"- **Actual QPS**: {data.get('actual_qps', 'N/A'):.1f}\")
    print(f\"- **Tokens/sec**: {data.get('tokens_per_sec', 'N/A'):.1f}\")
    print(f\"- **Median Latency**: {data.get('latency_p50_ms', 'N/A'):.1f} ms\")
    print(f\"- **P95 Latency**: {data.get('latency_p95_ms', 'N/A'):.1f} ms\")
    print(f\"- **P99 Latency**: {data.get('latency_p99_ms', 'N/A'):.1f} ms\")
    print(f\"- **Total Requests**: {data.get('total_requests', 'N/A')}\")
    print(f\"- **Completed**: {data.get('completed_requests', 'N/A')}\")
except Exception as e:
    print(f\"Error reading results: {e}\")
"
        echo ""
    fi
    
    if [ -f "${POWER_OUTPUT}" ]; then
        echo "## Power Metrics"
        echo ""
        python3 -c "
import json
try:
    data = json.load(open('${POWER_OUTPUT}'))
    power = data['total_power']
    print(f\"- **Average Power**: {power['avg_watts']:.1f} W\")
    print(f\"- **Peak Power**: {power['max_watts']:.1f} W\")
    print(f\"- **Total Energy**: {data['energy_joules']/1000:.1f} kJ\")
except Exception as e:
    print(f\"Error reading power metrics: {e}\")
"
        echo ""
    fi
    
    if [ -f "${OUTPUT_DIR}/cost_metrics.json" ]; then
        echo "## Cost Analysis"
        echo ""
        python3 -c "
import json
try:
    data = json.load(open('${OUTPUT_DIR}/cost_metrics.json'))
    print(f\"- **Cost per 1M tokens**: \${data['cost_per_million_tokens_usd']:.2f}\")
    print(f\"- **Tokens per dollar**: {data['tokens_per_dollar']:,.0f}\")
    print(f\"- **Operating cost**: \${data['cost_per_hour_usd']:.2f}/hour\")
except Exception as e:
    print(f\"Error reading cost metrics: {e}\")
"
        echo ""
    fi
    
    echo "## Files Generated"
    echo ""
    echo "- \`load_test_results.json\`: Detailed load test metrics"
    echo "- \`power_metrics.json\`: Power consumption data"
    echo "- \`cost_analysis.md\`: Detailed cost breakdown"
    echo "- \`system_info.txt\`: Hardware and environment info"
    echo "- \`load_test.log\`: Load test execution log"
    echo ""
    
    if [ ${LOAD_TEST_SUCCESS} -eq 1 ]; then
        echo "## Status"
        echo ""
        echo "[OK] **SUCCESS**: All tests completed successfully"
    else
        echo "## Status"
        echo ""
        echo "ERROR: **FAILURE**: Load test encountered errors"
        echo ""
        echo "Check \`load_test.log\` for details."
    fi
    
} > "${SUMMARY_REPORT}"

echo -e "${GREEN}Summary report generated${NC}"
echo ""

# Display summary
cat "${SUMMARY_REPORT}"

echo ""
echo "================================================================================"
echo "Load Test Complete"
echo "================================================================================"
echo ""
echo "All results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Key files:"
echo "  - SUMMARY.md: Test summary and results"
echo "  - load_test_results.json: Detailed metrics"
echo "  - power_metrics.json: Power consumption"
echo "  - cost_analysis.md: Cost breakdown"
echo "  - system_info.txt: Hardware info"
echo ""

if [ ${LOAD_TEST_SUCCESS} -eq 1 ]; then
    echo -e "${GREEN}[OK] Test completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}ERROR: Test completed with errors${NC}"
    echo "Check ${LOAD_TEST_LOG} for details"
    exit 1
fi
