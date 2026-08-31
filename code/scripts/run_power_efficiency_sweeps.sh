#!/bin/bash
# Run power efficiency sweeps across model sizes and precisions
# Usage: ./scripts/run_power_efficiency_sweeps.sh [output_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${1:-power_results_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

export TORCH_CUDA_ARCH_LIST=120
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "================================================================================"
echo "Power Efficiency Baseline Expansion"
echo "================================================================================"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Model configurations: (layers, d_model, heads, d_ff) -> approximate size
# 8B: (32, 5120, 40, 20480)
# 16B: (48, 6144, 48, 24576)
# Larger models may OOM, so we focus on these

declare -a MODEL_CONFIGS=(
    "8B:32:5120:40:20480"
    "16B:48:6144:48:24576"
)

declare -a BATCH_SIZES=(2 3 4)
declare -a PRECISIONS=("fp16" "bf16" "fp8_te")

SEQUENCE_LENGTH=4096

for model_config in "${MODEL_CONFIGS[@]}"; do
    IFS=':' read -r model_name layers d_model heads d_ff <<< "${model_config}"
    
    echo "================================================================================"
    echo "Model: ${model_name} (${layers} layers, d_model=${d_model}, heads=${heads})"
    echo "================================================================================"
    echo ""
    
    for batch_size in "${BATCH_SIZES[@]}"; do
        echo "  Batch size: ${batch_size}"
        
        # Create output file for this configuration
        output_file="${OUTPUT_DIR}/precision_power_results_${model_name}_batch${batch_size}.json"
        
        # Build precision modes list
        modes_str=$(IFS=' '; echo "${PRECISIONS[*]}")
        
        echo "  Running precision sweep: ${modes_str}"
        echo "  Output: ${output_file}"
        echo ""
        
        # Run precision power sweep
        python -m scripts.precision_power_sweep \
            --sequence-length ${SEQUENCE_LENGTH} \
            --model-layers ${layers} \
            --model-d-model ${d_model} \
            --model-heads ${heads} \
            --model-d-ff ${d_ff} \
            --batch-size ${batch_size} \
            --modes ${modes_str} \
            --skip-compile \
            --attention-backend sdpa \
            --output-json "${output_file}" \
            --output-markdown "${output_file%.json}.md" \
            2>&1 | tee "${output_file%.json}.log"
        
        if [ $? -eq 0 ]; then
            echo "  [OK] Completed: ${output_file}"
        else
            echo "  WARNING: Failed: Check ${output_file%.json}.log"
        fi
        echo ""
    done
done

# Generate summary
echo "================================================================================"
echo "Generating Summary"
echo "================================================================================"
echo ""

SUMMARY_FILE="${OUTPUT_DIR}/POWER_EFFICIENCY_SUMMARY.md"
cat > "${SUMMARY_FILE}" << 'EOF'
# Power Efficiency Baseline Expansion

Generated: $(date)

## Overview

This directory contains power efficiency measurements across:
- Model sizes: 8B, 16B parameters
- Batch sizes: 2, 3, 4
- Precision modes: FP16, BF16, FP8 (transformer_engine)

## Files

EOF

# List all JSON files
for json_file in "${OUTPUT_DIR}"/precision_power_results_*.json; do
    if [ -f "${json_file}" ]; then
        basename=$(basename "${json_file}")
        echo "- \`${basename}\`: Results for this configuration" >> "${SUMMARY_FILE}"
        echo "  - Markdown: \`${basename%.json}.md\`" >> "${SUMMARY_FILE}"
        echo "  - Log: \`${basename%.json}.log\`" >> "${SUMMARY_FILE}"
    fi
done

cat >> "${SUMMARY_FILE}" << 'EOF'

## Analysis

### Tokens per Joule

Compare tokens/joule across:
- Precision modes (FP16 vs BF16 vs FP8)
- Batch sizes (2 vs 3 vs 4)
- Model sizes (8B vs 16B)

### Cost per Million Tokens

Calculate cost efficiency:
- Electricity cost: $0.16/kWh (default)
- PUE: 1.5 (default)
- Compare across configurations

## Next Steps

1. Review individual result files
2. Compare tokens/joule across configurations
3. Identify optimal precision/batch size combinations
4. Update `docs/reference/performance_baseline.md` with results
5. Publish comprehensive power efficiency guide

## Related Documentation

- `docs/playbooks/torch_compile_troubleshooting.md` - Note: Using --skip-compile for reliability
- `scripts/precision_power_sweep.py` - Tool used for measurements
EOF

echo "Summary written to: ${SUMMARY_FILE}"
echo ""
echo "================================================================================"
echo "Power Efficiency Sweeps Complete"
echo "================================================================================"
echo ""
echo "Results directory: ${OUTPUT_DIR}"
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "To analyze results:"
echo "  python -m core.analysis.power_efficiency_analyzer ${OUTPUT_DIR}/*.json"
