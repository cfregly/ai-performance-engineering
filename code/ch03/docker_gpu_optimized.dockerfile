# Multi-stage Docker build for optimized GPU performance
FROM nvcr.io/nvidia/pytorch:25.11-py3 AS base

# Install additional dependencies for performance optimization
RUN apt-get update && apt-get install -y \
    numactl \
    libnuma-dev \
    htop \
    iotop \
    libtcmalloc-minimal4 \
    libjemalloc2 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for optimized memory allocation
ENV MALLOC_CONF="narenas:8,dirty_decay_ms:10000,muzzy_decay_ms:10000,background_thread:true"
ENV TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=536870912
ENV TCMALLOC_RELEASE_RATE=16

# Set CUDA environment variables for the pinned PyTorch 2.14.1 CUDA 13 stack
ENV PYTORCH_ALLOC_CONF=max_split_size_mb:512
ENV TORCH_CUDA_ARCH_LIST="10.0;10.3;12.0;12.1+PTX"
ENV CMAKE_CUDA_ARCHITECTURES="100;103;120;121"
ENV CUTLASS_NVCC_ARCHS="100;103;120;121"
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV AISP_VENV=/opt/aisp-torch2141
ENV PYTHONNOUSERSITE=1
ENV LD_PRELOAD=""

# Copy application code, including the reviewed wheels in vendor/python-wheels
COPY . /app
WORKDIR /app

# Install the exact Linux x86_64 and Python 3.12 graph in an isolated environment
RUN python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' && \
    python -m venv "${AISP_VENV}" && \
    "${AISP_VENV}/bin/python" -m pip install \
        --find-links vendor/python-wheels \
        --require-hashes \
        --no-deps \
        -r requirements_torch2141_linux_x86_64.lock && \
    "${AISP_VENV}/bin/python" -m pip check && \
    RUNTIME_WHEEL_LIB_DIRS="$("${AISP_VENV}/bin/python" -c 'from pathlib import Path; import sys; from core.harness.serving_stack import configure_serving_stack_runtime_env; paths = configure_serving_stack_runtime_env(); prefix = Path(sys.prefix).resolve(); assert paths and all(Path(path).resolve().is_relative_to(prefix) for path in paths), paths; print(":".join(paths))')" && \
    test -n "${RUNTIME_WHEEL_LIB_DIRS}" && \
    printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -euo pipefail' \
        'export LD_PRELOAD=' \
        "export LD_LIBRARY_PATH=${RUNTIME_WHEEL_LIB_DIRS}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
        > "${AISP_VENV}/runtime-env.sh" && \
    printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -euo pipefail' \
        'source "${AISP_VENV}/runtime-env.sh"' \
        'exec numactl --interleave=all python "$@"' \
        > /usr/local/bin/aisp-python-entrypoint && \
    chmod 0755 "${AISP_VENV}/runtime-env.sh" /usr/local/bin/aisp-python-entrypoint

# Do not retain CUDA ABI paths from the NGC base image. The entry point restores
# only the locked wheel directories plus the NVIDIA driver mount directories.
ENV PATH="${AISP_VENV}/bin:${PATH}"
ENV LD_LIBRARY_PATH=""

# Runtime stage for smaller final image
FROM base AS runtime

# Set up runtime environment
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Entry point selects the locked wheel libraries before Python starts
ENTRYPOINT ["/usr/local/bin/aisp-python-entrypoint"]
CMD ["train.py"]
