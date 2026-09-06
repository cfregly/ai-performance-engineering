FROM nvcr.io/nvidia/pytorch:26.01-py3@sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    libboost-program-options-dev \
    && rm -rf /var/lib/apt/lists/*

# Match legacy build-toolchain provenance used by old_container.
RUN python -m pip install \
    "pip==25.3" \
    "setuptools==80.10.1" \
    "wheel==0.45.1"

# Keep utility deps aligned with the open decoupled image.
RUN python -m pip install numpy matplotlib

# Match legacy DeepGEMM install provenance (VCS URL + pinned commit).
RUN python -m pip uninstall -y deep_gemm || true \
    && python -m pip install --no-build-isolation \
    "deep_gemm @ git+https://github.com/deepseek-ai/DeepGEMM.git@0f5f2662027f0db05d4e3f6a94e56e2d8fc45c51"

# nvbandwidth (open-source) for dedicated bandwidth bundle.
ARG NVBANDWIDTH_COMMIT=82fc4e8c6afa0babb8687793678f615b3b8d793e
RUN git init /opt/nvbandwidth \
    && git -C /opt/nvbandwidth remote add origin https://github.com/NVIDIA/nvbandwidth.git \
    && git -C /opt/nvbandwidth fetch --depth 1 --no-tags origin "${NVBANDWIDTH_COMMIT}" \
    && git -C /opt/nvbandwidth checkout --detach "${NVBANDWIDTH_COMMIT}" \
    && test "$(git -C /opt/nvbandwidth rev-parse HEAD)" = "${NVBANDWIDTH_COMMIT}" \
    && cmake -S /opt/nvbandwidth -B /opt/nvbandwidth/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/nvbandwidth/build -j"$(nproc)" \
    && cp /opt/nvbandwidth/build/nvbandwidth /usr/local/bin/nvbandwidth

WORKDIR /workspace

CMD ["bash"]
