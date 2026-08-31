cmake_minimum_required(VERSION 3.31.8)

# Set the architecture property before compiler detection and target creation.
# CMake then uses the same targets for compilation AND separable device linking.
include("${CMAKE_CURRENT_LIST_DIR}/BlackwellBuildConfig.cmake")
aisp_blackwell_architectures()
project(cutlass_blackwell_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    message(FATAL_ERROR "This module requires the repository's CUDA 13.0+ toolchain")
endif()

# Find CUTLASS
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass" CACHE PATH "CUTLASS checkout")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_DIR}")
endif()

# Find Python and PyTorch
find_package(Python REQUIRED COMPONENTS Interpreter Development)
aisp_query_torch("${Python_EXECUTABLE}")
# The ABI query and linked TorchConfig must refer to the same Python environment.
set(Torch_DIR "${AISP_TORCH_CMAKE_PREFIX}/Torch")
list(PREPEND CMAKE_PREFIX_PATH "${AISP_TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
find_library(AISP_TORCH_PYTHON_LIBRARY NAMES torch_python
    PATHS "${AISP_TORCH_LIBRARY_DIR}" NO_DEFAULT_PATH NO_CACHE REQUIRED)
message(STATUS "Selected torch ${AISP_TORCH_VERSION}: CXX11 ABI=${AISP_TORCH_CXX11_ABI}")

# Include paths
include_directories(
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${Python_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${AISP_TORCH_INCLUDE}
    ${AISP_TORCH_INCLUDE}/torch/csrc/api/include
)

# Compile the CUTLASS-based GEMM kernel as Python module
Python_add_library(cutlass_blackwell_gemm MODULE
    cutlass_gemm.cu
)

target_compile_options(cutlass_blackwell_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_compile_definitions(cutlass_blackwell_gemm PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${AISP_TORCH_CXX11_ABI}
    TORCH_EXTENSION_NAME=cutlass_blackwell_gemm
)

target_link_libraries(cutlass_blackwell_gemm PRIVATE
    ${TORCH_LIBRARIES}
    "${AISP_TORCH_PYTHON_LIBRARY}"
    Python::Python
    cuda
)

set_target_properties(cutlass_blackwell_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".so"
)
