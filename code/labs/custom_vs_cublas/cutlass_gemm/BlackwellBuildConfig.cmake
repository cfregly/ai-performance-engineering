# Build policy shared by the real CUDA project and host configuration checks.
# This module never substitutes a compiler, CUDA runtime, or torch installation.

function(aisp_blackwell_architectures)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "100a;103a" CACHE STRING
            "Architecture-specific targets for the CUTLASS tcgen05 module")
    endif()
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR "The CUTLASS tcgen05 module requires 100a and/or 103a")
    endif()
    foreach(architecture IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(NOT architecture MATCHES "^(100a|103a)(-real|-virtual)?$")
            message(FATAL_ERROR
                "The CUTLASS tcgen05 module requires 100a and/or 103a; got '${architecture}'")
        endif()
    endforeach()
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" PARENT_SCOPE)
endfunction()

function(aisp_query_torch python_executable)
    execute_process(
        COMMAND "${python_executable}" -c
            "import json, pathlib, torch; from torch.utils.cpp_extension import include_paths; print(json.dumps({'prefix': torch.utils.cmake_prefix_path, 'include': include_paths()[0], 'library': str(pathlib.Path(torch.__file__).resolve().parent / 'lib'), 'abi': int(torch.compiled_with_cxx11_abi()), 'version': torch.__version__}))"
        RESULT_VARIABLE query_result
        OUTPUT_VARIABLE torch_metadata
        ERROR_VARIABLE query_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT query_result EQUAL 0)
        message(FATAL_ERROR "Unable to query the selected Python's torch: ${query_error}")
    endif()
    string(JSON torch_prefix GET "${torch_metadata}" prefix)
    string(JSON torch_include GET "${torch_metadata}" include)
    string(JSON torch_library GET "${torch_metadata}" library)
    string(JSON torch_abi GET "${torch_metadata}" abi)
    string(JSON torch_version GET "${torch_metadata}" version)
    if(NOT torch_abi MATCHES "^[01]$")
        message(FATAL_ERROR "The selected torch reported an invalid CXX11 ABI: ${torch_abi}")
    endif()
    set(AISP_TORCH_CMAKE_PREFIX "${torch_prefix}" PARENT_SCOPE)
    set(AISP_TORCH_INCLUDE "${torch_include}" PARENT_SCOPE)
    set(AISP_TORCH_LIBRARY_DIR "${torch_library}" PARENT_SCOPE)
    set(AISP_TORCH_CXX11_ABI "${torch_abi}" PARENT_SCOPE)
    set(AISP_TORCH_VERSION "${torch_version}" PARENT_SCOPE)
endfunction()
