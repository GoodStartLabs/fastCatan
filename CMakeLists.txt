cmake_minimum_required(VERSION 3.27)
project(fastcatan CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Tell CMake to enable LTO when building Release.
include(CheckIPOSupported)
check_ipo_supported(RESULT _IPO_SUPPORTED OUTPUT _IPO_LOG)
if(_IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Common flags. HPC target per PLAN.md: GCC 14.2 + Linux x86_64.
set(FCATAN_RELEASE_FLAGS
    -O3 -march=native -fno-exceptions -fno-rtti -Wall -Wextra)

# Core rules library (static; the linker can fold into anything).
add_library(fastcatan_core STATIC
    src/catan/rules.cpp
    src/catan/obs.cpp
    src/catan/batched_env.cpp
)
target_include_directories(fastcatan_core PUBLIC include)
target_compile_options(fastcatan_core PRIVATE
    $<$<CONFIG:Release>:${FCATAN_RELEASE_FLAGS}>)
set_target_properties(fastcatan_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# OpenMP — optional. Macs need libomp from homebrew; HPC GCC has it built-in.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fastcatan_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(fastcatan_core PUBLIC FCATAN_HAVE_OPENMP=1)
    message(STATUS "OpenMP enabled")
else()
    message(STATUS "OpenMP not found — building single-threaded")
endif()

# ctypes shared lib (used by Python tests).
add_library(fastcatan SHARED tools/c_api.cpp)
target_link_libraries(fastcatan PRIVATE fastcatan_core)
target_compile_options(fastcatan PRIVATE
    $<$<CONFIG:Release>:${FCATAN_RELEASE_FLAGS}>)

# --------------------------------------------------------------------
# Python extension via nanobind (built when invoked through scikit-build-core).
# --------------------------------------------------------------------
if(SKBUILD)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_fastcatan
        STABLE_ABI
        NB_STATIC
        bindings/pycatan/bindings.cpp
    )
    target_link_libraries(_fastcatan PRIVATE fastcatan_core)
    # Nanobind needs RTTI + exceptions, so DON'T apply our -fno-* flags here.
    # Core library is still built without them; this binding TU compiles
    # with the compiler defaults.
    target_compile_options(_fastcatan PRIVATE -O3 -march=native)

    install(TARGETS _fastcatan LIBRARY DESTINATION fastcatan)
    install(FILES python/fastcatan/__init__.py
                  python/fastcatan/gym_env.py
                  python/fastcatan/pettingzoo_env.py
            DESTINATION fastcatan)
endif()

# Throughput benchmarks.
add_executable(bench_step    bench/bench_step.cpp)
add_executable(bench_batched bench/bench_batched.cpp)
target_link_libraries(bench_step    PRIVATE fastcatan_core)
target_link_libraries(bench_batched PRIVATE fastcatan_core)
foreach(bench_target bench_step bench_batched)
    target_compile_options(${bench_target} PRIVATE
        $<$<CONFIG:Release>:${FCATAN_RELEASE_FLAGS}>)
endforeach()
