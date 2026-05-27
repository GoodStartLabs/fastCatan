cmake_minimum_required(VERSION 3.27)
project(fastcatan CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)

# Tell CMake to enable LTO when building Release.
include(CheckIPOSupported)
check_ipo_supported(RESULT _IPO_SUPPORTED OUTPUT _IPO_LOG)
if(_IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Common flags. Target per PLAN.md: GCC 14.2 + Linux x86_64.
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

# OpenMP — optional. Macs need libomp from homebrew; Linux GCC has it built-in.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fastcatan_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(fastcatan_core PUBLIC FCATAN_HAVE_OPENMP=1)
    message(STATUS "OpenMP enabled")
else()
    message(STATUS "OpenMP not found — building single-threaded")
endif()

# --------------------------------------------------------------------
# Standalone C++ benchmarks — pure-C++ throughput floor, no Python.
# Built only in the standalone CMake build (not the pip wheel):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#   build/bench_step                  # single-env: step_one + random play
#   build/bench_batched 4096 5000     # batched (uses OpenMP if available)
# These corroborate the nanobind-dispatch isolation in
# bench/bench_throughput.py.
# --------------------------------------------------------------------
if(NOT SKBUILD)
    foreach(_bench bench_step bench_batched)
        add_executable(${_bench} bench/${_bench}.cpp)
        target_link_libraries(${_bench} PRIVATE fastcatan_core)
        target_compile_options(${_bench} PRIVATE
            $<$<CONFIG:Release>:${FCATAN_RELEASE_FLAGS}>)
    endforeach()

    # ----------------------------------------------------------------
    # Invariant fuzz harness — the M1 correctness gate (PLAN.md §M1,
    # Verification). Plays random-legal games in pure C++ and checks
    # per-step game invariants; the readable spec lives in
    # sim/tests/test_invariants.py. OpenMP pragmas live in this TU, so it
    # links OpenMP directly (the bench targets inherit it only via core).
    #   build/fuzz_invariants 10000000     # full thesis gate (~minutes)
    #   ctest -R invariants                # fast CI smoke (100k games)
    # ----------------------------------------------------------------
    add_executable(fuzz_invariants sim/fuzz_invariants.cpp)
    target_link_libraries(fuzz_invariants PRIVATE fastcatan_core)
    target_compile_options(fuzz_invariants PRIVATE
        $<$<CONFIG:Release>:${FCATAN_RELEASE_FLAGS}>)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fuzz_invariants PRIVATE OpenMP::OpenMP_CXX)
    endif()

    enable_testing()
    add_test(NAME invariants COMMAND fuzz_invariants 100000)
    set_tests_properties(invariants PROPERTIES
        TIMEOUT 900
        PASS_REGULAR_EXPRESSION "invariant failures: 0")
endif()

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
            DESTINATION fastcatan)
endif()

# --------------------------------------------------------------------
# IDE indexing target for bindings.cpp (clangd / VS Code).
# Not built by default; gives compile_commands.json an entry so clangd
# can resolve nanobind + Python headers. Skipped when configured via
# scikit-build (the real Python module is built there).
# --------------------------------------------------------------------
if(NOT SKBUILD)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module QUIET)
    if(Python_FOUND)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import nanobind, sys; sys.stdout.write(nanobind.include_dir())"
            OUTPUT_VARIABLE _NB_INCLUDE
            RESULT_VARIABLE _NB_RC
        )
        if(_NB_RC EQUAL 0 AND EXISTS "${_NB_INCLUDE}")
            add_library(bindings_ide OBJECT EXCLUDE_FROM_ALL
                bindings/pycatan/bindings.cpp)
            target_include_directories(bindings_ide PRIVATE
                ${_NB_INCLUDE}
                ${Python_INCLUDE_DIRS})
            target_link_libraries(bindings_ide PRIVATE fastcatan_core)
            target_compile_options(bindings_ide PRIVATE -O0)
            message(STATUS "IDE indexing target enabled for bindings.cpp")
        else()
            message(STATUS "nanobind not found in active Python; bindings.cpp IDE target skipped")
        endif()
    endif()
endif()
