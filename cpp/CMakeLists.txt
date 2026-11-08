cmake_minimum_required(VERSION 3.20)
project(attraction_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(ae_warnings INTERFACE)
target_compile_options(ae_warnings INTERFACE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor
    -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion
    -Wdouble-promotion)
add_library(ae_sanitize INTERFACE)
if(AE_SANITIZE)
    target_compile_options(ae_sanitize INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(ae_sanitize INTERFACE -fsanitize=address,undefined)
endif()
add_library(ae_tsan INTERFACE)
if(AE_TSAN)
    target_compile_options(ae_tsan INTERFACE -fsanitize=thread -g)
    target_link_options(ae_tsan INTERFACE -fsanitize=thread)
endif()

# ------------------------------- C23 library (sources compiled directly) -----
set(AENUM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../c23)
add_library(aenum STATIC ${AENUM_DIR}/src/core.c ${AENUM_DIR}/src/stats.c ${AENUM_DIR}/src/image.c)
target_include_directories(aenum PUBLIC ${AENUM_DIR}/include)
set_target_properties(aenum PROPERTIES C_STANDARD 23 C_STANDARD_REQUIRED ON)
target_compile_options(aenum PRIVATE -O3)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: parallel paths enabled")
else()
    message(STATUS "OpenMP not found: falling back to std::thread pools")
endif()

# ------------------------------- C++ core -----------------------------------
add_library(ae_core STATIC
    src/compat.cpp
    src/similarity.cpp
    src/ranking.cpp
    src/optimizer.cpp
    src/stats.cpp
    src/compat_matcher.cpp
    src/capacity.cpp
)
target_include_directories(ae_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ae_core PUBLIC aenum)
target_link_libraries(ae_core PRIVATE ae_warnings)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ae_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ae_core PUBLIC AE_HAS_OPENMP=1)
endif()
if(AE_SANITIZE)
    target_link_libraries(ae_core PUBLIC ae_sanitize)
endif()
if(AE_TSAN)
    target_link_libraries(ae_core PUBLIC ae_tsan)
endif()

add_library(ae_core_shared SHARED
    src/compat.cpp src/similarity.cpp src/ranking.cpp src/optimizer.cpp src/stats.cpp src/compat_matcher.cpp src/capacity.cpp
    ${AENUM_DIR}/src/core.c ${AENUM_DIR}/src/stats.c ${AENUM_DIR}/src/image.c)
target_include_directories(ae_core_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${AENUM_DIR}/include)
set_target_properties(ae_core_shared PROPERTIES OUTPUT_NAME ae_core)
target_compile_options(ae_core_shared PRIVATE -O3 -fPIC)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ae_core_shared PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ae_core_shared PUBLIC AE_HAS_OPENMP=1)
endif()
if(AE_SANITIZE)
    target_link_libraries(ae_core_shared PRIVATE ae_sanitize)
endif()

# ------------------------------- tests --------------------------------------
if(AE_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_executable(ae_tests
        tests/test_main.cpp
        tests/test_similarity.cpp
        tests/test_ranking.cpp
        tests/test_optimizer.cpp
        tests/test_stats.cpp
        tests/test_compat_matcher.cpp
        tests/test_concurrency.cpp)
    target_link_libraries(ae_tests PRIVATE ae_core GTest::gtest GTest::gtest_main pthread)
    if(AE_SANITIZE)
        target_link_libraries(ae_tests PRIVATE ae_sanitize)
    endif()
    add_test(NAME ae_tests COMMAND ae_tests)
endif()

# ------------------------------- benchmarks ---------------------------------
if(AE_BUILD_BENCH)
    find_package(benchmark REQUIRED)
    add_executable(ae_bench bench/core_bench.cpp)
    target_link_libraries(ae_bench PRIVATE ae_core benchmark::benchmark benchmark::benchmark_main pthread)
endif()

# ------------------------------- Python bindings ----------------------------
if(AE_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
    find_package(pybind11 CONFIG REQUIRED)
    # The CMake target is ae_python; the imported module name is ae_core
    # (set by PYBIND11_MODULE in the source).
    pybind11_add_module(ae_python bindings/ae_bindings.cpp)
    target_link_libraries(ae_python PRIVATE ae_core_shared)
    target_include_directories(ae_python PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(ae_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/../python/attraction_engine/_native)
    if(AE_SANITIZE)
        target_link_libraries(ae_python PRIVATE ae_sanitize)
    endif()
endif()
