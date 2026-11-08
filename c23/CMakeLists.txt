cmake_minimum_required(VERSION 3.20)
project(aenum C)

# Additive global-configuration targets so callers can opt in independently.
add_library(aenum_warnings INTERFACE)
target_compile_options(aenum_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    -Wcast-align -Wstrict-prototypes -Wmissing-prototypes -Wdouble-promotion
    -Wformat=2 -Wundef)

add_library(aenum_sanitize INTERFACE)
if(AENUM_SANITIZE)
    target_compile_options(aenum_sanitize INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(aenum_sanitize INTERFACE -fsanitize=address,undefined)
endif()

add_library(aenum_opt INTERFACE)
if(NOT AENUM_NO_SIMD)
    target_compile_options(aenum_opt INTERFACE -fno-math-errno -fno-trapping-math)
endif()

add_library(aenum STATIC src/core.c src/stats.c src/image.c)
set_target_properties(aenum PROPERTIES C_STANDARD 23 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)

target_include_directories(aenum PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aenum PRIVATE aenum_opt aenum_warnings)
if(AENUM_SANITIZE)
    target_link_libraries(aenum PRIVATE aenum_sanitize)
endif()
if(AENUM_NO_SIMD)
    target_compile_definitions(aenum PUBLIC AE_NO_SIMD=1)
endif()
target_compile_options(aenum PRIVATE -O3)

# Shared variant (used by pybind11 binding and the C++ core).
add_library(aenum_shared SHARED src/core.c src/stats.c src/image.c)
set_target_properties(aenum_shared PROPERTIES C_STANDARD 23 C_STANDARD_REQUIRED ON
                                              C_EXTENSIONS OFF OUTPUT_NAME aenum)
target_include_directories(aenum_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(aenum_shared PRIVATE -O3 -fPIC)
if(AENUM_NO_SIMD)
    target_compile_definitions(aenum_shared PUBLIC AE_NO_SIMD=1)
endif()

if(AENUM_BUILD_TESTS)
    enable_testing()
    add_executable(aenum_tests tests/test_aenum.c)
    target_link_libraries(aenum_tests PRIVATE aenum m)
    target_link_libraries(aenum_tests PRIVATE aenum_sanitize)
    target_include_directories(aenum_tests PRIVATE tests)
    add_test(NAME aenum_tests COMMAND aenum_tests)
endif()
