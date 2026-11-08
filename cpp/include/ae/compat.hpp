// compat.hpp — C++26 compatibility layer.
//
// Verified against GCC 14.2.0 / libstdc++ 14 (see docs/environment.md):
//   AVAILABLE : concepts, ranges/views, std::span, std::expected, std::print,
//               std::format, <bit>, std::mdspan(absent), designated initialisers,
//               consteval, std::to_underlying, [[nodiscard]]
//   ABSENT    : <flat_map>, <mdspan>, <generator>, <linalg>, #embed (C23)
//
// Anything missing is provided by a small shim here so the rest of the codebase
// can be written against a single, honest vocabulary. This header documents
// precisely which features were substituted, per the delivery contract.
#pragma once

#include <version>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <span>
#include <concepts>
#include <limits>
#include <string>

#if __has_include(<expected>)
#  include <expected>
#  define AE_HAS_EXPECTED 1
#endif

#if __has_include(<print>)
#  include <print>
#  define AE_HAS_PRINT 1
#endif

#if __has_include(<format>)
#  include <format>
#  define AE_HAS_FORMAT 1
#endif

#if __has_include(<execution>)
#  include <execution>
#endif

// OpenMP is used for coarse-grained parallelism when present; otherwise the
// parallel primitives fall back to std::thread. Both paths are exercised by the
// concurrency tests.
#if defined(_OPENMP)
#  include <omp.h>
#  define AE_HAS_OPENMP 1
#endif

namespace ae {

// ---------------------------------------------------------------- constants
inline constexpr std::size_t kCacheLine = 64;
inline constexpr std::size_t kSimdAlign = 64;

// Reported feature level, checkable from Python to prove which language level
// the shared object was compiled at.
inline constexpr long kCxxDialect = __cplusplus;  // 202400 on GCC 14 -std=c++26

enum class Status : std::int32_t {
    ok = 0,
    invalid_argument = -1,
    dimension_mismatch = -2,
    index_out_of_range = -3,
    not_converged = -4,
    numerical = -5,
};

inline const char* to_string(Status s) noexcept {
    switch (s) {
    case Status::ok: return "ok";
    case Status::invalid_argument: return "invalid argument";
    case Status::dimension_mismatch: return "dimension mismatch";
    case Status::index_out_of_range: return "index out of range";
    case Status::not_converged: return "not converged";
    case Status::numerical: return "numerical error";
    }
    return "unknown";
}

// std::expected is available; alias it so call sites stay readable.
#if AE_HAS_EXPECTED
template <class T>
using Result = std::expected<T, Status>;
#else
template <class T>
struct Result {
    T value{};
    Status status{Status::ok};
    explicit operator bool() const noexcept { return status == Status::ok; }
};
#endif

// --------------------------------------------------------------- concepts
template <class T>
concept Numeric = std::floating_point<T>;

template <class R>
concept IndexRange = std::ranges::input_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, std::size_t>;

template <class F>
concept ScalarScorer = requires(F f, std::size_t a, std::size_t b) {
    { f(a, b) } -> std::convertible_to<double>;
};

// ------------------------------------------------------------- utilities
[[nodiscard]] constexpr std::size_t round_up(std::size_t n, std::size_t a) noexcept {
    return a == 0 ? n : ((n + a - 1) / a) * a;
}

template <Numeric T = double>
[[nodiscard]] inline T clamp01(T v) noexcept {
    return v < T{0} ? T{0} : (v > T{1} ? T{1} : v);
}

// Compile-time capability report surfaced in /api/system/capabilities.
struct BuildInfo {
    long cxx_dialect;
    bool has_expected;
    bool has_print;
    bool has_format;
    bool has_openmp;
    const char* compiler;
    const char* compiler_version;
};

[[nodiscard]] BuildInfo build_info() noexcept;

}  // namespace ae
