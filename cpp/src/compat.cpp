// compat.cpp — build information reporting.
#include "ae/compat.hpp"

namespace ae {

BuildInfo build_info() noexcept {
    BuildInfo info{};
    info.cxx_dialect = static_cast<long>(__cplusplus);
#if defined(__clang__)
    info.compiler = "clang";
    info.compiler_version = __clang_version__;
#elif defined(__GNUC__)
    info.compiler = "gcc";
    info.compiler_version = __VERSION__;
#else
    info.compiler = "unknown";
    info.compiler_version = "";
#endif
#if defined(AE_HAS_EXPECTED)
    info.has_expected = true;
#else
    info.has_expected = false;
#endif
#if defined(AE_HAS_PRINT)
    info.has_print = true;
#else
    info.has_print = false;
#endif
#if defined(AE_HAS_FORMAT)
    info.has_format = true;
#else
    info.has_format = false;
#endif
#if defined(AE_HAS_OPENMP)
    info.has_openmp = true;
#else
    info.has_openmp = false;
#endif
    return info;
}

// Ensure <optional> is visible for ae::stats.hpp users that only include compat.hpp
// through stats.hpp; kept here to document the dependency explicitly.
}  // namespace ae
