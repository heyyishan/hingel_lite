/* aenum_bridge.h — C-compatible declarations used by the C++ core.
 *
 * The C++ translation units include this instead of <aenum.h> so the public C
 * library headers stay untouched by C++ name lookup quirks (e.g. `restrict`).
 * Signatures are identical; static_asserts in aenum_bridge_check.cpp enforce
 * that the two declarations cannot drift apart.
 */
#ifndef AE_AENUM_BRIDGE_H
#define AE_AENUM_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ae_status_bridge {
    AE_OK = 0,
    AE_ERR_NULL = -1,
    AE_ERR_SIZE_ZERO = -2,
    AE_ERR_DIM_MISMATCH = -3,
    AE_ERR_INDEX_RANGE = -4,
    AE_ERR_ALLOC = -5,
    AE_ERR_NOT_NORMALIZED = -6,
    AE_ERR_NAN = -7,
    AE_ERR_SINGULAR = -8,
    AE_ERR_UNSUPPORTED = -9,
    AE_ERR_OVERFLOW = -10,
    AE_ERR_ALIGNMENT = -11
};

void* ae_aligned_alloc(size_t bytes, size_t alignment);
void ae_aligned_free(void* p);
int ae_vec_norm(const double* x, size_t n, double* norm_out);
int ae_vec_normalize(const double* x, size_t n, double* out);
int ae_vec_dot(const double* a, const double* b, size_t n, double* out);
int ae_cosine_raw(const double* a, const double* b, size_t n, double* out);
int ae_cosine_unit(const double* a, const double* b, size_t n, double* out);
int ae_l2sq(const double* a, const double* b, size_t n, double* out);
int ae_batch_cosine_unit(const double* q, size_t m, const double* x, size_t n,
                         size_t d, double* out);
int ae_topk(const double* x, size_t n, size_t k, size_t* idx_out, double* val_out);
int ae_softmax(const double* x, size_t n, double* out);
int ae_stats_compute(const double* x, size_t n, void* out);
int ae_entropy(const double* p, size_t n, double* out);
uint32_t ae_cpu_features(void);
const char* ae_cpu_target_name(void);
const char* ae_status_str(int s);

#ifdef __cplusplus
}
#endif
#endif
