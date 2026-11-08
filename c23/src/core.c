/* core.c — status, CPU dispatch, owned memory, vector kernels */
#include "aenum.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <immintrin.h>

/* ------------------------------------------------------------------ status */
const char *ae_status_str(ae_status s)
{
    switch (s) {
    case AE_OK:                 return "ok";
    case AE_ERR_NULL:           return "null pointer";
    case AE_ERR_SIZE_ZERO:      return "zero length";
    case AE_ERR_DIM_MISMATCH:   return "dimension mismatch";
    case AE_ERR_INDEX_RANGE:    return "index out of range";
    case AE_ERR_ALLOC:          return "allocation failure";
    case AE_ERR_NOT_NORMALIZED: return "vector not normalised";
    case AE_ERR_NAN:            return "non-finite value";
    case AE_ERR_SINGULAR:       return "singular matrix";
    case AE_ERR_UNSUPPORTED:    return "unsupported";
    case AE_ERR_OVERFLOW:       return "overflow";
    case AE_ERR_ALIGNMENT:      return "alignment violation";
    }
    return "unknown status";
}

/* --------------------------------------------------------------------- CPU */
static _Atomic uint32_t g_cpu_mask = 0;
static _Atomic int      g_force    = 0;   /* 0 auto, 1 prefer best, -1 scalar */

/* libgcc exposes these on x86; they are the same primitives used by
 * __attribute__((target_clones)). We use them only on GCC/Clang/x86. */
#if defined(__x86_64__) || defined(__i386__)
#  define AE_X86 1
#endif

uint32_t ae_cpu_detect(void)
{
    uint32_t m = atomic_load_explicit(&g_cpu_mask, memory_order_acquire);
    if (m != 0) return m;

    uint32_t mask = AE_CPU_SCALAR;
#ifdef AE_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("sse2"))    mask |= AE_CPU_SSE2;
    if (__builtin_cpu_supports("sse4.2"))  mask |= AE_CPU_SSE4_2;
    if (__builtin_cpu_supports("avx2"))    mask |= AE_CPU_AVX2;
    if (__builtin_cpu_supports("fma"))     mask |= AE_CPU_FMA;
    if (__builtin_cpu_supports("avx512f")) mask |= AE_CPU_AVX512F;
#endif
    atomic_store_explicit(&g_cpu_mask, mask, memory_order_release);
    return mask;
}

uint32_t ae_cpu_features(void) { return ae_cpu_detect(); }

int ae_cpu_set_policy(ae_dispatch_policy policy)
{
    switch (policy) {
    case AE_DISPATCH_SCALAR: atomic_store(&g_force, -1); return 0;
    case AE_DISPATCH_AUTO:
    case AE_DISPATCH_BEST:   atomic_store(&g_force,  1); return 0;
    }
    return -1;
}

ae_dispatch_policy ae_cpu_get_policy(void)
{
    int f = atomic_load(&g_force);
    if (f < 0) return AE_DISPATCH_SCALAR;
    return AE_DISPATCH_AUTO;
}

const char *ae_cpu_target_name(void)
{
    int f = atomic_load(&g_force);
    if (f < 0) return "scalar";
    uint32_t m = ae_cpu_detect();
    if ((f >= 0) && (m & AE_CPU_AVX512F)) return "avx512f";
    if ((f >= 0) && (m & AE_CPU_AVX2))    return "avx2";
    if ((f >= 0) && (m & AE_CPU_SSE2))    return "sse2";
    return "scalar";
}

/* --------------------------------------------------------------- alignment */
void *ae_aligned_alloc(size_t bytes, size_t alignment)
{
    if (bytes == 0) return NULL;
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    /* round up to a power of two */
    size_t a = 1;
    while (a < alignment) a <<= 1;
    void *p = NULL;
#if defined(_WIN32)
    p = _aligned_malloc(bytes, a);
#else
    if (posix_memalign(&p, a, bytes) != 0) return NULL;
#endif
    return p;
}

void ae_aligned_free(void *p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

ae_buffer ae_buffer_alloc(size_t n)
{
    ae_buffer b = { NULL, 0 };
    if (n == 0) return b;
    size_t bytes = 0;
    if (__builtin_mul_overflow(n, sizeof(double), &bytes)) return b;
    b.data = (double *)ae_aligned_alloc(bytes, AE_ALIGN);
    if (b.data) b.len = n;
    return b;
}

void ae_buffer_free(ae_buffer *b)
{
    if (!b) return;
    ae_aligned_free(b->data);
    b->data = NULL;
    b->len  = 0;
}

ae_status ae_buffer_copy(const ae_buffer *src, ae_buffer *dst)
{
    if (!src || !dst) return AE_ERR_NULL;
    if (!src->data || src->len == 0) return AE_ERR_SIZE_ZERO;
    if (!dst->data || dst->len != src->len) return AE_ERR_DIM_MISMATCH;
    memcpy(dst->data, src->data, src->len * sizeof(double));
    return AE_OK;
}

ae_matrix ae_matrix_alloc(size_t rows, size_t cols)
{
    ae_matrix m = { NULL, 0, 0, 0 };
    if (rows == 0 || cols == 0) return m;
    /* pad row stride to a 64-byte boundary so every row is aligned */
    size_t lane = AE_ALIGN / sizeof(double);
    size_t stride = ((cols + lane - 1) / lane) * lane;
    size_t total = 0, bytes = 0;
    if (__builtin_mul_overflow(rows, stride, &total)) return m;
    if (__builtin_mul_overflow(total, sizeof(double), &bytes)) return m;
    m.data = (double *)ae_aligned_alloc(bytes, AE_ALIGN);
    if (!m.data) return m;
    memset(m.data, 0, bytes);
    m.rows = rows; m.cols = cols; m.stride = stride;
    return m;
}

void ae_matrix_free(ae_matrix *m)
{
    if (!m) return;
    ae_aligned_free(m->data);
    m->data = NULL; m->rows = m->cols = m->stride = 0;
}

double *ae_matrix_row(ae_matrix *m, size_t r)
{
    if (!m || !m->data || r >= m->rows) return NULL;
    return m->data + r * m->stride;
}

const double *ae_matrix_row_c(const ae_matrix *m, size_t r)
{
    if (!m || !m->data || r >= m->rows) return NULL;
    return m->data + r * m->stride;
}

/* ------------------------------------------------------- vector: scalar impl */
static void dot_scalar(const double *restrict a, const double *restrict b, size_t n, double *out)
{
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) s0 += a[i] * b[i];
    *out = (s0 + s1) + (s2 + s3);
}

/* --------------------------------------------------------- vector: AVX2+FMA */
#if defined(AE_X86) && defined(__GNUC__) && !defined(AE_NO_SIMD)
__attribute__((target("avx2,fma")))
static void dot_avx2(const double *restrict a, const double *restrict b, size_t n, double *out)
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i),     acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4), acc1);
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    double tmp[4];
    _mm256_storeu_pd(tmp, acc0);
    double s = ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3]));
    for (; i < n; ++i) s += a[i] * b[i];
    *out = s;
}

__attribute__((target("avx512f")))
static void dot_avx512(const double *restrict a, const double *restrict b, size_t n, double *out)
{
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i),      _mm512_loadu_pd(b + i),      acc0);
        acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 8),  _mm512_loadu_pd(b + i + 8),  acc1);
    }
    acc0 = _mm512_add_pd(acc0, acc1);
    double s = _mm512_reduce_add_pd(acc0);
    for (; i < n; ++i) s += a[i] * b[i];
    *out = s;
}
#endif

/* ---------------------------------------------------------------- dispatch */
typedef void (*dot_fn)(const double *, const double *, size_t, double *);

static dot_fn resolve_dot(void)
{
#if defined(AE_X86) && defined(__GNUC__) && !defined(AE_NO_SIMD)
    int f = atomic_load(&g_force);
    if (f < 0) return dot_scalar;
    uint32_t m = ae_cpu_detect();
    if (m & AE_CPU_AVX512F) return dot_avx512;
    if ((m & AE_CPU_AVX2) && (m & AE_CPU_FMA)) return dot_avx2;
#endif
    return dot_scalar;
}

/* Cached dispatch pointer; safe because resolution is idempotent. */
static void dot_dispatch(const double *a, const double *b, size_t n, double *out)
{
    static _Atomic(dot_fn) fn = NULL;
    dot_fn f = atomic_load_explicit(&fn, memory_order_acquire);
    if (!f) {
        f = resolve_dot();
        atomic_store_explicit(&fn, f, memory_order_release);
    }
    f(a, b, n, out);
}

/* Exposed for benchmarking: run a specific kernel regardless of dispatch. */
void ae_dot_scalar(const double *a, const double *b, size_t n, double *out) { dot_scalar(a, b, n, out); }
#if defined(AE_X86) && defined(__GNUC__) && !defined(AE_NO_SIMD)
void ae_dot_avx2(const double *a, const double *b, size_t n, double *out)   { dot_avx2(a, b, n, out); }
void ae_dot_avx512(const double *a, const double *b, size_t n, double *out) { dot_avx512(a, b, n, out); }
#endif

/* ------------------------------------------------------------- public API */
ae_status ae_vec_dot(const double *a, const double *b, size_t n, double *out)
{
    if (!a || !b || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    if (a == b) { /* legal, dot(x,x) */ }
    dot_dispatch(a, b, n, out);
    return isfinite(*out) ? AE_OK : AE_ERR_NAN;
}

ae_status ae_vec_norm(const double *x, size_t n, double *norm_out)
{
    if (!x || !norm_out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double s;
    dot_dispatch(x, x, n, &s);
    if (s < 0.0) return AE_ERR_NAN;
    *norm_out = sqrt(s);
    return AE_OK;
}

ae_status ae_vec_normalize(const double *x, size_t n, double *out)
{
    if (!x || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    if (x == out) return AE_ERR_UNSUPPORTED; /* alias forbidden by contract */
    double nrm;
    ae_status st = ae_vec_norm(x, n, &nrm);
    if (st != AE_OK) return st;
    if (nrm <= 0.0) return AE_ERR_NOT_NORMALIZED;
    const double inv = 1.0 / nrm;
    for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv;
    return AE_OK;
}

ae_status ae_vec_axpy(double alpha, const double *x, double *y, size_t n)
{
    if (!x || !y) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
    return AE_OK;
}

ae_status ae_vec_scale(double alpha, double *x, size_t n)
{
    if (!x) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    for (size_t i = 0; i < n; ++i) x[i] *= alpha;
    return AE_OK;
}

ae_status ae_cosine_raw(const double *a, const double *b, size_t n, double *out)
{
    if (!a || !b || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double ab, aa, bb;
    dot_dispatch(a, b, n, &ab);
    dot_dispatch(a, a, n, &aa);
    dot_dispatch(b, b, n, &bb);
    if (aa <= 0.0 || bb <= 0.0) return AE_ERR_NOT_NORMALIZED;
    double c = ab / (sqrt(aa) * sqrt(bb));
    if (c >  1.0) c =  1.0;   /* guard against rounding above 1 */
    if (c < -1.0) c = -1.0;
    *out = c;
    return AE_OK;
}

ae_status ae_cosine_unit(const double *a, const double *b, size_t n, double *out)
{
    if (!a || !b || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double c;
    dot_dispatch(a, b, n, &c);
    if (c >  1.0) c =  1.0;
    if (c < -1.0) c = -1.0;
    *out = c;
    return AE_OK;
}

ae_status ae_l2sq(const double *a, const double *b, size_t n, double *out)
{
    if (!a || !b || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) { double d = a[i] - b[i]; s += d * d; }
    *out = s;
    return AE_OK;
}

ae_status ae_batch_cosine_unit(const double *q, size_t m,
                               const double *x, size_t n, size_t d, double *out)
{
    if (!q || !x || !out) return AE_ERR_NULL;
    if (m == 0 || n == 0 || d == 0) return AE_ERR_SIZE_ZERO;
    for (size_t i = 0; i < m; ++i) {
        const double *qi = q + i * d;
        double *oi = out + i * n;
        for (size_t j = 0; j < n; ++j) {
            dot_dispatch(qi, x + j * d, d, &oi[j]);
            if (oi[j] > 1.0) oi[j] = 1.0;
        }
    }
    return AE_OK;
}

ae_status ae_batch_cosine_raw(const double *q, size_t m,
                              const double *x, size_t n, size_t d, double *out)
{
    if (!q || !x || !out) return AE_ERR_NULL;
    if (m == 0 || n == 0 || d == 0) return AE_ERR_SIZE_ZERO;
    /* normalise once into a scratch buffer; avoids m*n*sqrt costs */
    ae_buffer qn = ae_buffer_alloc(m * d);
    ae_buffer xn = ae_buffer_alloc(n * d);
    if (!qn.data || !xn.data) { ae_buffer_free(&qn); ae_buffer_free(&xn); return AE_ERR_ALLOC; }
    ae_status st = AE_OK;
    for (size_t i = 0; i < m && st == AE_OK; ++i)
        st = ae_vec_normalize(q + i * d, d, qn.data + i * d);
    for (size_t j = 0; j < n && st == AE_OK; ++j)
        st = ae_vec_normalize(x + j * d, d, xn.data + j * d);
    if (st == AE_OK) st = ae_batch_cosine_unit(qn.data, m, xn.data, n, d, out);
    ae_buffer_free(&qn);
    ae_buffer_free(&xn);
    return st;
}
