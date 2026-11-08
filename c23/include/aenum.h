/* aenum.h — ATTRACTION ENGINE numerical library (C23)
 *
 * Design contract
 * ---------------
 *  OWNERSHIP      Every function that allocates returns ownership to the caller
 *                 through a handle (ae_buffer / ae_matrix) or an out-pointer that
 *                 must be released with the matching *_free function. No function
 *                 stores a borrowed pointer beyond its own call.
 *  ERRORS         Every fallible function returns ae_status. AE_OK == 0, every
 *                 error is a distinct negative value. ae_status_str() maps codes
 *                 to stable strings.
 *  BOUNDS         All lengths are size_t and are validated before any pointer
 *                 arithmetic. Passing a NULL pointer where a length is non-zero is
 *                 an error, never UB. Overlapping regions are rejected where the
 *                 semantics forbid aliasing.
 *  THREADING      All functions are re-entrant and thread-safe: there is no global
 *                 mutable state. The single piece of global state is the CPU feature
 *                 bitmask, which is written once by ae_cpu_detect() (idempotent) and
 *                 read-only afterwards.
 *  NUMERICS       Accumulation uses double. SIMD paths use FMA where available.
 *                 SIMD and scalar paths are required by tests to agree to <= 1e-12
 *                 in the similarity measure.
 *
 * NOTE ON C23: GCC 14.2 accepts -std=c23 (verified) but does not implement the
 * C23 <stdembed.h> / #embed facility. No portable C23 embedding is therefore used;
 * the fallback path is documented in docs/environment.md.
 */
#ifndef AENUM_H
#define AENUM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(AENUM_BUILD_SHARED)
#  define AE_API __declspec(dllexport)
#elif defined(__GNUC__)
#  define AE_API __attribute__((visibility("default")))
#else
#  define AE_API
#endif

/* ------------------------------------------------------------------ status */
typedef enum ae_status : int32_t {
    AE_OK                 =   0,
    AE_ERR_NULL           =  -1,  /* required pointer was NULL                   */
    AE_ERR_SIZE_ZERO      =  -2,  /* length argument was zero                    */
    AE_ERR_DIM_MISMATCH   =  -3,  /* incompatible dimensions                     */
    AE_ERR_INDEX_RANGE    =  -4,  /* index outside [0, n)                        */
    AE_ERR_ALLOC          =  -5,  /* allocation failure                          */
    AE_ERR_NOT_NORMALIZED =  -6,  /* pre-condition ||v||_2 ~= 0 violated         */
    AE_ERR_NAN            =  -7,  /* non-finite value encountered                */
    AE_ERR_SINGULAR       =  -8,  /* matrix not invertible                       */
    AE_ERR_UNSUPPORTED    =  -9,  /* requested capability not available          */
    AE_ERR_OVERFLOW       = -10,  /* result would overflow the destination type  */
    AE_ERR_ALIGNMENT      = -11   /* buffer does not satisfy requested alignment */
} ae_status;

AE_API const char *ae_status_str(ae_status s);

/* ------------------------------------------------------------------- CPU */
enum ae_cpu_feature : uint32_t {
    AE_CPU_SCALAR  = 0u,
    AE_CPU_SSE2    = 1u << 0,
    AE_CPU_SSE4_2  = 1u << 1,
    AE_CPU_AVX2    = 1u << 2,
    AE_CPU_FMA     = 1u << 3,
    AE_CPU_AVX512F = 1u << 4
};

AE_API uint32_t ae_cpu_detect(void);      /* idempotent, caches result           */
AE_API uint32_t ae_cpu_features(void);    /* returns cached mask (detect if 0)   */
AE_API const char *ae_cpu_target_name(void); /* "avx512f" | "avx2" | "sse2" | "scalar" */

/* Dispatch policy. AE_CPU_SCALAR is 0 so it cannot double as "auto"; the
 * policy is therefore an explicit enum to remove the ambiguity. */
typedef enum ae_dispatch_policy : int32_t {
    AE_DISPATCH_SCALAR = -1,  /* force the portable kernel (testing)             */
    AE_DISPATCH_AUTO   =  0,  /* pick the widest kernel the CPU supports         */
    AE_DISPATCH_BEST   =  1   /* same as AUTO; explicit intent, self-documenting */
} ae_dispatch_policy;

AE_API int ae_cpu_set_policy(ae_dispatch_policy policy);
AE_API ae_dispatch_policy ae_cpu_get_policy(void);

/* ---------------------------------------------------------- owned buffer */
/* Alignment is fixed at 64 bytes (one AVX-512 register / cache line). */
#define AE_ALIGN 64

typedef struct ae_buffer {
    double *data;   /* owned, 64-byte aligned, length `len` (may be NULL iff len==0) */
    size_t  len;
} ae_buffer;

AE_API void  *ae_aligned_alloc(size_t bytes, size_t alignment);
AE_API void   ae_aligned_free(void *p);
AE_API ae_buffer ae_buffer_alloc(size_t n);
AE_API void   ae_buffer_free(ae_buffer *b);     /* idempotent; NULLs the handle   */
AE_API ae_status ae_buffer_copy(const ae_buffer *src, ae_buffer *dst);

/* Row-major dense matrix, owned storage, 64-byte aligned. */
typedef struct ae_matrix {
    double *data;   /* owned; rows*cols doubles */
    size_t  rows;
    size_t  cols;
    size_t  stride; /* in doubles, >= cols (kept for alignment padding) */
} ae_matrix;

AE_API ae_matrix ae_matrix_alloc(size_t rows, size_t cols);
AE_API void      ae_matrix_free(ae_matrix *m);
AE_API double   *ae_matrix_row(ae_matrix *m, size_t r);
AE_API const double *ae_matrix_row_c(const ae_matrix *m, size_t r);

/* -------------------------------------------------------- vector kernels */
/* y = x / ||x||_2 .  out-of-place; ae_ is never allowed to alias x. */
AE_API ae_status ae_vec_normalize(const double *x, size_t n, double *out);
AE_API ae_status ae_vec_norm(const double *x, size_t n, double *norm_out);
AE_API ae_status ae_vec_dot(const double *a, const double *b, size_t n, double *out);
AE_API ae_status ae_vec_axpy(double alpha, const double *x, double *y, size_t n);
AE_API ae_status ae_vec_scale(double alpha, double *x, size_t n);

/* cosine similarity of raw vectors (normalises internally, no allocation) */
AE_API ae_status ae_cosine_raw(const double *a, const double *b, size_t n, double *out);
/* cosine similarity of pre-normalised vectors (invariant: unit norm)      */
AE_API ae_status ae_cosine_unit(const double *a, const double *b, size_t n, double *out);
/* Euclidean distance and squared distance */
AE_API ae_status ae_l2sq(const double *a, const double *b, size_t n, double *out);

/* Batched similarity: Q (m x d) against X (n x d) row-major, no allocation.
 * out is (m x n) row-major, out[i*n + j] = cosine(Q_i, X_j).
 * Both operands must be unit-normalised; use ae_batch_cosine_raw otherwise. */
AE_API ae_status ae_batch_cosine_unit(const double *q, size_t m,
                                      const double *x, size_t n, size_t d,
                                      double *out);
AE_API ae_status ae_batch_cosine_raw(const double *q, size_t m,
                                     const double *x, size_t n, size_t d,
                                     double *out);

/* ------------------------------------------------------------- statistics */
typedef struct ae_stats {
    double mean;
    double variance;      /* population variance (1/n)        */
    double sample_variance; /* (1/(n-1)), AE_ERR_SIZE_ZERO-safe */
    double stddev;
    double skewness;      /* Fisher-Pearson, bias-corrected absent */
    double kurtosis;      /* excess kurtosis                      */
    double min;
    double max;
    double median;
    double iqr;           /* Q3 - Q1, linear interpolation        */
    size_t n;
} ae_stats;

AE_API ae_status ae_stats_compute(const double *x, size_t n, ae_stats *out);
AE_API ae_status ae_mean(const double *x, size_t n, double *out);
AE_API ae_status ae_variance(const double *x, size_t n, double *out);
AE_API ae_status ae_percentile(const double *x, size_t n, double p, double *out);

typedef struct ae_histogram {
    size_t  bins;
    double  lo;
    double  hi;
    double *counts;   /* owned, bins entries */
    double *edges;    /* owned, bins+1 entries */
} ae_histogram;

AE_API ae_status ae_histogram_alloc(size_t bins, double lo, double hi, ae_histogram *out);
AE_API void      ae_histogram_free(ae_histogram *h);
AE_API ae_status ae_histogram_compute(const double *x, size_t n, ae_histogram *h);
AE_API ae_status ae_histogram_entropy(const ae_histogram *h, double *out); /* bits */

/* Shannon entropy of a probability vector (nats). */
AE_API ae_status ae_entropy(const double *p, size_t n, double *out);

/* --------------------------------------------------------- image features */
/* Image analytics operating on an 8-bit single channel or interleaved RGB
 * buffer. These are the primitives behind Module A's image report; the
 * Python layer adds semantics, the C23 layer guarantees the numerics. */
typedef struct ae_image_stats {
    double mean;          /* [0,1] average luma                                   */
    double stddev;
    double contrast_rms;  /* RMS contrast                                         */
    double exposure_clip_low;  /* fraction of samples <= 0.02                     */
    double exposure_clip_high; /* fraction of samples >= 0.98                     */
    double entropy;       /* Shannon entropy of a 256-bin luma histogram (bits)  */
    double shadow_fraction;
    double highlight_fraction;
    double edge_density;  /* mean |Sobel| normalised by 255                       */
    double background_complexity; /* high-frequency energy fraction               */
    size_t pixels;
} ae_image_stats;

AE_API ae_status ae_image_analyze_gray(const uint8_t *gray, size_t w, size_t h,
                                       ae_image_stats *out);
AE_API ae_status ae_image_analyze_rgb(const uint8_t *rgb, size_t w, size_t h,
                                      ae_image_stats *out);
/* Normalised to zero mean, unit L2 norm; used for deterministic fallback
 * embeddings when no pretrained vision model is available.               */
AE_API ae_status ae_image_descriptor_rgb(const uint8_t *rgb, size_t w, size_t h,
                                         size_t grid, ae_buffer *out);

/* --------------------------------------------------------------- linear algebra */
AE_API ae_status ae_matmul(const ae_matrix *a, const ae_matrix *b, ae_matrix *out);
AE_API ae_status ae_matvec(const ae_matrix *a, const double *x, double *out);
AE_API ae_status ae_mat_transpose(const ae_matrix *a, ae_matrix *out);
AE_API ae_status ae_covariance(const ae_matrix *x, ae_matrix *out,
                               bool sample_normalisation);

/* --------------------------------------------------------------- utilities */
AE_API ae_status ae_softmax(const double *x, size_t n, double *out);
AE_API ae_status ae_logsumexp(const double *x, size_t n, double *out);
AE_API ae_status ae_topk(const double *x, size_t n, size_t k,
                         size_t *idx_out, double *val_out);
AE_API ae_status ae_argmax(const double *x, size_t n, size_t *out);
/* Stable two-pass Kahan-compensated summation; used as the reference in tests. */
AE_API ae_status ae_sum_kahan(const double *x, size_t n, double *out);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AENUM_H */
