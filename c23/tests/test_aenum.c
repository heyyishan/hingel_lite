/* test_aenum.c — dependency-free test harness for the C23 numerical library.
 *
 * Coverage: memory safety (release builds run under ASan+UBSan via
 * AENUM_SANITIZE), bounds validation, numerical accuracy against closed-form
 * references, SIMD/scalar equivalence, and edge cases.
 */
#include "aenum.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;
static const char *g_case = "";

#define CHECK(cond) do { \
    if (!(cond)) { printf("  FAIL %s:%d [%s] %s\n", __FILE__, __LINE__, g_case, #cond); g_fail++; } \
    else { g_pass++; } \
} while (0)

#define CHECK_ST(expr, expected) do { \
    ae_status _s = (expr); \
    if (_s != (expected)) { printf("  FAIL %s:%d [%s] %s -> %s (want %s)\n", __FILE__, __LINE__, \
        g_case, #expr, ae_status_str(_s), ae_status_str(expected)); g_fail++; } \
    else { g_pass++; } \
} while (0)

#define NEAR(a, b, tol) (fabs((a) - (b)) <= (tol))

static void begin(const char *name) { g_case = name; printf("[case] %s\n", name); }

/* deterministic xorshift so test data is reproducible */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double urand(void)
{
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) / (double)(1ull << 53);
}

/* ------------------------------------------------------------- unit tests */
static void test_status_strings(void)
{
    begin("status_strings");
    CHECK(strcmp(ae_status_str(AE_OK), "ok") == 0);
    CHECK(strlen(ae_status_str(AE_ERR_ALLOC)) > 0);
    CHECK(strcmp(ae_status_str((ae_status)999), "unknown status") == 0);
}

static void test_cpu_detection(void)
{
    begin("cpu_detection");
    uint32_t m = ae_cpu_detect();
    CHECK(m != 0);
    CHECK(ae_cpu_features() == m);             /* idempotent + cached */
    const char *name = ae_cpu_target_name();
    printf("  target=%s mask=0x%x\n", name, m);
    CHECK(strlen(name) > 0);
#if defined(__x86_64__)
    CHECK((m & AE_CPU_SSE2) != 0u);            /* x86-64 baseline */
#endif
}

static void test_alignment(void)
{
    begin("alignment");
    for (size_t n = 1; n < 200; n += 7) {
        ae_buffer b = ae_buffer_alloc(n);
        CHECK(b.data != NULL);
        CHECK(b.len == n);
        CHECK(((uintptr_t)b.data % AE_ALIGN) == 0u);
        ae_buffer_free(&b);
        CHECK(b.data == NULL && b.len == 0);
        ae_buffer_free(&b);                    /* idempotent free */
    }
    ae_buffer z = ae_buffer_alloc(0);
    CHECK(z.data == NULL && z.len == 0);

    ae_matrix m = ae_matrix_alloc(5, 7);
    CHECK(m.data != NULL && m.rows == 5 && m.cols == 7);
    CHECK(m.stride >= 7);
    for (size_t r = 0; r < m.rows; ++r)
        CHECK(((uintptr_t)ae_matrix_row(&m, r) % AE_ALIGN) == 0u);
    ae_matrix_free(&m);
    CHECK(m.data == NULL);
}

static void test_null_and_bounds(void)
{
    begin("bounds_validation");
    double out = 0.0, v[4] = { 1, 2, 3, 4 };
    CHECK_ST(ae_vec_dot(NULL, v, 4, &out), AE_ERR_NULL);
    CHECK_ST(ae_vec_dot(v, v, 4, NULL), AE_ERR_NULL);
    CHECK_ST(ae_vec_dot(v, v, 0, &out), AE_ERR_SIZE_ZERO);
    CHECK_ST(ae_vec_normalize(v, 0, v), AE_ERR_SIZE_ZERO);
    CHECK_ST(ae_vec_normalize(v, 4, v), AE_ERR_UNSUPPORTED);  /* alias rejected */
    CHECK_ST(ae_mean(NULL, 4, &out), AE_ERR_NULL);
    CHECK_ST(ae_mean(v, 0, &out), AE_ERR_SIZE_ZERO);

    size_t idx[4]; double val[4];
    CHECK_ST(ae_topk(v, 4, 0, idx, val), AE_ERR_SIZE_ZERO);
    CHECK_ST(ae_topk(v, 4, 5, idx, val), AE_ERR_INDEX_RANGE);

    ae_histogram h;
    CHECK_ST(ae_histogram_alloc(0, 0, 1, &h), AE_ERR_SIZE_ZERO);
    CHECK_ST(ae_histogram_alloc(8, 1, 1, &h), AE_ERR_DIM_MISMATCH);

    ae_matrix a = ae_matrix_alloc(2, 3), b = ae_matrix_alloc(2, 3), c = ae_matrix_alloc(2, 3);
    CHECK_ST(ae_matmul(&a, &b, &c), AE_ERR_DIM_MISMATCH);   /* inner dims differ */
    ae_matrix_free(&a); ae_matrix_free(&b); ae_matrix_free(&c);
}

static void test_normalize_cosine(void)
{
    begin("normalize_cosine");
    double x[4] = { 3.0, 4.0, 0.0, 0.0 };
    double z[4];
    CHECK_ST(ae_vec_normalize(x, 4, z), AE_OK);
    CHECK(NEAR(z[0], 0.6, 1e-15));
    CHECK(NEAR(z[1], 0.8, 1e-15));
    double nrm;
    ae_vec_norm(z, 4, &nrm);
    CHECK(NEAR(nrm, 1.0, 1e-15));

    double c;
    CHECK_ST(ae_cosine_unit(z, z, 4, &c), AE_OK);
    CHECK(NEAR(c, 1.0, 1e-15));

    double y[4] = { -4.0, 3.0, 0.0, 0.0 };
    double zy[4];
    ae_vec_normalize(y, 4, zy);
    ae_cosine_unit(z, zy, 4, &c);
    CHECK(NEAR(c, 0.0, 1e-15));                 /* orthogonal */

    /* zero vector is rejected, not NaN-propagated */
    double zero[3] = { 0, 0, 0 }, zo[3];
    CHECK_ST(ae_vec_normalize(zero, 3, zo), AE_ERR_NOT_NORMALIZED);
    CHECK_ST(ae_cosine_raw(zero, zero, 3, &c), AE_ERR_NOT_NORMALIZED);
}

static void test_simd_scalar_equivalence(void)
{
    begin("simd_scalar_equivalence");
    /* Compare every available kernel against a double-double reference. */
    extern void ae_dot_scalar(const double *, const double *, size_t, double *);
#if defined(__x86_64__)
    extern void ae_dot_avx2(const double *, const double *, size_t, double *);
    extern void ae_dot_avx512(const double *, const double *, size_t, double *);
#endif
    const size_t dims[] = { 1, 3, 7, 8, 15, 16, 17, 64, 127, 128, 384, 512, 1000, 4096 };
    for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); ++di) {
        size_t n = dims[di];
        double *a = (double *)malloc(n * sizeof(double));
        double *b = (double *)malloc(n * sizeof(double));
        for (size_t i = 0; i < n; ++i) { a[i] = urand() * 2.0 - 1.0; b[i] = urand() * 2.0 - 1.0; }
        /* long-double reference */
        long double ref = 0.0L;
        for (size_t i = 0; i < n; ++i) ref += (long double)a[i] * (long double)b[i];
        double refd = (double)ref;

        double s, s2, s3;
        ae_dot_scalar(a, b, n, &s);
        CHECK(NEAR(s, refd, 1e-12 * (fabs(refd) + 1.0)));
#if defined(__x86_64__)
        if (ae_cpu_features() & AE_CPU_AVX2) {
            ae_dot_avx2(a, b, n, &s2);
            CHECK(NEAR(s2, refd, 1e-12 * (fabs(refd) + 1.0)));
            CHECK(NEAR(s2, s, 1e-12 * (fabs(refd) + 1.0)));
        }
        if (ae_cpu_features() & AE_CPU_AVX512F) {
            ae_dot_avx512(a, b, n, &s3);
            CHECK(NEAR(s3, refd, 1e-12 * (fabs(refd) + 1.0)));
            CHECK(NEAR(s3, s, 1e-12 * (fabs(refd) + 1.0)));
        }
#endif
        free(a); free(b);
    }
}

static void test_batch_cosine(void)
{
    begin("batch_cosine");
    const size_t m = 5, n = 9, d = 32;
    double *q = (double *)malloc(m * d * sizeof(double));
    double *x = (double *)malloc(n * d * sizeof(double));
    double *out = (double *)malloc(m * n * sizeof(double));
    double *qn = (double *)malloc(m * d * sizeof(double));
    double *xn = (double *)malloc(n * d * sizeof(double));
    for (size_t i = 0; i < m * d; ++i) q[i] = urand() - 0.5;
    for (size_t i = 0; i < n * d; ++i) x[i] = urand() - 0.5;
    for (size_t i = 0; i < m; ++i) ae_vec_normalize(q + i * d, d, qn + i * d);
    for (size_t j = 0; j < n; ++j) ae_vec_normalize(x + j * d, d, xn + j * d);

    CHECK_ST(ae_batch_cosine_unit(qn, m, xn, n, d, out), AE_OK);

    /* reference via cosine_raw, which normalises independently */
    int ok = 1;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            double ref;
            ae_cosine_raw(q + i * d, x + j * d, d, &ref);
            if (!NEAR(ref, out[i * n + j], 1e-12)) ok = 0;
        }
    CHECK(ok);

    double *out2 = (double *)malloc(m * n * sizeof(double));
    CHECK_ST(ae_batch_cosine_raw(q, m, x, n, d, out2), AE_OK);
    ok = 1;
    for (size_t i = 0; i < m * n; ++i) if (!NEAR(out[i], out2[i], 1e-10)) ok = 0;
    CHECK(ok);

    free(q); free(x); free(out); free(out2); free(qn); free(xn);
}

static void test_stats_accuracy(void)
{
    begin("stats_accuracy");
    /* closed form for {1..10}: mean 5.5, population var 8.25, sample var 9.1666.. */
    double v[10]; for (int i = 0; i < 10; ++i) v[i] = (double)(i + 1);
    ae_stats s;
    CHECK_ST(ae_stats_compute(v, 10, &s), AE_OK);
    CHECK(NEAR(s.mean, 5.5, 1e-14));
    CHECK(NEAR(s.variance, 8.25, 1e-13));
    CHECK(NEAR(s.sample_variance, 9.166666666666666, 1e-13));
    CHECK(NEAR(s.stddev, sqrt(8.25), 1e-13));
    CHECK(NEAR(s.min, 1.0, 0.0));
    CHECK(NEAR(s.max, 10.0, 0.0));
    CHECK(NEAR(s.median, 5.5, 1e-13));
    CHECK(NEAR(s.iqr, 4.5, 1e-13));  /* type-7: q1=3.25, q3=7.75 */

    /* symmetry: skew ~ 0 for a symmetric sample */
    double sym[9] = { -4, -3, -2, -1, 0, 1, 2, 3, 4 };
    ae_stats ss;
    ae_stats_compute(sym, 9, &ss);
    CHECK(NEAR(ss.skewness, 0.0, 1e-12));

    /* Kahan summation beats naive on a cancelling sequence */
    double big[10002];
    big[0] = 1e16;
    for (int i = 1; i <= 10000; ++i) big[i] = 1.0;
    big[10001] = -1e16;
    double ksum;
    ae_sum_kahan(big, 10002, &ksum);
    CHECK(NEAR(ksum, 10000.0, 1e-6));

    double p;
    ae_percentile(v, 10, 0.0, &p);   CHECK(NEAR(p, 1.0, 1e-14));
    ae_percentile(v, 10, 100.0, &p); CHECK(NEAR(p, 10.0, 1e-14));
    ae_percentile(v, 10, 50.0, &p);  CHECK(NEAR(p, 5.5, 1e-14));
}

static void test_histogram_entropy(void)
{
    begin("histogram_entropy");
    size_t n = 10000;
    double *v = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) v[i] = urand();  /* uniform on [0,1) */
    ae_histogram h;
    CHECK_ST(ae_histogram_alloc(16, 0.0, 1.0, &h), AE_OK);
    CHECK_ST(ae_histogram_compute(v, n, &h), AE_OK);
    double total = 0;
    for (size_t i = 0; i < h.bins; ++i) total += h.counts[i];
    CHECK(NEAR(total, (double)n, 0.0));
    for (size_t i = 0; i < h.bins; ++i)
        CHECK(h.counts[i] > (double)n / 16.0 * 0.85 &&
              h.counts[i] < (double)n / 16.0 * 1.15);
    double H;
    CHECK_ST(ae_histogram_entropy(&h, &H), AE_OK);
    CHECK(H > 3.8 && H <= 4.0);   /* 16 equiprobable bins -> 4 bits */
    ae_histogram_free(&h);

    /* entropy of a point mass is 0 */
    double pm[4] = { 1.0, 0, 0, 0 }, e;
    ae_entropy(pm, 4, &e);
    CHECK(NEAR(e, 0.0, 0.0));
    double uniform[4] = { 0.25, 0.25, 0.25, 0.25 };
    ae_entropy(uniform, 4, &e);
    CHECK(NEAR(e, log(4.0), 1e-15));
    free(v);
}

static void test_topk(void)
{
    begin("topk");
    const size_t n = 1000;
    double *v = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) v[i] = urand();
    for (size_t k = 1; k <= 20; k += 3) {
        size_t *idx = (size_t *)malloc(k * sizeof(size_t));
        double *val = (double *)malloc(k * sizeof(double));
        CHECK_ST(ae_topk(v, n, k, idx, val), AE_OK);
        int sorted = 1, correct = 1;
        for (size_t i = 1; i < k; ++i) if (val[i] > val[i - 1]) sorted = 0;
        for (size_t i = 0; i < k; ++i) if (!NEAR(val[i], v[idx[i]], 0.0)) correct = 0;
        CHECK(sorted); CHECK(correct);
        /* ground truth: count elements strictly greater than the k-th value */
        size_t greater = 0;
        for (size_t i = 0; i < n; ++i) if (v[i] > val[k - 1]) greater++;
        CHECK(greater <= k - 1);
        free(idx); free(val);
    }
    free(v);
}

static void test_softmax_logsumexp(void)
{
    begin("softmax_logsumexp");
    double x[5] = { 1000.0, 1001.0, 1002.0, -1000.0, 0.0 };
    double out[5];
    CHECK_ST(ae_softmax(x, 5, out), AE_OK);
    double s = 0.0;
    for (int i = 0; i < 5; ++i) { CHECK(out[i] >= 0.0); s += out[i]; }
    CHECK(NEAR(s, 1.0, 1e-14));   /* stable under huge offsets */
    double lse;
    ae_logsumexp(x, 5, &lse);
    CHECK(lse > 1002.0 && lse < 1002.7);
    double zero[3] = { 0, 0, 0 };
    ae_softmax(zero, 3, out);
    for (int i = 0; i < 3; ++i) CHECK(NEAR(out[i], 1.0 / 3.0, 1e-15));
}

static void test_matmul_transpose(void)
{
    begin("matmul_transpose");
    ae_matrix a = ae_matrix_alloc(3, 4), b = ae_matrix_alloc(4, 2), c = ae_matrix_alloc(3, 2);
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 4; ++j) ae_matrix_row(&a, i)[j] = (double)(i * 4 + j + 1);
    for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 2; ++j) ae_matrix_row(&b, i)[j] = (double)(i + j + 1);
    CHECK_ST(ae_matmul(&a, &b, &c), AE_OK);
    /* row0 of A = [1 2 3 4]; col0 of B = [1 2 3 4] -> (1+4+9+16)=30 */
    CHECK(NEAR(ae_matrix_row(&c, 0)[0], 30.0, 1e-12));
    /* col1 of B = [2 3 4 5] -> 2+6+12+20 = 40 */
    CHECK(NEAR(ae_matrix_row(&c, 0)[1], 40.0, 1e-12));

    ae_matrix t = ae_matrix_alloc(4, 3);
    CHECK_ST(ae_mat_transpose(&a, &t), AE_OK);
    CHECK(NEAR(ae_matrix_row(&t, 2)[1], 7.0, 0.0));   /* a[1][2] */
    ae_matrix_free(&a); ae_matrix_free(&b); ae_matrix_free(&c); ae_matrix_free(&t);
}

static void test_covariance(void)
{
    begin("covariance");
    ae_matrix x = ae_matrix_alloc(4, 2);
    double v[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    for (size_t i = 0; i < 4; ++i) { ae_matrix_row(&x, i)[0] = v[2 * i]; ae_matrix_row(&x, i)[1] = v[2 * i + 1]; }
    ae_matrix cov = ae_matrix_alloc(2, 2);
    CHECK_ST(ae_covariance(&x, &cov, true), AE_OK);
    /* x and y are perfectly correlated with var = 20/3 (sample) */
    CHECK(NEAR(ae_matrix_row(&cov, 0)[0], 20.0 / 3.0, 1e-12));
    CHECK(NEAR(ae_matrix_row(&cov, 0)[1], 20.0 / 3.0, 1e-12));
    ae_matrix_free(&x); ae_matrix_free(&cov);
}

static void test_image_stats(void)
{
    begin("image_stats");
    /* 8x8 constant mid-grey: zero contrast, zero edge density, entropy 0 */
    size_t w = 8, h = 8;
    uint8_t *g = (uint8_t *)calloc(w * h, 1);
    memset(g, 128, w * h);
    ae_image_stats st;
    CHECK_ST(ae_image_analyze_gray(g, w, h, &st), AE_OK);
    CHECK(NEAR(st.mean, 128.0 / 255.0, 1e-12));
    CHECK(NEAR(st.stddev, 0.0, 1e-12));
    CHECK(NEAR(st.entropy, 0.0, 1e-9));
    CHECK(NEAR(st.edge_density, 0.0, 1e-12));
    CHECK(NEAR(st.exposure_clip_low, 0.0, 0.0));
    CHECK(st.pixels == 64);

    /* half black / half white: max contrast, high entropy, clipping */
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) g[y * w + x] = (x < w / 2) ? 0 : 255;
    ae_image_analyze_gray(g, w, h, &st);
    CHECK(NEAR(st.mean, 0.5, 1e-12));
    CHECK(NEAR(st.entropy, 1.0, 1e-9));
    CHECK(NEAR(st.exposure_clip_low, 0.5, 1e-12));
    CHECK(NEAR(st.exposure_clip_high, 0.5, 1e-12));
    CHECK(st.edge_density > 0.0);
    free(g);

    /* RGB path + descriptor */
    size_t n = 32 * 32;
    uint8_t *rgb = (uint8_t *)malloc(n * 3);
    for (size_t i = 0; i < n; ++i) {
        rgb[3 * i] = (uint8_t)(i % 255);
        rgb[3 * i + 1] = (uint8_t)((i * 7) % 255);
        rgb[3 * i + 2] = (uint8_t)((i * 13) % 255);
    }
    ae_image_stats st2;
    CHECK_ST(ae_image_analyze_rgb(rgb, 32, 32, &st2), AE_OK);
    CHECK(st2.pixels == n);
    ae_buffer d1 = { 0 }, d2 = { 0 };
    CHECK_ST(ae_image_descriptor_rgb(rgb, 32, 32, 4, &d1), AE_OK);
    CHECK(d1.len == 4 * 4 * 6);
    ae_image_descriptor_rgb(rgb, 32, 32, 4, &d2);
    int same = 1;
    for (size_t i = 0; i < d1.len; ++i) if (d1.data[i] != d2.data[i]) same = 0;
    CHECK(same);                                   /* deterministic */
    double nrm; ae_vec_norm(d1.data, d1.len, &nrm);
    CHECK(NEAR(nrm, 1.0, 1e-12));                  /* unit norm */
    ae_buffer_free(&d1); ae_buffer_free(&d2);
    free(rgb);

    /* degenerate sizes are rejected, not UB */
    ae_image_stats tmp;
    CHECK_ST(ae_image_analyze_gray(NULL, 4, 4, &tmp), AE_ERR_NULL);
    CHECK_ST(ae_image_analyze_gray(g, 0, 4, &tmp), AE_ERR_SIZE_ZERO);
}

static void test_force_target(void)
{
    begin("force_target");
    ae_cpu_set_policy(AE_DISPATCH_SCALAR);
    CHECK(ae_cpu_get_policy() == AE_DISPATCH_SCALAR);
    CHECK(strcmp(ae_cpu_target_name(), "scalar") == 0);
    double a[8] = { 1, 2, 3, 4, 5, 6, 7, 8 }, b[8] = { 1, 1, 1, 1, 1, 1, 1, 1 }, s;
    CHECK_ST(ae_vec_dot(a, b, 8, &s), AE_OK);
    CHECK(NEAR(s, 36.0, 1e-12));
    ae_cpu_set_policy(AE_DISPATCH_AUTO);
    CHECK(ae_cpu_get_policy() == AE_DISPATCH_AUTO);
    CHECK(strcmp(ae_cpu_target_name(), "scalar") != 0 || ae_cpu_features() == AE_CPU_SCALAR);
}

int main(void)
{
    printf("aenum test suite\n");
    printf("target=%s\n", ae_cpu_target_name());
    test_status_strings();
    test_cpu_detection();
    test_alignment();
    test_null_and_bounds();
    test_normalize_cosine();
    test_simd_scalar_equivalence();
    test_batch_cosine();
    test_stats_accuracy();
    test_histogram_entropy();
    test_topk();
    test_softmax_logsumexp();
    test_matmul_transpose();
    test_covariance();
    test_image_stats();
    test_force_target();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
