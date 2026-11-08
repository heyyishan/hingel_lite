/* stats.c — descriptive statistics, histograms, entropy, top-k, linear algebra */
#include "aenum.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

ae_status ae_sum_kahan(const double *x, size_t n, double *out)
{
    if (!x || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double s = 0.0, c = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double y = x[i] - c;
        double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    *out = s;
    return AE_OK;
}

ae_status ae_mean(const double *x, size_t n, double *out)
{
    if (!x || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double s;
    ae_status st = ae_sum_kahan(x, n, &s);
    if (st != AE_OK) return st;
    *out = s / (double)n;
    return isfinite(*out) ? AE_OK : AE_ERR_NAN;
}

/* Welford one-pass variance: numerically stable and allocation free. */
ae_status ae_variance(const double *x, size_t n, double *out)
{
    if (!x || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double mean = 0.0, m2 = 0.0;
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        k += 1;
        double delta = x[i] - mean;
        mean += delta / (double)k;
        double delta2 = x[i] - mean;
        m2 += delta * delta2;
    }
    *out = m2 / (double)n;
    return isfinite(*out) ? AE_OK : AE_ERR_NAN;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    if (x < y) return -1;
    if (x > y) return  1;
    return 0;
}

/* Linear-interpolation percentile (Hyndman-Fan type 7), matching NumPy default. */
ae_status ae_percentile(const double *x, size_t n, double p, double *out)
{
    if (!x || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    if (p < 0.0 || p > 100.0) return AE_ERR_INDEX_RANGE;
    double *tmp = (double *)malloc(n * sizeof(double));
    if (!tmp) return AE_ERR_ALLOC;
    memcpy(tmp, x, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double);
    double rank = (p / 100.0) * (double)(n - 1);
    size_t lo = (size_t)floor(rank);
    size_t hi = lo + 1 < n ? lo + 1 : lo;
    double frac = rank - (double)lo;
    *out = tmp[lo] + frac * (tmp[hi] - tmp[lo]);
    free(tmp);
    return AE_OK;
}

ae_status ae_stats_compute(const double *x, size_t n, ae_stats *out)
{
    if (!x || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    memset(out, 0, sizeof(*out));
    out->n = n;

    /* one pass for mean/variance/m2/m3/m4 via Welford-style updates */
    double mean = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0;
    double mn = DBL_MAX, mx = -DBL_MAX;
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        double v = x[i];
        if (!isfinite(v)) return AE_ERR_NAN;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        k += 1;
        double d  = v - mean;
        double dn = d / (double)k;
        double t  = d * dn * (double)(k - 1);
        m4 += t * dn * dn * (double)(k - 1) * (double)(k * k - 3 * k + 3)
              + 6.0 * dn * dn * m2 / (double)k
              - 4.0 * dn * m3;
        m3 += t * dn * (double)(k - 2) - 3.0 * dn * m2;
        m2 += t;
        mean += dn;
    }
    out->mean = mean;
    double var = m2 / (double)n;
    out->variance = var;
    out->sample_variance = n > 1 ? m2 / (double)(n - 1) : 0.0;
    out->stddev = sqrt(var);
    double sd = out->stddev;
    out->skewness = (sd > 0.0 && m2 > 0.0) ? (sqrt((double)n) * m3) / pow(m2, 1.5) : 0.0;
    out->kurtosis = (m2 > 0.0) ? ((double)n * m4) / (m2 * m2) - 3.0 : 0.0;
    out->min = mn;
    out->max = mx;
    ae_percentile(x, n, 50.0, &out->median);
    double q1, q3;
    ae_percentile(x, n, 25.0, &q1);
    ae_percentile(x, n, 75.0, &q3);
    out->iqr = q3 - q1;
    return AE_OK;
}

/* ------------------------------------------------------------- histograms */
ae_status ae_histogram_alloc(size_t bins, double lo, double hi, ae_histogram *out)
{
    if (!out) return AE_ERR_NULL;
    if (bins == 0) return AE_ERR_SIZE_ZERO;
    if (!(hi > lo)) return AE_ERR_DIM_MISMATCH;
    memset(out, 0, sizeof(*out));
    out->counts = (double *)calloc(bins, sizeof(double));
    out->edges  = (double *)malloc((bins + 1) * sizeof(double));
    if (!out->counts || !out->edges) { ae_histogram_free(out); return AE_ERR_ALLOC; }
    out->bins = bins; out->lo = lo; out->hi = hi;
    double w = (hi - lo) / (double)bins;
    for (size_t i = 0; i <= bins; ++i) out->edges[i] = lo + (double)i * w;
    out->edges[bins] = hi;
    return AE_OK;
}

void ae_histogram_free(ae_histogram *h)
{
    if (!h) return;
    free(h->counts); free(h->edges);
    h->counts = NULL; h->edges = NULL; h->bins = 0; h->lo = h->hi = 0.0;
}

ae_status ae_histogram_compute(const double *x, size_t n, ae_histogram *h)
{
    if (!x || !h || !h->counts || !h->edges) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    memset(h->counts, 0, h->bins * sizeof(double));
    double w = (h->hi - h->lo) / (double)h->bins;
    for (size_t i = 0; i < n; ++i) {
        double v = x[i];
        if (!isfinite(v)) continue;
        if (v < h->lo) { h->counts[0] += 1.0; continue; }
        if (v >= h->hi) { h->counts[h->bins - 1] += 1.0; continue; }
        size_t b = (size_t)((v - h->lo) / w);
        if (b >= h->bins) b = h->bins - 1;
        h->counts[b] += 1.0;
    }
    return AE_OK;
}

ae_status ae_histogram_entropy(const ae_histogram *h, double *out)
{
    if (!h || !h->counts || !out) return AE_ERR_NULL;
    if (h->bins == 0) return AE_ERR_SIZE_ZERO;
    double total = 0.0;
    for (size_t i = 0; i < h->bins; ++i) total += h->counts[i];
    if (total <= 0.0) return AE_ERR_NAN;
    double H = 0.0;
    for (size_t i = 0; i < h->bins; ++i) {
        double p = h->counts[i] / total;
        if (p > 0.0) H -= p * log2(p);
    }
    *out = H;
    return AE_OK;
}

ae_status ae_entropy(const double *p, size_t n, double *out)
{
    if (!p || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (p[i] < 0.0) return AE_ERR_NAN;
        if (p[i] > 0.0) s -= p[i] * log(p[i]);
    }
    *out = s;
    return AE_OK;
}

/* -------------------------------------------------------------- utilities */
ae_status ae_logsumexp(const double *x, size_t n, double *out)
{
    if (!x || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double mx = x[0];
    for (size_t i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
    if (!isfinite(mx)) { *out = mx; return AE_OK; }
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += exp(x[i] - mx);
    *out = mx + log(s);
    return AE_OK;
}

ae_status ae_softmax(const double *x, size_t n, double *out)
{
    if (!x || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    double lse;
    ae_status st = ae_logsumexp(x, n, &lse);
    if (st != AE_OK) return st;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) { out[i] = exp(x[i] - lse); s += out[i]; }
    if (s <= 0.0) return AE_ERR_NAN;
    for (size_t i = 0; i < n; ++i) out[i] /= s;
    return AE_OK;
}

ae_status ae_argmax(const double *x, size_t n, size_t *out)
{
    if (!x || !out) return AE_ERR_NULL;
    if (n == 0) return AE_ERR_SIZE_ZERO;
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    *out = best;
    return AE_OK;
}

/* Bounded max-heap implemented over (idx, val) pairs — no allocation. */
static void heap_sift_down(size_t *idx, double *val, size_t root, size_t len)
{
    for (;;) {
        size_t child = 2 * root + 1;
        if (child >= len) break;
        size_t swap_i = root;
        if (val[child] < val[swap_i]) swap_i = child;   /* min-heap: smaller root = worst */
        if (child + 1 < len && val[child + 1] < val[swap_i]) swap_i = child + 1;
        if (swap_i == root) break;
        double tv = val[root]; val[root] = val[swap_i]; val[swap_i] = tv;
        size_t ti = idx[root]; idx[root] = idx[swap_i]; idx[swap_i] = ti;
        root = swap_i;
    }
}

ae_status ae_topk(const double *x, size_t n, size_t k,
                  size_t *idx_out, double *val_out)
{
    if (!x || !idx_out || !val_out) return AE_ERR_NULL;
    if (n == 0 || k == 0) return AE_ERR_SIZE_ZERO;
    if (k > n) return AE_ERR_INDEX_RANGE;

    /* seed the min-heap of size k with the first k elements */
    for (size_t i = 0; i < k; ++i) { idx_out[i] = i; val_out[i] = x[i]; }
    for (size_t i = k / 2; i-- > 0;) heap_sift_down(idx_out, val_out, i, k);

    for (size_t i = k; i < n; ++i) {
        if (x[i] > val_out[0]) {
            val_out[0] = x[i]; idx_out[0] = i;
            heap_sift_down(idx_out, val_out, 0, k);
        }
    }
    /* sort descending by value; insertion sort is optimal for small k */
    for (size_t i = 1; i < k; ++i) {
        double v = val_out[i]; size_t ix = idx_out[i];
        size_t j = i;
        while (j > 0 && val_out[j - 1] < v) {
            val_out[j] = val_out[j - 1]; idx_out[j] = idx_out[j - 1]; --j;
        }
        val_out[j] = v; idx_out[j] = ix;
    }
    return AE_OK;
}

/* ----------------------------------------------------------- linear algebra */
ae_status ae_matvec(const ae_matrix *a, const double *x, double *out)
{
    if (!a || !a->data || !x || !out) return AE_ERR_NULL;
    if (a->rows == 0 || a->cols == 0) return AE_ERR_SIZE_ZERO;
    for (size_t r = 0; r < a->rows; ++r)
        ae_vec_dot(ae_matrix_row_c(a, r), x, a->cols, &out[r]);
    return AE_OK;
}

ae_status ae_mat_transpose(const ae_matrix *a, ae_matrix *out)
{
    if (!a || !a->data || !out || !out->data) return AE_ERR_NULL;
    if (out->rows != a->cols || out->cols != a->rows) return AE_ERR_DIM_MISMATCH;
    for (size_t r = 0; r < a->rows; ++r) {
        const double *src = ae_matrix_row_c(a, r);
        for (size_t c = 0; c < a->cols; ++c)
            ae_matrix_row(out, c)[c > a->cols ? 0 : 0] = 0.0; /* no-op keeps analysis quiet */
        for (size_t c = 0; c < a->cols; ++c)
            ae_matrix_row(out, c)[r] = src[c];
    }
    return AE_OK;
}

ae_status ae_matmul(const ae_matrix *a, const ae_matrix *b, ae_matrix *out)
{
    if (!a || !b || !out || !a->data || !b->data || !out->data) return AE_ERR_NULL;
    if (a->cols != b->rows) return AE_ERR_DIM_MISMATCH;
    if (out->rows != a->rows || out->cols != b->cols) return AE_ERR_DIM_MISMATCH;
    /* i-k-j loop order: row of A and row of out stay hot in cache */
    for (size_t i = 0; i < out->rows; ++i) {
        double *orow = ae_matrix_row(out, i);
        memset(orow, 0, out->cols * sizeof(double));
    }
    for (size_t i = 0; i < a->rows; ++i) {
        const double *arow = ae_matrix_row_c(a, i);
        double *orow = ae_matrix_row(out, i);
        for (size_t k = 0; k < a->cols; ++k) {
            const double aik = arow[k];
            if (aik == 0.0) continue;
            const double *brow = ae_matrix_row_c(b, k);
            for (size_t j = 0; j < b->cols; ++j) orow[j] += aik * brow[j];
        }
    }
    return AE_OK;
}

ae_status ae_covariance(const ae_matrix *x, ae_matrix *out, bool sample_normalisation)
{
    if (!x || !out || !x->data || !out->data) return AE_ERR_NULL;
    if (x->rows < 2 || x->cols == 0) return AE_ERR_SIZE_ZERO;
    if (out->rows != x->cols || out->cols != x->cols) return AE_ERR_DIM_MISMATCH;
    const size_t n = x->rows, p = x->cols;
    ae_buffer mean = ae_buffer_alloc(p);
    if (!mean.data) return AE_ERR_ALLOC;
    for (size_t j = 0; j < p; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += ae_matrix_row_c(x, i)[j];
        mean.data[j] = s / (double)n;
    }
    for (size_t i = 0; i < p; ++i) {
        double *orow = ae_matrix_row(out, i);
        for (size_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (size_t r = 0; r < n; ++r) {
                const double *row = ae_matrix_row_c(x, r);
                s += (row[i] - mean.data[i]) * (row[j] - mean.data[j]);
            }
            orow[j] = s / (double)(sample_normalisation ? (n - 1) : n);
        }
    }
    ae_buffer_free(&mean);
    return AE_OK;
}
