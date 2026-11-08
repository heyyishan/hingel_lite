/* image.c — deterministic image statistics and fallback descriptors.
 * Deliberately free of external dependencies: the Python layer supplies pixels.
 */
#include "aenum.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline double luma_from_rgb(uint8_t r, uint8_t g, uint8_t b)
{
    /* Rec. 709 luma coefficients */
    return (0.2126 * (double)r + 0.7152 * (double)g + 0.0722 * (double)b) / 255.0;
}

/* Single-pass statistics over a normalised luma plane. */
static ae_status analyze_plane(const double *plane, size_t w, size_t h,
                               ae_image_stats *out,
                               double *grad_mag_mean, double *hf_energy)
{
    const size_t n = w * h;
    if (n == 0) return AE_ERR_SIZE_ZERO;

    double sum = 0.0;
    size_t clip_lo = 0, clip_hi = 0, shadow = 0, highlight = 0;
    size_t hist[256] = { 0 };

    for (size_t i = 0; i < n; ++i) {
        double v = plane[i];
        sum += v;
        if (v <= 0.02) clip_lo++;
        if (v >= 0.98) clip_hi++;
        if (v < 0.20) shadow++;
        if (v > 0.85) highlight++;
        size_t b = (size_t)(v * 255.0);
        if (b > 255) b = 255;
        hist[b]++;
    }
    const double mean = sum / (double)n;
    /* Two-pass variance. The one-pass sumsq/n - mean^2 form is catastrophically
     * cancelling for near-constant planes (measured 3.6e-8 stddev on a flat
     * 8x8 mid-grey image, which a regression test caught). */
    double dsum = 0.0;
    for (size_t i = 0; i < n; ++i) { double d = plane[i] - mean; dsum += d * d; }
    const double var = dsum / (double)n;
    out->mean = mean;
    out->stddev = sqrt(var > 0.0 ? var : 0.0);
    out->contrast_rms = out->stddev;
    out->exposure_clip_low  = (double)clip_lo / (double)n;
    out->exposure_clip_high = (double)clip_hi / (double)n;
    out->shadow_fraction    = (double)shadow / (double)n;
    out->highlight_fraction = (double)highlight / (double)n;

    double H = 0.0;
    for (size_t b = 0; b < 256; ++b) {
        if (hist[b] == 0) continue;
        double p = (double)hist[b] / (double)n;
        H -= p * log2(p);
    }
    out->entropy = H;

    /* Sobel gradient magnitude (interior pixels only) */
    double gsum = 0.0;
    size_t gcount = 0;
    double hf = 0.0;
    if (w >= 3 && h >= 3) {
        for (size_t y = 1; y + 1 < h; ++y) {
            const double *r0 = plane + (y - 1) * w;
            const double *r1 = plane + y * w;
            const double *r2 = plane + (y + 1) * w;
            for (size_t x = 1; x + 1 < w; ++x) {
                double gx = (r0[x + 1] + 2.0 * r1[x + 1] + r2[x + 1])
                          - (r0[x - 1] + 2.0 * r1[x - 1] + r2[x - 1]);
                double gy = (r2[x - 1] + 2.0 * r2[x] + r2[x + 1])
                          - (r0[x - 1] + 2.0 * r0[x] + r0[x + 1]);
                double m = sqrt(gx * gx + gy * gy) / 4.0; /* normalised to <= ~1 */
                gsum += m;
                gcount++;
                if (m > 0.12) hf += 1.0;
            }
        }
    }
    out->edge_density = gcount ? gsum / (double)gcount : 0.0;
    out->background_complexity = gcount ? hf / (double)gcount : 0.0;
    out->pixels = n;
    if (grad_mag_mean) *grad_mag_mean = out->edge_density;
    if (hf_energy) *hf_energy = out->background_complexity;
    return AE_OK;
}

ae_status ae_image_analyze_gray(const uint8_t *gray, size_t w, size_t h,
                                ae_image_stats *out)
{
    if (!gray || !out) return AE_ERR_NULL;
    if (w == 0 || h == 0) return AE_ERR_SIZE_ZERO;
    size_t n = w * h;
    double *plane = (double *)malloc(n * sizeof(double));
    if (!plane) return AE_ERR_ALLOC;
    for (size_t i = 0; i < n; ++i) plane[i] = (double)gray[i] / 255.0;
    ae_status st = analyze_plane(plane, w, h, out, NULL, NULL);
    free(plane);
    return st;
}

ae_status ae_image_analyze_rgb(const uint8_t *rgb, size_t w, size_t h,
                               ae_image_stats *out)
{
    if (!rgb || !out) return AE_ERR_NULL;
    if (w == 0 || h == 0) return AE_ERR_SIZE_ZERO;
    size_t n = w * h;
    double *plane = (double *)malloc(n * sizeof(double));
    if (!plane) return AE_ERR_ALLOC;
    for (size_t i = 0; i < n; ++i)
        plane[i] = luma_from_rgb(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]);
    ae_status st = analyze_plane(plane, w, h, out, NULL, NULL);
    free(plane);
    return st;
}

/* Block-averaged colour descriptor: grid x grid cells, 6 channels each
 * (mean R,G,B, local stddev luma, mean |Sobel_x|, mean |Sobel_y|).
 * Produces 6*grid^2 numbers, L2-normalised. Deterministic and dependency-free;
 * used only when a pretrained vision encoder is unavailable. */
ae_status ae_image_descriptor_rgb(const uint8_t *rgb, size_t w, size_t h,
                                  size_t grid, ae_buffer *out)
{
    if (!rgb || !out) return AE_ERR_NULL;
    if (w < 2 || h < 2) return AE_ERR_SIZE_ZERO;
    if (grid < 1 || grid > 32) return AE_ERR_INDEX_RANGE;
    const size_t dim = grid * grid * 6;
    ae_buffer b = ae_buffer_alloc(dim);
    if (!b.data) return AE_ERR_ALLOC;

    for (size_t gy = 0; gy < grid; ++gy) {
        for (size_t gx = 0; gx < grid; ++gx) {
            size_t x0 = (gx * w) / grid, x1 = ((gx + 1) * w) / grid;
            size_t y0 = (gy * h) / grid, y1 = ((gy + 1) * h) / grid;
            if (x1 <= x0) x1 = x0 + 1;
            if (y1 <= y0) y1 = y0 + 1;
            double sr = 0, sg = 0, sb = 0, sl = 0, sl2 = 0, gx_sum = 0, gy_sum = 0;
            size_t cnt = 0, gcnt = 0;
            for (size_t y = y0; y < y1 && y < h; ++y) {
                for (size_t x = x0; x < x1 && x < w; ++x) {
                    size_t i = y * w + x;
                    double r = rgb[3 * i] / 255.0, g = rgb[3 * i + 1] / 255.0, bl = rgb[3 * i + 2] / 255.0;
                    sr += r; sg += g; sb += bl;
                    double l = luma_from_rgb(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]);
                    sl += l; sl2 += l * l; cnt++;
                    if (x + 1 < w && y + 1 < h) {
                        size_t j = i + 1, k = i + w;
                        double lx = luma_from_rgb(rgb[3 * j], rgb[3 * j + 1], rgb[3 * j + 2]);
                        double ly = luma_from_rgb(rgb[3 * k], rgb[3 * k + 1], rgb[3 * k + 2]);
                        gx_sum += fabs(lx - l);
                        gy_sum += fabs(ly - l);
                        gcnt++;
                    }
                }
            }
            if (cnt == 0) cnt = 1;
            if (gcnt == 0) gcnt = 1;
            double ml = sl / (double)cnt;
            double varl = sl2 / (double)cnt - ml * ml;
            double *dst = b.data + (gy * grid + gx) * 6;
            dst[0] = sr / (double)cnt;
            dst[1] = sg / (double)cnt;
            dst[2] = sb / (double)cnt;
            dst[3] = sqrt(varl > 0.0 ? varl : 0.0);
            dst[4] = gx_sum / (double)gcnt;
            dst[5] = gy_sum / (double)gcnt;
        }
    }
    /* centre and L2-normalise */
    double mu;
    ae_status st = ae_mean(b.data, dim, &mu);
    if (st != AE_OK) { ae_buffer_free(&b); return st; }
    for (size_t i = 0; i < dim; ++i) b.data[i] -= mu;
    double nrm;
    st = ae_vec_norm(b.data, dim, &nrm);
    if (st != AE_OK || nrm <= 0.0) { ae_buffer_free(&b); return AE_ERR_NOT_NORMALIZED; }
    for (size_t i = 0; i < dim; ++i) b.data[i] /= nrm;
    *out = b;
    return AE_OK;
}
