#ifndef KMEANS_COMMON_H
#define KMEANS_COMMON_H

/**
 * DSA5208 Project 3: Scalable Parallel K-Means
 * Shared Header: Common Data Structures, Numerical Functions & Validation
 *
 * Provides header-only implementations for:
 * 1. Vectorized Squared Euclidean Distance
 * 2. Deterministic Initial Centroid Selection
 * 3. Exact Clustering Objective: Inertia / Sum of Squared Errors (SSE)
 * 4. Cache-Line Memory Alignment & Padding Helpers
 * 5. Numerical Parity & Centroid Comparison Engine
 * 6. Empty Cluster Detection
 */

#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cfloat>
#include <cstdint>
#include <algorithm>

namespace kmeans {

// Compute squared Euclidean distance between observation pt and centroid c
inline double squared_euclidean_distance(const double* pt, const double* c, size_t d) {
    double dist = 0.0;
#if defined(_OPENMP)
    #pragma omp simd reduction(+:dist)
#endif
    for (size_t j = 0; j < d; ++j) {
        double diff = pt[j] - c[j];
        dist += diff * diff;
    }
    return dist;
}

// Deterministic centroid initialization matching serial baseline: idx = k * (N / K)
inline void initialize_centroids_deterministic(const double* data, double* centroids,
                                              size_t N, size_t K, size_t d) {
    size_t step = (K > 0) ? (N / K) : 0;
    for (size_t k = 0; k < K; ++k) {
        size_t idx = k * step;
        for (size_t j = 0; j < d; ++j) {
            centroids[k * d + j] = data[idx * d + j];
        }
    }
}

// Compute clustering objective: Sum of Squared Errors (Inertia / Distortion)
// J = \sum_{i=0}^{N-1} \min_{k=0..K-1} \|x_i - \mu_k\|^2
inline double compute_inertia_sse(const double* data, const double* centroids,
                                  size_t N, size_t K, size_t d) {
    double total_sse = 0.0;
#if defined(_OPENMP)
    #pragma omp parallel for reduction(+:total_sse) schedule(static)
#endif
    for (size_t i = 0; i < N; ++i) {
        const double* pt = &data[i * d];
        double best_dist = DBL_MAX;
        for (size_t k = 0; k < K; ++k) {
            double dist = squared_euclidean_distance(pt, &centroids[k * d], d);
            if (dist < best_dist) {
                best_dist = dist;
            }
        }
        total_sse += best_dist;
    }
    return total_sse;
}

// Helper to calculate 64-byte (cache-line) aligned element strides for thread-private buffers
inline size_t get_padded_stride(size_t element_count, size_t alignment_elements = 8) {
    return ((element_count + alignment_elements - 1) / alignment_elements) * alignment_elements;
}

// Check for empty (starved) clusters and report their indices
inline size_t detect_empty_clusters(const unsigned long long* counts, size_t K,
                                    std::vector<size_t>& empty_indices) {
    empty_indices.clear();
    for (size_t k = 0; k < K; ++k) {
        if (counts[k] == 0) {
            empty_indices.push_back(k);
        }
    }
    return empty_indices.size();
}

// Centroid comparison result structure
struct ValidationResult {
    bool passed = false;
    double max_abs_error = 0.0;
    double rmse = 0.0;
    size_t nonfinite_count = 0;
    size_t evaluated_elements = 0;
    std::string summary = "";
};

// Compare candidate centroids against a reference vector
inline ValidationResult compare_centroids(const double* candidate, const double* reference,
                                          size_t K, size_t d, double tolerance = 1e-9) {
    ValidationResult res;
    res.evaluated_elements = K * d;
    double sum_sq_err = 0.0;

    for (size_t i = 0; i < K * d; ++i) {
        double c_val = candidate[i];
        double r_val = reference[i];

        if (std::isnan(c_val) || std::isinf(c_val) || std::isnan(r_val) || std::isinf(r_val)) {
            res.nonfinite_count++;
            continue;
        }

        double diff = std::abs(c_val - r_val);
        if (diff > res.max_abs_error) {
            res.max_abs_error = diff;
        }
        sum_sq_err += diff * diff;
    }

    if (res.evaluated_elements > 0) {
        res.rmse = std::sqrt(sum_sq_err / static_cast<double>(res.evaluated_elements));
    }

    res.passed = (res.nonfinite_count == 0) && (res.max_abs_error <= tolerance);

    std::ostringstream oss;
    oss << "Evaluated: " << res.evaluated_elements << " floats | "
        << "Max Abs Error: " << std::scientific << std::setprecision(4) << res.max_abs_error << " | "
        << "RMSE: " << std::scientific << std::setprecision(4) << res.rmse << " | "
        << "Status: " << (res.passed ? "PASSED" : "FAILED (exceeds tolerance or nonfinite)");
    res.summary = oss.str();
    return res;
}

// Load reference centroids from formatted text file
inline bool load_centroids_file(const std::string& filename, std::vector<double>& centroids,
                                size_t& K, size_t& d) {
    std::ifstream in(filename);
    if (!in.is_open()) return false;

    if (!(in >> K >> d) || K == 0 || d == 0) return false;

    centroids.resize(K * d);
    for (size_t i = 0; i < K * d; ++i) {
        if (!(in >> centroids[i])) return false;
    }
    return true;
}

// Save centroids with metadata comments for complete provenance
inline bool save_centroids_with_provenance(const std::string& filename, const double* centroids,
                                          size_t K, size_t d, size_t iters = 0,
                                          double total_time_sec = 0.0, double sse = 0.0) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    out << std::setprecision(14);
    out << K << " " << d << "\n";
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < d; ++j) {
            out << centroids[k * d + j] << (j + 1 == d ? "" : " ");
        }
        out << "\n";
    }

    if (iters > 0 || total_time_sec > 0.0 || sse > 0.0) {
        out << "# Metadata:\n"
            << "# Iterations: " << iters << "\n"
            << "# Total Time (s): " << std::fixed << std::setprecision(6) << total_time_sec << "\n"
            << "# Inertia / SSE : " << std::scientific << std::setprecision(8) << sse << "\n";
    }
    return true;
}

} // namespace kmeans

#endif // KMEANS_COMMON_H
