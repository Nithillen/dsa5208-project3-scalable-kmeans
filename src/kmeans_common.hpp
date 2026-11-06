#ifndef KMEANS_COMMON_HPP
#define KMEANS_COMMON_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cfloat>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

// Fast deterministic counter-based SplitMix64 PRNG so any rank can generate
// global point `global_idx` identically to Serial (independent of MPI rank count P!)
inline uint64_t splitmix64_stateless(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline double uniform01_stateless(uint64_t seed, uint64_t stream_idx) {
    uint64_t h = splitmix64_stateless(seed ^ (stream_idx * 0x9e3779b97f4a7c15ULL));
    return (h >> 11) * (1.0 / 9007199254740992.0); // [0, 1) with 53-bit precision
}

// Box-Muller transform for stateless deterministic Gaussian N(0, sigma^2) per (global_idx, dim)
inline double normal_stateless(uint64_t seed, uint64_t global_idx, size_t dim, double sigma = 3.0) {
    uint64_t key1 = (global_idx * 65537ULL + static_cast<uint64_t>(dim) * 2ULL + 1ULL);
    uint64_t key2 = (global_idx * 65537ULL + static_cast<uint64_t>(dim) * 2ULL + 2ULL);
    double u1 = uniform01_stateless(seed, key1);
    double u2 = uniform01_stateless(seed, key2);
    if (u1 < 1e-15) u1 = 1e-15;
    return sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}

// Generate deterministic ground-truth cluster centers (identical on all ranks and serial)
inline void generate_true_centers(std::vector<double>& true_centers, size_t K, size_t d, unsigned int seed) {
    true_centers.resize(K * d);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> center_dist(-50.0, 50.0);
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < d; ++j) {
            true_centers[k * d + j] = center_dist(gen);
        }
    }
}

// Generate a single synthetic point at global row index `global_idx` (bitwise identical across Serial, MPI, OpenMP, GPU)
inline void generate_synthetic_point(double* pt_out, const std::vector<double>& true_centers,
                                     size_t global_idx, size_t d, size_t K, unsigned int seed) {
    uint64_t c_hash = splitmix64_stateless(static_cast<uint64_t>(seed) + global_idx * 0x517cc1b727220a95ULL);
    size_t c = static_cast<size_t>(c_hash % K);
    for (size_t j = 0; j < d; ++j) {
        pt_out[j] = true_centers[c * d + j] + normal_stateless(seed, global_idx, j, 3.0);
    }
}

// Generate contiguous partition [global_start, global_start + local_N) AND deterministic initial centroids
// Initial centroids always pick evenly spaced global points `idx = k * (total_N / K)` — matching Serial 100%!
inline void generate_synthetic_partition_and_centroids(std::vector<double>& local_data,
                                                       std::vector<double>& centroids,
                                                       size_t total_N, size_t global_start, size_t local_N,
                                                       size_t d, size_t K, unsigned int seed) {
    std::vector<double> true_centers;
    generate_true_centers(true_centers, K, d, seed);

    local_data.resize(local_N * d);
    for (size_t i = 0; i < local_N; ++i) {
        generate_synthetic_point(&local_data[i * d], true_centers, global_start + i, d, K, seed);
    }

    centroids.resize(K * d);
    size_t step = total_N / K;
    for (size_t k = 0; k < K; ++k) {
        size_t idx = k * step;
        generate_synthetic_point(&centroids[k * d], true_centers, idx, d, K, seed);
    }
}

// Deterministic initialization from loaded dataset: pick evenly spaced points `idx = k * (N / K)`
inline void initialize_centroids(const std::vector<double>& data, std::vector<double>& centroids,
                                 size_t N, size_t K, size_t d) {
    centroids.resize(K * d);
    size_t step = N / K;
    for (size_t k = 0; k < K; ++k) {
        size_t idx = k * step;
        for (size_t j = 0; j < d; ++j) {
            centroids[k * d + j] = data[idx * d + j];
        }
    }
}

// High-speed CSV loader using strtod pointer scanning for multi-million row files (up to 11M HIGGS rows)
inline bool load_csv_data(const std::string& filename, std::vector<double>& data,
                          size_t& N, size_t& d, size_t max_rows = 0) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::vector<char> io_buf(8 * 1024 * 1024);
    file.rdbuf()->pubsetbuf(io_buf.data(), io_buf.size());

    std::string line;
    data.clear();
    N = 0;
    d = 0;

    std::vector<double> row;
    row.reserve(32);

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        row.clear();
        const char* ptr = line.c_str();
        char* endptr = nullptr;
        bool valid_row = true;

        while (*ptr != '\0' && *ptr != '\r' && *ptr != '\n') {
            double val = std::strtod(ptr, &endptr);
            if (ptr == endptr) {
                valid_row = false;
                break;
            }
            row.push_back(val);
            ptr = endptr;
            while (*ptr == ',' || *ptr == ' ' || *ptr == '\t') {
                ++ptr;
            }
        }

        if (valid_row && !row.empty()) {
            if (d == 0) {
                d = row.size();
                if (max_rows > 0) {
                    data.reserve(max_rows * d);
                } else {
                    data.reserve(600000ULL * d);
                }
            } else if (row.size() != d) {
                continue;
            }
            data.insert(data.end(), row.begin(), row.end());
            N++;
            if (max_rows > 0 && N >= max_rows) break;
        }
    }
    return (N > 0 && d > 0);
}

// Memory-safe parallel/chunked CSV partition reader for distributed ranks (avoids full dataset allocation on Rank 0)
inline bool load_csv_partition(const std::string& filename, std::vector<double>& local_data,
                               std::vector<double>& centroids, size_t& total_N, size_t& local_N,
                               size_t& d, size_t K, int rank, int num_procs, size_t max_rows = 0) {
    // Pass 1: Count total valid rows and dimension d
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) return false;
    std::vector<char> io_buf(8 * 1024 * 1024);
    file.rdbuf()->pubsetbuf(io_buf.data(), io_buf.size());

    std::string line;
    total_N = 0;
    d = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (d == 0) {
            const char* ptr = line.c_str();
            char* endptr = nullptr;
            while (*ptr != '\0' && *ptr != '\r' && *ptr != '\n') {
                std::strtod(ptr, &endptr);
                if (ptr == endptr) break;
                d++;
                ptr = endptr;
                while (*ptr == ',' || *ptr == ' ' || *ptr == '\t') ++ptr;
            }
        }
        total_N++;
        if (max_rows > 0 && total_N >= max_rows) break;
    }
    if (total_N == 0 || d == 0) return false;

    size_t base_rows = total_N / static_cast<size_t>(num_procs);
    size_t rem_rows  = total_N % static_cast<size_t>(num_procs);
    size_t row_start = 0;
    for (int p = 0; p < rank; ++p) {
        row_start += base_rows + (static_cast<size_t>(p) < rem_rows ? 1 : 0);
    }
    local_N = base_rows + (static_cast<size_t>(rank) < rem_rows ? 1 : 0);
    size_t row_end = row_start + local_N;

    local_data.clear();
    local_data.reserve(local_N * d);
    centroids.assign(K * d, 0.0);
    size_t step = total_N / K;

    // Pass 2: Read only [row_start, row_end) into local_data and capture initial centroids at idx = k * step
    file.clear();
    file.seekg(0, std::ios::beg);
    size_t cur_row = 0;
    std::vector<double> row;
    row.reserve(d);

    while (cur_row < total_N && std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        bool is_centroid_row = (step > 0 && (cur_row % step == 0) && (cur_row / step < K));
        bool is_local_row    = (cur_row >= row_start && cur_row < row_end);

        if (is_centroid_row || is_local_row) {
            row.clear();
            const char* ptr = line.c_str();
            char* endptr = nullptr;
            while (*ptr != '\0' && *ptr != '\r' && *ptr != '\n') {
                double val = std::strtod(ptr, &endptr);
                if (ptr == endptr) break;
                row.push_back(val);
                ptr = endptr;
                while (*ptr == ',' || *ptr == ' ' || *ptr == '\t') ++ptr;
            }
            if (row.size() == d) {
                if (is_centroid_row) {
                    size_t k = cur_row / step;
                    for (size_t j = 0; j < d; ++j) centroids[k * d + j] = row[j];
                }
                if (is_local_row) {
                    local_data.insert(local_data.end(), row.begin(), row.end());
                }
            }
        }
        cur_row++;
    }
    return true;
}

// Compute local Within-Cluster Sum of Squares (SSE / Inertia): J_local = sum_{i=1}^{local_N} ||x_i - mu_{a_i}||^2
inline double compute_local_sse(const std::vector<double>& local_data,
                                const std::vector<double>& centroids,
                                size_t local_N, size_t d, size_t K) {
    double sse = 0.0;
    for (size_t i = 0; i < local_N; ++i) {
        const double* pt = &local_data[i * d];
        double best_dist = DBL_MAX;
        for (size_t k = 0; k < K; ++k) {
            const double* c = &centroids[k * d];
            double dist = 0.0;
            for (size_t j = 0; j < d; ++j) {
                double diff = pt[j] - c[j];
                dist += diff * diff;
            }
            if (dist < best_dist) best_dist = dist;
        }
        sse += best_dist;
    }
    return sse;
}

// Save centroids (and optional Final SSE in header comment or footer)
inline void save_centroids_with_sse(const std::string& filename,
                                    const std::vector<double>& centroids,
                                    size_t K, size_t d, double final_sse = -1.0) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Warning: Could not save centroids to " << filename << std::endl;
        return;
    }
    out << std::setprecision(12);
    out << K << " " << d;
    if (final_sse >= 0.0) {
        out << " " << std::scientific << std::setprecision(12) << final_sse;
    }
    out << "\n" << std::defaultfloat << std::setprecision(12);
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < d; ++j) {
            out << centroids[k * d + j] << (j + 1 == d ? "" : " ");
        }
        out << "\n";
    }
    out.close();
}

// Validate parallel centroids and SSE against serial reference file
inline bool validate_centroids_and_sse(const std::string& ref_file,
                                       const std::vector<double>& centroids,
                                       size_t K, size_t d,
                                       double final_sse = -1.0,
                                       double tol_centroid = 1e-9) {
    std::ifstream in(ref_file);
    if (!in.is_open()) {
        std::cout << "[Validation] Reference file " << ref_file << " not found; skipping comparison.\n";
        return false;
    }
    std::string header_line;
    if (!std::getline(in, header_line)) return false;
    std::istringstream iss(header_line);
    size_t ref_K = 0, ref_d = 0;
    double ref_sse = -1.0;
    if (!(iss >> ref_K >> ref_d) || ref_K != K || ref_d != d) {
        std::cout << "[Validation] Dimension mismatch with " << ref_file << " (expected "
                  << K << "x" << d << ", got " << ref_K << "x" << ref_d << ").\n";
        return false;
    }
    if (!(iss >> ref_sse)) {
        ref_sse = -1.0;
    }

    double max_abs_err = 0.0;
    double sum_sq_err = 0.0;
    for (size_t i = 0; i < K * d; ++i) {
        double ref_val = 0.0;
        in >> ref_val;
        double err = std::abs(centroids[i] - ref_val);
        if (err > max_abs_err) max_abs_err = err;
        sum_sq_err += err * err;
    }
    double rmse = std::sqrt(sum_sq_err / static_cast<double>(K * d));
    double rel_sse_err = (ref_sse > 0.0 && final_sse > 0.0)
                         ? std::abs(final_sse - ref_sse) / ref_sse
                         : 0.0;

    bool passed = (max_abs_err <= tol_centroid);
    std::cout << "\n================ Correctness Validation ================\n"
              << "Reference File   : " << ref_file << "\n"
              << "Max Abs Error    : " << std::scientific << std::setprecision(4) << max_abs_err << "\n"
              << "Centroid RMSE    : " << std::scientific << std::setprecision(4) << rmse << "\n";
    if (final_sse > 0.0) {
        std::cout << "Final SSE (WCSS) : " << std::scientific << std::setprecision(6) << final_sse << "\n";
    }
    if (ref_sse > 0.0 && final_sse > 0.0) {
        std::cout << "Ref SSE (Serial) : " << std::scientific << std::setprecision(6) << ref_sse << "\n"
                  << "Rel SSE Diff     : " << std::scientific << std::setprecision(4) << rel_sse_err << "\n";
    }
    std::cout << "Validation Status: " << (passed ? "PASSED (Matches Serial Baseline)" : "WARNING (Exceeds tolerance)") << "\n"
              << "========================================================\n";
    return passed;
}

#endif // KMEANS_COMMON_HPP
