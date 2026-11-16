/**
 * DSA5208 Project 3: Scalable Parallel K-Means
 * Standalone Pure OpenMP Shared-Memory Implementation (src/kmeans_omp_standalone.cpp)
 *
 * Key Capabilities:
 * 1. Zero MPI dependency: pure shared-memory multithreading using OpenMP 4.5+
 * 2. Compares Parallel Accumulation Strategies:
 *    - Cache-Padded Thread-Private Accumulators (zero lock contention, aligned to 64 bytes)
 *    - Shared Atomic Updates (#pragma omp atomic update)
 * 3. Compares OpenMP Loop Schedulers:
 *    - static, dynamic (chunked), and guided
 * 4. Exact Mathematical Objective: Calculates Inertia / Sum of Squared Errors (SSE)
 * 5. Deterministic thread reduction guaranteeing bit-exact reproducibility
 * 6. Automated verification against reference centroids
 */

#include "kmeans_common.h"
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cfloat>
#include <cstdlib>
#include <algorithm>

struct Config {
    std::string input_file = "";
    std::string ref_file = "results/centroids_serial.txt";
    std::string output_file = "results/centroids_omp_standalone.txt";
    std::string schedule_type = "static"; // "static", "dynamic", "guided"
    std::string accum_mode = "private";   // "private" (thread-local padded) vs "atomic"
    size_t N = 597638;       // Default points
    size_t d = 28;           // Default dimensions (28 for HIGGS)
    size_t K = 8;            // Number of clusters
    size_t max_iters = 50;   // Maximum iterations
    int num_threads = 0;     // 0 = use OMP_NUM_THREADS
    int chunk_size = 1024;   // Chunk size for dynamic/guided
    double tolerance = 1e-5; // Convergence tolerance
    unsigned int seed = 42;  // Deterministic seed
    bool use_synthetic = true;
};

// High-speed CSV reader using buffered stream and pointer parsing
bool load_csv_data(const std::string& filename, std::vector<double>& data,
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
            while (*ptr == ',' || *ptr == ' ' || *ptr == '\t') ++ptr;
        }

        if (valid_row && !row.empty()) {
            if (d == 0) {
                d = row.size();
                data.reserve((max_rows > 0 ? max_rows : 600000ULL) * d);
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

// Generate synthetic Gaussian clustered data
void generate_synthetic_data(std::vector<double>& data, size_t N, size_t d, size_t K, unsigned int seed) {
    data.resize(N * d);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> center_dist(-50.0, 50.0);
    std::normal_distribution<double> noise_dist(0.0, 3.0);

    std::vector<double> true_centers(K * d);
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < d; ++j) {
            true_centers[k * d + j] = center_dist(gen);
        }
    }

    std::uniform_int_distribution<size_t> cluster_choice(0, K - 1);
    for (size_t i = 0; i < N; ++i) {
        size_t c = cluster_choice(gen);
        for (size_t j = 0; j < d; ++j) {
            data[i * d + j] = true_centers[c * d + j] + noise_dist(gen);
        }
    }
}

int main(int argc, char* argv[]) {
    Config cfg;
    size_t max_rows_override = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) { cfg.N = std::stoull(argv[++i]); max_rows_override = cfg.N; }
        else if (arg == "-d" && i + 1 < argc) cfg.d = std::stoull(argv[++i]);
        else if (arg == "-k" && i + 1 < argc) cfg.K = std::stoull(argv[++i]);
        else if (arg == "-m" && i + 1 < argc) cfg.max_iters = std::stoull(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) cfg.tolerance = std::stod(argv[++i]);
        else if (arg == "-f" && i + 1 < argc) { cfg.input_file = argv[++i]; cfg.use_synthetic = false; }
        else if (arg == "-r" && i + 1 < argc) cfg.ref_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) cfg.output_file = argv[++i];
        else if (arg == "-threads" && i + 1 < argc) cfg.num_threads = std::stoi(argv[++i]);
        else if (arg == "-s" && i + 1 < argc) cfg.schedule_type = argv[++i];
        else if (arg == "-c" && i + 1 < argc) cfg.chunk_size = std::stoi(argv[++i]);
        else if (arg == "-a" && i + 1 < argc) cfg.accum_mode = argv[++i];
    }

    if (cfg.num_threads > 0) {
        omp_set_num_threads(cfg.num_threads);
    }
    int active_threads = omp_get_max_threads();

    // Configure OpenMP runtime schedule
    if (cfg.schedule_type == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, cfg.chunk_size);
    } else if (cfg.schedule_type == "guided") {
        omp_set_schedule(omp_sched_guided, cfg.chunk_size);
    } else {
        omp_set_schedule(omp_sched_static, 0);
    }

    std::vector<double> data;
    if (cfg.use_synthetic) {
        std::cout << "====================================================================\n"
                  << "  Standalone OpenMP K-Means - Synthetic Clustered Data              \n"
                  << "====================================================================\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        generate_synthetic_data(data, cfg.N, cfg.d, cfg.K, cfg.seed);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Generated " << cfg.N << " x " << cfg.d << " points in "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms.\n";
    } else {
        std::cout << "====================================================================\n"
                  << "  Standalone OpenMP K-Means - UCI HIGGS Dataset                     \n"
                  << "====================================================================\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!load_csv_data(cfg.input_file, data, cfg.N, cfg.d, max_rows_override)) {
            return 1;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "CSV load time: " << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double>(t1 - t0).count() << " seconds.\n";
    }

    std::cout << "Configuration Details:\n"
              << "  Total Points (N)  : " << cfg.N << "\n"
              << "  Features (d)      : " << cfg.d << "\n"
              << "  Clusters (K)      : " << cfg.K << "\n"
              << "  OpenMP Threads    : " << active_threads << "\n"
              << "  Accumulation Mode : " << cfg.accum_mode << "\n"
              << "  Loop Schedule     : " << cfg.schedule_type << "\n"
              << "  Max Iterations    : " << cfg.max_iters << "\n"
              << "  Convergence Tol   : " << cfg.tolerance << "\n\n";

    // Initialize centroids deterministically
    std::vector<double> centroids(cfg.K * cfg.d);
    kmeans::initialize_centroids_deterministic(data.data(), centroids.data(), cfg.N, cfg.K, cfg.d);

    const size_t K = cfg.K;
    const size_t d = cfg.d;
    const size_t N = cfg.N;

    std::vector<double> new_centroids(K * d, 0.0);
    std::vector<unsigned long long> cluster_counts(K, 0ULL);

    // Cache-padded thread-private accumulator strides (multiples of 64 bytes)
    const size_t stride_sums = kmeans::get_padded_stride(K * d, 8);
    const size_t stride_cnts = kmeans::get_padded_stride(K, 8);

    std::vector<double> all_thread_sums(active_threads * stride_sums, 0.0);
    std::vector<unsigned long long> all_thread_counts(active_threads * stride_cnts, 0ULL);

    auto t_start = std::chrono::high_resolution_clock::now();
    size_t iter = 0;
    double max_movement = 0.0;
    double final_sse = 0.0;

    for (iter = 0; iter < cfg.max_iters; ++iter) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        std::fill(new_centroids.begin(), new_centroids.end(), 0.0);
        std::fill(cluster_counts.begin(), cluster_counts.end(), 0ULL);

        if (cfg.accum_mode == "atomic") {
            // Strategy B: Shared accumulators with OpenMP atomic operations (lock contention)
            #pragma omp parallel for schedule(runtime)
            for (size_t i = 0; i < N; ++i) {
                const double* pt = &data[i * d];
                size_t best_cluster = 0;
                double best_dist = DBL_MAX;

                for (size_t k = 0; k < K; ++k) {
                    double dist = kmeans::squared_euclidean_distance(pt, &centroids[k * d], d);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cluster = k;
                    }
                }

                #pragma omp atomic update
                cluster_counts[best_cluster]++;
                for (size_t j = 0; j < d; ++j) {
                    #pragma omp atomic update
                    new_centroids[best_cluster * d + j] += pt[j];
                }
            }
        } else {
            // Strategy A (Default): Cache-padded Thread-Private Accumulators (zero lock contention)
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double* t_sums = &all_thread_sums[tid * stride_sums];
                unsigned long long* t_cnts = &all_thread_counts[tid * stride_cnts];

                std::fill(t_sums, t_sums + K * d, 0.0);
                std::fill(t_cnts, t_cnts + K, 0ULL);

                #pragma omp for schedule(runtime) nowait
                for (size_t i = 0; i < N; ++i) {
                    const double* pt = &data[i * d];
                    size_t best_cluster = 0;
                    double best_dist = DBL_MAX;

                    for (size_t k = 0; k < K; ++k) {
                        double dist = kmeans::squared_euclidean_distance(pt, &centroids[k * d], d);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_cluster = k;
                        }
                    }

                    t_cnts[best_cluster]++;
                    double* sum_ptr = &t_sums[best_cluster * d];
                    for (size_t j = 0; j < d; ++j) {
                        sum_ptr[j] += pt[j];
                    }
                }
            }

            // Deterministic thread reduction
            for (int t = 0; t < active_threads; ++t) {
                const double* t_sums = &all_thread_sums[t * stride_sums];
                const unsigned long long* t_cnts = &all_thread_counts[t * stride_cnts];
                for (size_t k = 0; k < K; ++k) {
                    cluster_counts[k] += t_cnts[k];
                    for (size_t j = 0; j < d; ++j) {
                        new_centroids[k * d + j] += t_sums[k * d + j];
                    }
                }
            }
        }

        // Maximization step: update centroids and calculate maximum centroid displacement
        max_movement = 0.0;
        for (size_t k = 0; k < K; ++k) {
            unsigned long long count = cluster_counts[k];
            if (count > 0) {
                double movement = 0.0;
                for (size_t j = 0; j < d; ++j) {
                    double updated_val = new_centroids[k * d + j] / static_cast<double>(count);
                    double diff = updated_val - centroids[k * d + j];
                    movement += diff * diff;
                    centroids[k * d + j] = updated_val;
                }
                movement = std::sqrt(movement);
                if (movement > max_movement) max_movement = movement;
            }
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();

        if (iter < 3 || (iter + 1) % 10 == 0 || max_movement < cfg.tolerance) {
            std::cout << "Iteration " << std::setw(2) << (iter + 1)
                      << " | Max Centroid Shift: " << std::scientific << std::setprecision(4) << max_movement
                      << " | Time: " << std::fixed << std::setprecision(2) << iter_ms << " ms\n";
        }

        if (max_movement < cfg.tolerance) {
            std::cout << "Converged after " << (iter + 1) << " iterations!\n";
            iter++;
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    // Compute exact objective: Sum of Squared Errors (Inertia)
    final_sse = kmeans::compute_inertia_sse(data.data(), centroids.data(), N, K, d);

    std::cout << "\n================ OpenMP Benchmark Summary ================\n"
              << "OpenMP Threads   : " << active_threads << "\n"
              << "Accumulation     : " << cfg.accum_mode << "\n"
              << "Schedule Policy  : " << cfg.schedule_type << "\n"
              << "Total Iterations : " << iter << "\n"
              << "Total Time       : " << std::fixed << std::setprecision(4) << total_sec << " seconds\n"
              << "Avg Time/Iter    : " << std::fixed << std::setprecision(2) << (total_sec * 1000.0 / iter) << " ms\n"
              << "Throughput       : " << std::fixed << std::setprecision(2) << ((N * iter) / (total_sec * 1e6)) << " M points/sec\n"
              << "Final SSE (WCSS) : " << std::scientific << std::setprecision(6) << final_sse << "\n"
              << "==========================================================\n";

    // Save centroids with metadata comments
    kmeans::save_centroids_with_provenance(cfg.output_file, centroids.data(), K, d, iter, total_sec, final_sse);
    std::cout << "Centroids saved to " << cfg.output_file << "\n";

    // Automated verification against serial reference centroids if file exists
    std::vector<double> ref_centroids;
    size_t ref_K = 0, ref_d = 0;
    if (kmeans::load_centroids_file(cfg.ref_file, ref_centroids, ref_K, ref_d)) {
        if (ref_K == K && ref_d == d) {
            auto val_res = kmeans::compare_centroids(centroids.data(), ref_centroids.data(), K, d, 1e-9);
            std::cout << "\n[Validation Result] " << val_res.summary << "\n";
        }
    }

    return 0;
}
