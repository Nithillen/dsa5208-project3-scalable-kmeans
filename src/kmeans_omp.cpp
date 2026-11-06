#include <mpi.h>
#include <omp.h>
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
#include <algorithm>

struct Config {
    std::string input_file = "";
    std::string ref_file = "results/centroids_serial.txt";
    std::string output_file = "results/centroids_omp.txt";
    std::string schedule_type = "static"; // "static", "dynamic", "guided"
    std::string accum_mode = "private";   // "private" (thread-local padded) vs "atomic"
    size_t N = 597638;
    size_t d = 28;
    size_t K = 8;
    size_t max_iters = 50;
    int num_threads = 0;     // 0 = use OMP_NUM_THREADS or default
    int chunk_size = 1024;   // Chunk size for dynamic/guided scheduling
    double tolerance = 1e-5;
    unsigned int seed = 42;
    bool use_synthetic = true;
    bool weak_scaling = false;
    bool use_gpu_target = false; // OpenMP target offload to GPU (-g)
};

bool load_csv_data(const std::string& filename, std::vector<double>& data, size_t& N, size_t& d, size_t max_rows = 0) {
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
                if (max_rows > 0) data.reserve(max_rows * d);
                else data.reserve(600000ULL * d);
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

void generate_local_synthetic_data(std::vector<double>& local_data,
                                   std::vector<double>& centroids,
                                   size_t local_N, size_t d, size_t K,
                                   unsigned int seed, int rank) {
    std::mt19937 center_gen(seed);
    std::uniform_real_distribution<double> center_dist(-50.0, 50.0);

    std::vector<double> true_centers(K * d);
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < d; ++j) {
            true_centers[k * d + j] = center_dist(center_gen);
        }
    }

    std::mt19937 pt_gen(seed + 10007U * static_cast<unsigned int>(rank + 1));
    std::normal_distribution<double> noise_dist(0.0, 3.0);
    std::uniform_int_distribution<size_t> cluster_choice(0, K - 1);

    local_data.resize(local_N * d);
    for (size_t i = 0; i < local_N; ++i) {
        size_t c = cluster_choice(pt_gen);
        for (size_t j = 0; j < d; ++j) {
            local_data[i * d + j] = true_centers[c * d + j] + noise_dist(pt_gen);
        }
    }

    centroids.resize(K * d);
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < d; ++j) {
            centroids[k * d + j] = true_centers[k * d + j] + 1.5;
        }
    }
}

inline double squared_euclidean_distance(const double* pt, const double* centroid, size_t d) {
    double dist = 0.0;
    #pragma omp simd reduction(+:dist)
    for (size_t j = 0; j < d; ++j) {
        double diff = pt[j] - centroid[j];
        dist += diff * diff;
    }
    return dist;
}

void initialize_centroids(const std::vector<double>& data, std::vector<double>& centroids, size_t N, size_t K, size_t d) {
    centroids.resize(K * d);
    size_t step = N / K;
    for (size_t k = 0; k < K; ++k) {
        size_t idx = k * step;
        for (size_t j = 0; j < d; ++j) {
            centroids[k * d + j] = data[idx * d + j];
        }
    }
}

void save_centroids(const std::string& filename, const std::vector<double>& centroids, size_t K, size_t d) {
    std::ofstream out(filename);
    if (!out.is_open()) return;
    out << std::setprecision(12);
    out << K << " " << d << "\n";
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < d; ++j) {
            out << centroids[k * d + j] << (j + 1 == d ? "" : " ");
        }
        out << "\n";
    }
}

void validate_centroids(const std::string& ref_file, const std::vector<double>& centroids, size_t K, size_t d) {
    std::ifstream in(ref_file);
    if (!in.is_open()) return;
    size_t ref_K = 0, ref_d = 0;
    if (!(in >> ref_K >> ref_d) || ref_K != K || ref_d != d) return;

    double max_abs_err = 0.0;
    double sum_sq_err = 0.0;
    for (size_t i = 0; i < K * d; ++i) {
        double ref_val = 0.0;
        in >> ref_val;
        double err = std::abs(centroids[i] - ref_val);
        if (err > max_abs_err) max_abs_err = err;
        sum_sq_err += err * err;
    }
    double rmse = std::sqrt(sum_sq_err / (K * d));

    std::cout << "\n================ Correctness Validation ================\n"
              << "Reference File   : " << ref_file << "\n"
              << "Max Abs Error    : " << std::scientific << std::setprecision(4) << max_abs_err << "\n"
              << "Centroid RMSE    : " << std::scientific << std::setprecision(4) << rmse << "\n"
              << "Validation Status: " << (max_abs_err < 1e-9 ? "PASSED (Exact Match with Serial)" : "WARNING") << "\n"
              << "========================================================\n";
}

int main(int argc, char* argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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
        else if (arg == "-w") cfg.weak_scaling = true;
        else if (arg == "-g") cfg.use_gpu_target = true;
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

    std::vector<double> local_data;
    std::vector<double> centroids;
    size_t total_N = 0;
    size_t local_N = 0;

    if (!cfg.use_synthetic) {
        std::vector<double> full_data;
        if (rank == 0) {
            if (!load_csv_data(cfg.input_file, full_data, total_N, cfg.d, max_rows_override)) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            initialize_centroids(full_data, centroids, total_N, cfg.K, cfg.d);
        }

        unsigned long long meta[2] = { static_cast<unsigned long long>(total_N),
                                       static_cast<unsigned long long>(cfg.d) };
        MPI_Bcast(meta, 2, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        total_N = static_cast<size_t>(meta[0]);
        cfg.d   = static_cast<size_t>(meta[1]);

        if (rank != 0) centroids.resize(cfg.K * cfg.d);
        MPI_Bcast(centroids.data(), static_cast<int>(cfg.K * cfg.d), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<int> sendcounts(num_procs, 0);
        std::vector<int> displs(num_procs, 0);
        size_t base_rows = total_N / num_procs;
        size_t rem_rows  = total_N % num_procs;
        int offset = 0;
        for (int p = 0; p < num_procs; ++p) {
            size_t rows_p = base_rows + (static_cast<size_t>(p) < rem_rows ? 1 : 0);
            sendcounts[p] = static_cast<int>(rows_p * cfg.d);
            displs[p] = offset;
            offset += sendcounts[p];
        }

        local_N = static_cast<size_t>(sendcounts[rank]) / cfg.d;
        local_data.resize(local_N * cfg.d);

        MPI_Scatterv(rank == 0 ? full_data.data() : nullptr,
                     sendcounts.data(), displs.data(), MPI_DOUBLE,
                     local_data.data(), sendcounts[rank], MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    } else {
        if (cfg.weak_scaling) {
            local_N = cfg.N;
            total_N = cfg.N * static_cast<size_t>(num_procs);
        } else {
            total_N = cfg.N;
            size_t base_rows = total_N / num_procs;
            size_t rem_rows  = total_N % num_procs;
            local_N = base_rows + (static_cast<size_t>(rank) < rem_rows ? 1 : 0);
        }
        generate_local_synthetic_data(local_data, centroids, local_N, cfg.d, cfg.K, cfg.seed, rank);
    }

    if (rank == 0) {
        std::cout << "====================================================================\n"
                  << "  Hybrid MPI + OpenMP K-Means (" << (cfg.use_gpu_target ? "GPU Target Offload" : "CPU Shared-Memory") << ")\n"
                  << "====================================================================\n"
                  << "  MPI Ranks (P)     : " << num_procs << "\n"
                  << "  OMP Threads/Rank  : " << active_threads << "\n"
                  << "  Total Workers     : " << (num_procs * active_threads) << "\n"
                  << "  Accumulation Mode : " << cfg.accum_mode << "\n"
                  << "  OMP Schedule      : " << cfg.schedule_type << "\n"
                  << "  Total Points (N)  : " << total_N << " (" << local_N << " pts/rank)\n"
                  << "  Features (d)      : " << cfg.d << " | Clusters (K): " << cfg.K << "\n\n";
    }

    std::vector<double> local_sums(cfg.K * cfg.d, 0.0);
    std::vector<double> global_sums(cfg.K * cfg.d, 0.0);
    std::vector<unsigned long long> local_counts(cfg.K, 0);
    std::vector<unsigned long long> global_counts(cfg.K, 0);

    // Pre-allocate cache-line padded thread-private buffers to eliminate false sharing & heap allocation inside loop
    const size_t stride_sums = ((cfg.K * cfg.d + 7) / 8) * 8; // Multiple of 64 bytes (8 doubles)
    const size_t stride_cnts = ((cfg.K + 7) / 8) * 8;         // Multiple of 64 bytes (8 ull)
    std::vector<double> all_thread_sums(active_threads * stride_sums, 0.0);
    std::vector<unsigned long long> all_thread_counts(active_threads * stride_cnts, 0ULL);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start = MPI_Wtime();
    double total_comp_time = 0.0;
    double total_comm_time = 0.0;

    size_t iter = 0;
    double max_movement = 0.0;

    const size_t K = cfg.K;
    const size_t d = cfg.d;

    for (iter = 0; iter < cfg.max_iters; ++iter) {
        double t_iter_start = MPI_Wtime();

        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_counts.begin(), local_counts.end(), 0ULL);

        if (cfg.use_gpu_target) {
            // OpenMP Target GPU Offload Mode (runs on NVIDIA GPUs when compiled with nvc++ -mp=gpu)
            const double* raw_data = local_data.data();
            const double* raw_cent = centroids.data();
            double* raw_sums = local_sums.data();
            unsigned long long* raw_cnts = local_counts.data();
            size_t kd = K * d;
            size_t nd = local_N * d;

            #pragma omp target teams distribute parallel for \
                map(to: raw_data[0:nd], raw_cent[0:kd]) \
                map(tofrom: raw_sums[0:kd], raw_cnts[0:K])
            for (size_t i = 0; i < local_N; ++i) {
                const double* pt = &raw_data[i * d];
                size_t best_cluster = 0;
                double best_dist = DBL_MAX;

                for (size_t k = 0; k < K; ++k) {
                    double dist = 0.0;
                    for (size_t j = 0; j < d; ++j) {
                        double diff = pt[j] - raw_cent[k * d + j];
                        dist += diff * diff;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cluster = k;
                    }
                }

                #pragma omp atomic update
                raw_cnts[best_cluster]++;
                for (size_t j = 0; j < d; ++j) {
                    #pragma omp atomic update
                    raw_sums[best_cluster * d + j] += pt[j];
                }
            }
        } else if (cfg.accum_mode == "atomic") {
            // Design Choice B: Shared accumulators with OpenMP atomic updates (high contention)
            #pragma omp parallel for schedule(runtime)
            for (size_t i = 0; i < local_N; ++i) {
                const double* pt = &local_data[i * d];
                size_t best_cluster = 0;
                double best_dist = DBL_MAX;

                for (size_t k = 0; k < K; ++k) {
                    double dist = squared_euclidean_distance(pt, &centroids[k * d], d);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cluster = k;
                    }
                }

                #pragma omp atomic update
                local_counts[best_cluster]++;
                double* sum_ptr = &local_sums[best_cluster * d];
                for (size_t j = 0; j < d; ++j) {
                    #pragma omp atomic update
                    sum_ptr[j] += pt[j];
                }
            }
        } else {
            // Design Choice A (Recommended): Cache-padded Thread-Private Accumulators (zero lock contention)
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double* t_sums = &all_thread_sums[tid * stride_sums];
                unsigned long long* t_cnts = &all_thread_counts[tid * stride_cnts];

                std::fill(t_sums, t_sums + K * d, 0.0);
                std::fill(t_cnts, t_cnts + K, 0ULL);

                #pragma omp for schedule(runtime) nowait
                for (size_t i = 0; i < local_N; ++i) {
                    const double* pt = &local_data[i * d];
                    size_t best_cluster = 0;
                    double best_dist = DBL_MAX;

                    for (size_t k = 0; k < K; ++k) {
                        double dist = squared_euclidean_distance(pt, &centroids[k * d], d);
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

            // Deterministic sequential reduction across thread buffers (guarantees identical FP rounding!)
            for (int t = 0; t < active_threads; ++t) {
                const double* t_sums = &all_thread_sums[t * stride_sums];
                const unsigned long long* t_cnts = &all_thread_counts[t * stride_cnts];
                for (size_t k = 0; k < K; ++k) {
                    local_counts[k] += t_cnts[k];
                    for (size_t j = 0; j < d; ++j) {
                        local_sums[k * d + j] += t_sums[k * d + j];
                    }
                }
            }
        }

        double t_comp_end = MPI_Wtime();

        // Global MPI Allreduce across nodes/ranks
        MPI_Allreduce(local_sums.data(), global_sums.data(),
                      static_cast<int>(K * d), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(),
                      static_cast<int>(K), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        double t_comm_end = MPI_Wtime();

        // Update centroids
        max_movement = 0.0;
        for (size_t k = 0; k < K; ++k) {
            unsigned long long count = global_counts[k];
            if (count > 0) {
                double movement = 0.0;
                for (size_t j = 0; j < d; ++j) {
                    double updated_val = global_sums[k * d + j] / static_cast<double>(count);
                    double diff = updated_val - centroids[k * d + j];
                    movement += diff * diff;
                    centroids[k * d + j] = updated_val;
                }
                movement = std::sqrt(movement);
                if (movement > max_movement) max_movement = movement;
            }
        }
        double t_iter_end = MPI_Wtime();

        double iter_comp = (t_comp_end - t_iter_start) + (t_iter_end - t_comm_end);
        double iter_comm = (t_comm_end - t_comp_end);
        total_comp_time += iter_comp;
        total_comm_time += iter_comm;

        if (rank == 0 && (iter < 3 || (iter + 1) % 10 == 0 || max_movement < cfg.tolerance)) {
            std::cout << "Iteration " << std::setw(2) << (iter + 1)
                      << " | Max Shift: " << std::scientific << std::setprecision(4) << max_movement
                      << " | Comp: " << std::fixed << std::setprecision(2) << (iter_comp * 1000.0) << " ms"
                      << " | Comm: " << std::fixed << std::setprecision(2) << (iter_comm * 1000.0) << " ms\n";
        }

        if (max_movement < cfg.tolerance) {
            if (rank == 0) std::cout << "Converged after " << (iter + 1) << " iterations!\n";
            iter++;
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_end = MPI_Wtime();
    double total_sec = t_total_end - t_total_start;

    double max_comp_time = 0.0, max_comm_time = 0.0;
    MPI_Reduce(&total_comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double comm_pct = (total_sec > 0.0) ? (100.0 * max_comm_time / total_sec) : 0.0;
        std::cout << "\n================ Hybrid MPI+OpenMP Summary ================\n"
                  << "MPI Ranks (P)    : " << num_procs << "\n"
                  << "OMP Threads (T)  : " << active_threads << "\n"
                  << "Total Cores (P*T): " << (num_procs * active_threads) << "\n"
                  << "Total Iterations : " << iter << "\n"
                  << "Total Time       : " << std::fixed << std::setprecision(4) << total_sec << " seconds\n"
                  << "Computation Time : " << std::fixed << std::setprecision(4) << max_comp_time << " seconds ("
                  << std::setprecision(1) << (100.0 - comm_pct) << "%)\n"
                  << "MPI_Allreduce Tm : " << std::fixed << std::setprecision(4) << max_comm_time << " seconds ("
                  << std::setprecision(1) << comm_pct << "%)\n"
                  << "Avg Time/Iter    : " << std::fixed << std::setprecision(2) << (total_sec * 1000.0 / iter) << " ms\n"
                  << "Throughput       : " << std::fixed << std::setprecision(2)
                  << ((static_cast<double>(total_N) * iter) / (total_sec * 1e6)) << " M points/sec\n"
                  << "===========================================================\n";

        save_centroids(cfg.output_file, centroids, K, d);
        if (!cfg.use_synthetic) {
            validate_centroids(cfg.ref_file, centroids, K, d);
        }
    }

    MPI_Finalize();
    return 0;
}
