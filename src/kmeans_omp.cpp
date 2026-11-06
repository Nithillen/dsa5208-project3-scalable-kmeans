#ifndef STANDALONE_OMP
#include <mpi.h>
#endif
#include <omp.h>
#include <chrono>
#include "kmeans_common.hpp"

struct Config {
    std::string input_file = "";
    std::string ref_file = "results/centroids_serial.txt";
    std::string output_file = "results/centroids_omp.txt";
    std::string schedule_type = "static"; // "static", "dynamic", "guided"
    std::string accum_mode = "private";   // "private" (thread-local padded) vs "atomic"
    std::string allreduce_mode = "separate"; // "separate" or "packed"
    std::string io_mode = "scatter";         // "scatter" or "parallel"
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

inline double squared_euclidean_distance(const double* pt, const double* centroid, size_t d) {
    double dist = 0.0;
    #pragma omp simd reduction(+:dist)
    for (size_t j = 0; j < d; ++j) {
        double diff = pt[j] - centroid[j];
        dist += diff * diff;
    }
    return dist;
}

inline double wall_time_sec() {
#ifndef STANDALONE_OMP
    return MPI_Wtime();
#else
    return omp_get_wtime();
#endif
}

int main(int argc, char* argv[]) {
    int rank = 0, num_procs = 1;
#ifndef STANDALONE_OMP
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

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
        else if (arg == "-allreduce" && i + 1 < argc) cfg.allreduce_mode = argv[++i];
        else if (arg == "-io" && i + 1 < argc) cfg.io_mode = argv[++i];
        else if (arg == "-p") cfg.allreduce_mode = "packed";
        else if (arg == "-w") cfg.weak_scaling = true;
        else if (arg == "-g") cfg.use_gpu_target = true;
    }

    if (cfg.num_threads > 0) {
        omp_set_num_threads(cfg.num_threads);
    }
    int active_threads = omp_get_max_threads();

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
#ifdef STANDALONE_OMP
        if (!load_csv_data(cfg.input_file, local_data, total_N, cfg.d, max_rows_override)) {
            return 1;
        }
        local_N = total_N;
        initialize_centroids(local_data, centroids, total_N, cfg.K, cfg.d);
#else
        if (cfg.io_mode == "parallel") {
            if (!load_csv_partition(cfg.input_file, local_data, centroids, total_N, local_N,
                                    cfg.d, cfg.K, rank, num_procs, max_rows_override)) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
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

            size_t base_rows = total_N / static_cast<size_t>(num_procs);
            size_t rem_rows  = total_N % static_cast<size_t>(num_procs);
            local_N = base_rows + (static_cast<size_t>(rank) < rem_rows ? 1 : 0);
            local_data.resize(local_N * cfg.d);

            if (total_N * cfg.d < 2000000000ULL) {
                std::vector<int> sendcounts(num_procs, 0);
                std::vector<int> displs(num_procs, 0);
                int offset = 0;
                for (int p = 0; p < num_procs; ++p) {
                    size_t rows_p = base_rows + (static_cast<size_t>(p) < rem_rows ? 1 : 0);
                    sendcounts[p] = static_cast<int>(rows_p * cfg.d);
                    displs[p] = offset;
                    offset += sendcounts[p];
                }
                MPI_Scatterv(rank == 0 ? full_data.data() : nullptr,
                             sendcounts.data(), displs.data(), MPI_DOUBLE,
                             local_data.data(), sendcounts[rank], MPI_DOUBLE,
                             0, MPI_COMM_WORLD);
            } else {
                if (rank == 0) {
                    std::copy(full_data.begin(), full_data.begin() + local_N * cfg.d, local_data.begin());
                    size_t offset_floats = local_N * cfg.d;
                    for (int p = 1; p < num_procs; ++p) {
                        size_t rows_p = base_rows + (static_cast<size_t>(p) < rem_rows ? 1 : 0);
                        size_t count_floats = rows_p * cfg.d;
                        MPI_Send(full_data.data() + offset_floats, static_cast<int>(count_floats),
                                 MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                        offset_floats += count_floats;
                    }
                } else {
                    MPI_Recv(local_data.data(), static_cast<int>(local_N * cfg.d),
                             MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
#endif
    } else {
        size_t global_start = 0;
        if (cfg.weak_scaling) {
            local_N = cfg.N;
            total_N = cfg.N * static_cast<size_t>(num_procs);
            global_start = static_cast<size_t>(rank) * cfg.N;
        } else {
            total_N = cfg.N;
            size_t base_rows = total_N / static_cast<size_t>(num_procs);
            size_t rem_rows  = total_N % static_cast<size_t>(num_procs);
            for (int p = 0; p < rank; ++p) {
                global_start += base_rows + (static_cast<size_t>(p) < rem_rows ? 1 : 0);
            }
            local_N = base_rows + (static_cast<size_t>(rank) < rem_rows ? 1 : 0);
        }
        generate_synthetic_partition_and_centroids(local_data, centroids, total_N, global_start, local_N,
                                                   cfg.d, cfg.K, cfg.seed);
    }

    if (rank == 0) {
        std::cout << "====================================================================\n"
#ifdef STANDALONE_OMP
                  << "  Standalone OpenMP K-Means (CPU Shared-Memory)\n"
#else
                  << "  Hybrid MPI + OpenMP K-Means (" << (cfg.use_gpu_target ? "GPU Target Offload" : "CPU Shared-Memory") << ")\n"
#endif
                  << "====================================================================\n"
                  << "  MPI Ranks (P)     : " << num_procs << "\n"
                  << "  OMP Threads/Rank  : " << active_threads << "\n"
                  << "  Total Workers     : " << (num_procs * active_threads) << "\n"
                  << "  Accumulation Mode : " << cfg.accum_mode << "\n"
                  << "  OMP Schedule      : " << cfg.schedule_type << "\n"
                  << "  Allreduce Mode    : " << cfg.allreduce_mode << "\n"
                  << "  Total Points (N)  : " << total_N << " (" << local_N << " pts/rank)\n"
                  << "  Features (d)      : " << cfg.d << " | Clusters (K): " << cfg.K << "\n\n";
    }

    const size_t K = cfg.K;
    const size_t d = cfg.d;
#ifndef STANDALONE_OMP
    const bool use_packed = (cfg.allreduce_mode == "packed");
#endif

    std::vector<double> local_sums(K * d, 0.0);
    std::vector<double> global_sums(K * d, 0.0);
    std::vector<unsigned long long> local_counts(K, 0);
    std::vector<unsigned long long> global_counts(K, 0);

    const size_t packed_len = K * d + K + 1;
    std::vector<double> local_packed(packed_len, 0.0);
    std::vector<double> global_packed(packed_len, 0.0);

    // Pre-allocate cache-line padded thread-private buffers to eliminate false sharing
    const size_t stride_sums = ((K * d + 7) / 8) * 8; // Multiple of 64 bytes (8 doubles)
    const size_t stride_cnts = ((K + 7) / 8) * 8;     // Multiple of 64 bytes (8 ull)
    std::vector<double> all_thread_sums(active_threads * stride_sums, 0.0);
    std::vector<unsigned long long> all_thread_counts(active_threads * stride_cnts, 0ULL);
    std::vector<double> all_thread_sse(active_threads * 8, 0.0); // Cache-line padded per-thread SSE

#ifndef STANDALONE_OMP
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    double t_total_start = wall_time_sec();
    double total_comp_time = 0.0;
    double total_comm_time = 0.0;

    size_t iter = 0;
    double max_movement = 0.0;
    double local_sse = 0.0;
    double global_sse = 0.0;

    for (iter = 0; iter < cfg.max_iters; ++iter) {
        double t_iter_start = wall_time_sec();

        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_counts.begin(), local_counts.end(), 0ULL);
        local_sse = 0.0;

        if (cfg.use_gpu_target) {
            const double* raw_data = local_data.data();
            const double* raw_cent = centroids.data();
            double* raw_sums = local_sums.data();
            unsigned long long* raw_cnts = local_counts.data();
            size_t kd = K * d;
            size_t nd = local_N * d;
            double gpu_sse = 0.0;

            #pragma omp target teams distribute parallel for reduction(+:gpu_sse) \
                map(to: raw_data[0:nd], raw_cent[0:kd]) \
                map(tofrom: raw_sums[0:kd], raw_cnts[0:K], gpu_sse)
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

                gpu_sse += best_dist;
                #pragma omp atomic update
                raw_cnts[best_cluster]++;
                for (size_t j = 0; j < d; ++j) {
                    #pragma omp atomic update
                    raw_sums[best_cluster * d + j] += pt[j];
                }
            }
            local_sse = gpu_sse;
        } else if (cfg.accum_mode == "atomic") {
            double accum_sse = 0.0;
            #pragma omp parallel for schedule(runtime) reduction(+:accum_sse)
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

                accum_sse += best_dist;
                #pragma omp atomic update
                local_counts[best_cluster]++;
                double* sum_ptr = &local_sums[best_cluster * d];
                for (size_t j = 0; j < d; ++j) {
                    #pragma omp atomic update
                    sum_ptr[j] += pt[j];
                }
            }
            local_sse = accum_sse;
        } else {
            // Cache-padded Thread-Private Accumulators (zero lock contention)
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double* t_sums = &all_thread_sums[tid * stride_sums];
                unsigned long long* t_cnts = &all_thread_counts[tid * stride_cnts];
                double t_sse = 0.0;

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

                    t_sse += best_dist;
                    t_cnts[best_cluster]++;
                    double* sum_ptr = &t_sums[best_cluster * d];
                    for (size_t j = 0; j < d; ++j) {
                        sum_ptr[j] += pt[j];
                    }
                }
                all_thread_sse[tid * 8] = t_sse;
            }

            // Deterministic sequential reduction across thread buffers
            for (int t = 0; t < active_threads; ++t) {
                local_sse += all_thread_sse[t * 8];
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

        double t_comp_end = wall_time_sec();

#ifdef STANDALONE_OMP
        global_sums = local_sums;
        global_counts = local_counts;
        global_sse = local_sse;
#else
        if (use_packed) {
            for (size_t i = 0; i < K * d; ++i) local_packed[i] = local_sums[i];
            for (size_t k = 0; k < K; ++k) local_packed[K * d + k] = static_cast<double>(local_counts[k]);
            local_packed[K * d + K] = local_sse;

            MPI_Allreduce(local_packed.data(), global_packed.data(),
                          static_cast<int>(packed_len), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            for (size_t i = 0; i < K * d; ++i) global_sums[i] = global_packed[i];
            for (size_t k = 0; k < K; ++k) global_counts[k] = static_cast<unsigned long long>(std::llround(global_packed[K * d + k]));
            global_sse = global_packed[K * d + K];
        } else {
            MPI_Allreduce(local_sums.data(), global_sums.data(),
                          static_cast<int>(K * d), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(local_counts.data(), global_counts.data(),
                          static_cast<int>(K), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
#endif
        double t_comm_end = wall_time_sec();

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
        double t_iter_end = wall_time_sec();

        double iter_comp = (t_comp_end - t_iter_start) + (t_iter_end - t_comm_end);
        double iter_comm = (t_comm_end - t_comp_end);
        total_comp_time += iter_comp;
        total_comm_time += iter_comm;

        if (rank == 0 && (iter < 3 || (iter + 1) % 10 == 0 || max_movement < cfg.tolerance)) {
            std::cout << "Iteration " << std::setw(2) << (iter + 1)
                      << " | Max Shift: " << std::scientific << std::setprecision(4) << max_movement
                      << " | SSE: " << std::scientific << std::setprecision(6) << global_sse
                      << " | Comp: " << std::fixed << std::setprecision(2) << (iter_comp * 1000.0) << " ms"
                      << " | Comm: " << std::fixed << std::setprecision(2) << (iter_comm * 1000.0) << " ms\n";
        }

        if (max_movement < cfg.tolerance) {
            if (rank == 0) std::cout << "Converged after " << (iter + 1) << " iterations!\n";
            iter++;
            break;
        }
    }

#ifndef STANDALONE_OMP
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    double t_total_end = wall_time_sec();
    double total_sec = t_total_end - t_total_start;

    double final_local_sse = compute_local_sse(local_data, centroids, local_N, d, K);
    double final_global_sse = final_local_sse;
    double max_comp_time = total_comp_time, max_comm_time = total_comm_time;

#ifndef STANDALONE_OMP
    MPI_Allreduce(&final_local_sse, &final_global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Reduce(&total_comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#endif

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
                  << "Final SSE (WCSS) : " << std::scientific << std::setprecision(6) << final_global_sse << "\n"
                  << "===========================================================\n";

        save_centroids_with_sse(cfg.output_file, centroids, K, d, final_global_sse);
        if (!cfg.weak_scaling) {
            validate_centroids_and_sse(cfg.ref_file, centroids, K, d, final_global_sse);
        }
    }

#ifndef STANDALONE_OMP
    MPI_Finalize();
#endif
    return 0;
}
