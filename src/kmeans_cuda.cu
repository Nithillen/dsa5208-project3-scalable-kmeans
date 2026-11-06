/**
 * Hybrid MPI + CUDA K-Means Clustering (src/kmeans_cuda.cu)
 * Target Platform: NUS Vanda Supercomputer (NVIDIA GPUs with CUDA-aware OpenMPI)
 *
 * Key Parallel GPU Optimizations & Safety Features:
 * 1. Shared-Memory Centroid Caching (__shared__ double s_centroids[]) with explicit
 *    hardware shared-memory capacity check against cudaDevAttrMaxSharedMemoryPerBlock.
 * 2. Two-Stage Block Reduction for feature sums, cluster counts, and SSE/WCSS objective.
 * 3. Deterministic initialization matching Serial/MPI/OpenMP in both CSV and Synthetic modes.
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include "kmeans_common.hpp"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// CUDA kernel using shared-memory cached centroids and block-level partial reduction (including SSE)
__global__ void kmeans_assign_accumulate_kernel(const double* __restrict__ d_data,
                                                const double* __restrict__ d_centroids,
                                                double* __restrict__ d_sums,
                                                unsigned long long* __restrict__ d_counts,
                                                double* __restrict__ d_sse,
                                                size_t local_N, size_t d, size_t K,
                                                bool use_shared_mem) {
    extern __shared__ double shared_mem[];
    int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

    double* s_centroids = nullptr;
    double* s_sums      = nullptr;
    unsigned long long* s_counts = nullptr;
    __shared__ double s_block_sse;

    if (tid == 0) {
        s_block_sse = 0.0;
    }

    if (use_shared_mem) {
        s_centroids = shared_mem;
        s_sums      = &shared_mem[K * d];
        s_counts    = reinterpret_cast<unsigned long long*>(&shared_mem[2 * K * d]);

        for (size_t i = tid; i < K * d; i += blockDim.x) {
            s_centroids[i] = d_centroids[i];
            s_sums[i] = 0.0;
        }
        for (size_t k = tid; k < K; k += blockDim.x) {
            s_counts[k] = 0ULL;
        }
    }
    __syncthreads();

    if (idx < local_N) {
        const double* pt = &d_data[idx * d];
        size_t best_k = 0;
        double best_dist = DBL_MAX;
        const double* cent_src = use_shared_mem ? s_centroids : d_centroids;

        for (size_t k = 0; k < K; ++k) {
            double dist = 0.0;
            const double* c = &cent_src[k * d];
            for (size_t j = 0; j < d; ++j) {
                double diff = pt[j] - c[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        atomicAdd(&s_block_sse, best_dist);

        if (use_shared_mem) {
            atomicAdd(&s_counts[best_k], 1ULL);
            for (size_t j = 0; j < d; ++j) {
                atomicAdd(&s_sums[best_k * d + j], pt[j]);
            }
        } else {
            atomicAdd(&d_counts[best_k], 1ULL);
            for (size_t j = 0; j < d; ++j) {
                atomicAdd(&d_sums[best_k * d + j], pt[j]);
            }
        }
    }
    __syncthreads();

    if (tid == 0 && s_block_sse > 0.0) {
        atomicAdd(d_sse, s_block_sse);
    }

    if (use_shared_mem) {
        for (size_t i = tid; i < K * d; i += blockDim.x) {
            if (s_sums[i] != 0.0) {
                atomicAdd(&d_sums[i], s_sums[i]);
            }
        }
        for (size_t k = tid; k < K; k += blockDim.x) {
            if (s_counts[k] > 0ULL) {
                atomicAdd(&d_counts[k], s_counts[k]);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank = 0, num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string input_file  = "data/higgs_clean.csv";
    std::string ref_file    = "results/centroids_serial.txt";
    std::string output_file = "results/centroids_cuda.txt";
    size_t N = 597638, d = 28, K = 8, max_iters = 50;
    size_t max_rows_override = 0;
    double tolerance = 1e-5;
    unsigned int seed = 42;
    int block_size = 256;
    bool use_synthetic = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) { input_file = argv[++i]; use_synthetic = false; }
        else if (arg == "-s") { use_synthetic = true; }
        else if (arg == "-n" && i + 1 < argc) { N = std::stoull(argv[++i]); max_rows_override = N; }
        else if (arg == "-d" && i + 1 < argc) d = std::stoull(argv[++i]);
        else if (arg == "-r" && i + 1 < argc) ref_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "-k" && i + 1 < argc) K = std::stoull(argv[++i]);
        else if (arg == "-m" && i + 1 < argc) max_iters = std::stoull(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) tolerance = std::stod(argv[++i]);
        else if (arg == "-b" && i + 1 < argc) block_size = std::stoi(argv[++i]);
    }

    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int dev_id = 0;
    if (num_devices > 0) {
        dev_id = rank % num_devices;
        CUDA_CHECK(cudaSetDevice(dev_id));
    }

    std::vector<double> full_data, local_data, centroids;
    size_t local_N = 0;

    if (!use_synthetic) {
        if (rank == 0) {
            if (!load_csv_data(input_file, full_data, N, d, max_rows_override)) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            initialize_centroids(full_data, centroids, N, K, d);
        }

        unsigned long long meta[2] = { static_cast<unsigned long long>(N), static_cast<unsigned long long>(d) };
        MPI_Bcast(meta, 2, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        N = static_cast<size_t>(meta[0]);
        d = static_cast<size_t>(meta[1]);

        if (rank != 0) centroids.resize(K * d);
        MPI_Bcast(centroids.data(), static_cast<int>(K * d), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<int> sendcounts(num_procs, 0), displs(num_procs, 0);
        size_t base_rows = N / num_procs, rem_rows = N % num_procs;
        int offset = 0;
        for (int p = 0; p < num_procs; ++p) {
            size_t rows_p = base_rows + (static_cast<size_t>(p) < rem_rows ? 1 : 0);
            sendcounts[p] = static_cast<int>(rows_p * d);
            displs[p] = offset;
            offset += sendcounts[p];
        }

        local_N = static_cast<size_t>(sendcounts[rank]) / d;
        local_data.resize(local_N * d);
        MPI_Scatterv(rank == 0 ? full_data.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                     local_data.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        size_t base_rows = N / static_cast<size_t>(num_procs);
        size_t rem_rows  = N % static_cast<size_t>(num_procs);
        size_t global_start = 0;
        for (int p = 0; p < rank; ++p) {
            global_start += base_rows + (static_cast<size_t>(p) < rem_rows ? 1 : 0);
        }
        local_N = base_rows + (static_cast<size_t>(rank) < rem_rows ? 1 : 0);
        generate_synthetic_partition_and_centroids(local_data, centroids, N, global_start, local_N, d, K, seed);
    }

    // Check GPU shared memory capacity before launching kernel
    int max_shared_bytes = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlock, dev_id));
    size_t required_shared_bytes = 2 * K * d * sizeof(double) + K * sizeof(unsigned long long);
    bool use_shared_mem = (required_shared_bytes <= static_cast<size_t>(max_shared_bytes));
    size_t kernel_shared_bytes = use_shared_mem ? required_shared_bytes : 0;

    // Allocate GPU device memory once outside the iterative loop
    double *d_data = nullptr, *d_centroids = nullptr, *d_sums = nullptr, *d_sse = nullptr;
    unsigned long long *d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, local_N * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_centroids, K * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sums, K * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_counts, K * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sse, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_data, local_data.data(), local_N * d * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> local_sums(K * d, 0.0), global_sums(K * d, 0.0);
    std::vector<unsigned long long> local_counts(K, 0), global_counts(K, 0);

    int grid_size = static_cast<int>((local_N + block_size - 1) / block_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    double total_gpu_time = 0.0, total_comm_time = 0.0;
    size_t iter = 0;
    double local_sse = 0.0, global_sse = 0.0;

    for (iter = 0; iter < max_iters; ++iter) {
        double t0 = MPI_Wtime();
        CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), K * d * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sums, 0, K * d * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_counts, 0, K * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_sse, 0, sizeof(double)));

        kmeans_assign_accumulate_kernel<<<grid_size, block_size, kernel_shared_bytes>>>(
            d_data, d_centroids, d_sums, d_counts, d_sse, local_N, d, K, use_shared_mem
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(local_sums.data(), d_sums, K * d * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_counts.data(), d_counts, K * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&local_sse, d_sse, sizeof(double), cudaMemcpyDeviceToHost));
        double t1 = MPI_Wtime();

        MPI_Allreduce(local_sums.data(), global_sums.data(), static_cast<int>(K * d), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), static_cast<int>(K), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double t2 = MPI_Wtime();

        double max_movement = 0.0;
        for (size_t k = 0; k < K; ++k) {
            if (global_counts[k] > 0) {
                double movement = 0.0;
                for (size_t j = 0; j < d; ++j) {
                    double updated = global_sums[k * d + j] / static_cast<double>(global_counts[k]);
                    double diff = updated - centroids[k * d + j];
                    movement += diff * diff;
                    centroids[k * d + j] = updated;
                }
                movement = std::sqrt(movement);
                if (movement > max_movement) max_movement = movement;
            }
        }
        double t3 = MPI_Wtime();
        total_gpu_time += (t1 - t0) + (t3 - t2);
        total_comm_time += (t2 - t1);

        if (max_movement < tolerance) { iter++; break; }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double total_sec = MPI_Wtime() - t_start;

    double final_local_sse = compute_local_sse(local_data, centroids, local_N, d, K);
    double final_global_sse = 0.0;
    MPI_Allreduce(&final_local_sse, &final_global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n================ Hybrid MPI+CUDA Summary ================\n"
                  << "MPI Ranks (GPUs) : " << num_procs << "\n"
                  << "Shared Mem Cache : " << (use_shared_mem ? "ENABLED" : "FALLBACK (Global Mem)") << "\n"
                  << "Total Iterations : " << iter << "\n"
                  << "Total Time       : " << std::fixed << std::setprecision(4) << total_sec << " seconds\n"
                  << "GPU Kernel+Copy  : " << std::fixed << std::setprecision(4) << total_gpu_time << " seconds\n"
                  << "MPI_Allreduce Tm : " << std::fixed << std::setprecision(4) << total_comm_time << " seconds\n"
                  << "Avg Time/Iter    : " << std::fixed << std::setprecision(2) << (total_sec * 1000.0 / iter) << " ms\n"
                  << "Throughput       : " << std::fixed << std::setprecision(2)
                  << ((static_cast<double>(N) * iter) / (total_sec * 1e6)) << " M points/sec\n"
                  << "Final SSE (WCSS) : " << std::scientific << std::setprecision(6) << final_global_sse << "\n"
                  << "=========================================================\n";
        save_centroids_with_sse(output_file, centroids, K, d, final_global_sse);
        validate_centroids_and_sse(ref_file, centroids, K, d, final_global_sse, 1e-6);
    }

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_sse);
    MPI_Finalize();
    return 0;
}
