/**
 * Hybrid MPI + CUDA K-Means Clustering (src/kmeans_cuda.cu)
 * Target Platform: NUS Vanda Supercomputer (NVIDIA GPUs with CUDA-aware OpenMPI)
 *
 * Key Parallel GPU Optimizations:
 * 1. Shared-Memory Centroid Caching (__shared__ double s_centroids[]):
 *    All threads in a CUDA thread block cooperatively load the K * d centroids
 *    into fast on-chip shared memory before evaluating distances, reducing global
 *    DRAM centroid reads by 256x.
 * 2. Two-Stage Block Reduction:
 *    Threads accumulate partial centroid sums in shared memory (or L2 atomicAdd)
 *    before writing once per block to global GPU memory, followed by MPI_Allreduce
 *    across multi-GPU MPI ranks.
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cfloat>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// CUDA kernel using shared-memory cached centroids and block-level partial reduction
__global__ void kmeans_assign_accumulate_kernel(const double* __restrict__ d_data,
                                                const double* __restrict__ d_centroids,
                                                double* __restrict__ d_sums,
                                                unsigned long long* __restrict__ d_counts,
                                                size_t local_N, size_t d, size_t K) {
    extern __shared__ double shared_mem[];
    double* s_centroids = shared_mem;                   // K * d doubles
    double* s_sums      = &shared_mem[K * d];           // K * d doubles for block reduction
    unsigned long long* s_counts = reinterpret_cast<unsigned long long*>(&shared_mem[2 * K * d]);

    int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

    // Cooperatively load centroids and initialize block accumulators in shared memory
    for (size_t i = tid; i < K * d; i += blockDim.x) {
        s_centroids[i] = d_centroids[i];
        s_sums[i] = 0.0;
    }
    for (size_t k = tid; k < K; k += blockDim.x) {
        s_counts[k] = 0ULL;
    }
    __syncthreads();

    if (idx < local_N) {
        const double* pt = &d_data[idx * d];
        size_t best_k = 0;
        double best_dist = DBL_MAX;

        for (size_t k = 0; k < K; ++k) {
            double dist = 0.0;
            const double* c = &s_centroids[k * d];
            for (size_t j = 0; j < d; ++j) {
                double diff = pt[j] - c[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        // Fast shared-memory atomic accumulation within the thread block
        atomicAdd(&s_counts[best_k], 1ULL);
        for (size_t j = 0; j < d; ++j) {
            atomicAdd(&s_sums[best_k * d + j], pt[j]);
        }
    }
    __syncthreads();

    // Flush block-level shared-memory accumulators to global GPU memory once per block
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

bool load_csv_data(const std::string& filename, std::vector<double>& data, size_t& N, size_t& d, size_t max_rows = 0) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) return false;
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
            if (ptr == endptr) { valid_row = false; break; }
            row.push_back(val);
            ptr = endptr;
            while (*ptr == ',' || *ptr == ' ' || *ptr == '\t') ++ptr;
        }
        if (valid_row && !row.empty()) {
            if (d == 0) {
                d = row.size();
                data.reserve((max_rows > 0 ? max_rows : 600000ULL) * d);
            } else if (row.size() != d) continue;
            data.insert(data.end(), row.begin(), row.end());
            N++;
            if (max_rows > 0 && N >= max_rows) break;
        }
    }
    return (N > 0 && d > 0);
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank = 0, num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string input_file = "data/higgs_clean.csv";
    std::string ref_file   = "results/centroids_serial.txt";
    size_t N = 597638, d = 28, K = 8, max_iters = 50;
    double tolerance = 1e-5;
    int block_size = 256;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "-r" && i + 1 < argc) ref_file = argv[++i];
        else if (arg == "-k" && i + 1 < argc) K = std::stoull(argv[++i]);
        else if (arg == "-m" && i + 1 < argc) max_iters = std::stoull(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) tolerance = std::stod(argv[++i]);
        else if (arg == "-b" && i + 1 < argc) block_size = std::stoi(argv[++i]);
    }

    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % num_devices));
    }

    std::vector<double> full_data, local_data, centroids;
    if (rank == 0) {
        if (!load_csv_data(input_file, full_data, N, d)) {
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

    size_t local_N = static_cast<size_t>(sendcounts[rank]) / d;
    local_data.resize(local_N * d);
    MPI_Scatterv(rank == 0 ? full_data.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_data.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate GPU device memory once outside the iterative loop
    double *d_data = nullptr, *d_centroids = nullptr, *d_sums = nullptr;
    unsigned long long *d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, local_N * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_centroids, K * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sums, K * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_counts, K * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_data, local_data.data(), local_N * d * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> local_sums(K * d, 0.0), global_sums(K * d, 0.0);
    std::vector<unsigned long long> local_counts(K, 0), global_counts(K, 0);

    int grid_size = static_cast<int>((local_N + block_size - 1) / block_size);
    size_t shared_bytes = 2 * K * d * sizeof(double) + K * sizeof(unsigned long long);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    double total_gpu_time = 0.0, total_comm_time = 0.0;
    size_t iter = 0;

    for (iter = 0; iter < max_iters; ++iter) {
        double t0 = MPI_Wtime();
        CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), K * d * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sums, 0, K * d * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_counts, 0, K * sizeof(unsigned long long)));

        kmeans_assign_accumulate_kernel<<<grid_size, block_size, shared_bytes>>>(
            d_data, d_centroids, d_sums, d_counts, local_N, d, K
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(local_sums.data(), d_sums, K * d * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_counts.data(), d_counts, K * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double t1 = MPI_Wtime();

        MPI_Allreduce(local_sums.data(), global_sums.data(), static_cast<int>(K * d), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), static_cast<int>(K), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
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

    if (rank == 0) {
        std::cout << "\n================ Hybrid MPI+CUDA Summary ================\n"
                  << "MPI Ranks (GPUs) : " << num_procs << "\n"
                  << "Total Iterations : " << iter << "\n"
                  << "Total Time       : " << std::fixed << std::setprecision(4) << total_sec << " seconds\n"
                  << "GPU Kernel+Copy  : " << std::fixed << std::setprecision(4) << total_gpu_time << " seconds\n"
                  << "MPI_Allreduce Tm : " << std::fixed << std::setprecision(4) << total_comm_time << " seconds\n"
                  << "Avg Time/Iter    : " << std::fixed << std::setprecision(2) << (total_sec * 1000.0 / iter) << " ms\n"
                  << "Throughput       : " << std::fixed << std::setprecision(2)
                  << ((static_cast<double>(N) * iter) / (total_sec * 1e6)) << " M points/sec\n"
                  << "=========================================================\n";
    }

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_counts);
    MPI_Finalize();
    return 0;
}
