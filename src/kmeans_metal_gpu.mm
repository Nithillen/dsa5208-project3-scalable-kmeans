/**
 * Hybrid MPI + Apple Metal GPU K-Means Clustering (src/kmeans_metal_gpu.mm)
 * Executes point-to-centroid distance evaluation on the Apple M-series GPU
 * using Threadgroup Shared Memory (equivalent to CUDA __shared__ memory)
 * and Unified Shared Memory (zero-copy CPU/GPU buffers), combined with
 * OpenMP double-precision accumulation and MPI_Allreduce synchronization.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cfloat>
#include <cstdlib>

static const char* METAL_SHADER_SRC = R"(
#include <metal_stdlib>
using namespace metal;

kernel void kmeans_assign_kernel(
    device const float* data          [[buffer(0)]],
    device const float* centroids     [[buffer(1)]],
    device uint* assignments          [[buffer(2)]],
    constant uint& local_N            [[buffer(3)]],
    constant uint& d                  [[buffer(4)]],
    constant uint& K                  [[buffer(5)]],
    uint gid                          [[thread_position_in_grid]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint tg_size                      [[threads_per_threadgroup]])
{
    // On-chip threadgroup shared memory for K*d centroids (up to K=32, d=32 -> 1024 floats)
    threadgroup float s_centroids[1024];

    uint kd = K * d;
    for (uint i = tid; i < kd; i += tg_size) {
        s_centroids[i] = centroids[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid >= local_N) return;

    device const float* pt = data + gid * d;
    uint best_k = 0;
    float best_dist = MAXFLOAT;

    for (uint k = 0; k < K; ++k) {
        float dist = 0.0f;
        threadgroup const float* c = s_centroids + k * d;
        for (uint j = 0; j < d; ++j) {
            float diff = pt[j] - c[j];
            dist = fma(diff, diff, dist);
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }
    assignments[gid] = best_k;
}
)";

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

void validate_centroids(const std::string& ref_file, const std::vector<double>& centroids, size_t K, size_t d) {
    std::ifstream in(ref_file);
    if (!in.is_open()) return;
    size_t ref_K = 0, ref_d = 0;
    if (!(in >> ref_K >> ref_d) || ref_K != K || ref_d != d) return;

    double max_abs_err = 0.0, sum_sq_err = 0.0;
    for (size_t i = 0; i < K * d; ++i) {
        double ref_val = 0.0;
        in >> ref_val;
        double err = std::abs(centroids[i] - ref_val);
        if (err > max_abs_err) max_abs_err = err;
        sum_sq_err += err * err;
    }
    double rmse = std::sqrt(sum_sq_err / (K * d));
    std::cout << "\n================ GPU Correctness Validation ================\n"
              << "Reference File   : " << ref_file << "\n"
              << "Max Abs Error    : " << std::scientific << std::setprecision(4) << max_abs_err << "\n"
              << "Centroid RMSE    : " << std::scientific << std::setprecision(4) << rmse << "\n"
              << "Validation Status: " << (rmse < 1e-3 ? "PASSED (FP32 GPU Shader Matches FP64 Serial)" : "WARNING") << "\n"
              << "============================================================\n";
}

int main(int argc, char* argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string input_file = "data/higgs_clean.csv";
    std::string ref_file   = "results/centroids_serial.txt";
    size_t N = 597638, d = 28, K = 8, max_iters = 50;
    double tolerance = 1e-5;
    int num_threads = 4;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "-r" && i + 1 < argc) ref_file = argv[++i];
        else if (arg == "-k" && i + 1 < argc) K = std::stoull(argv[++i]);
        else if (arg == "-m" && i + 1 < argc) max_iters = std::stoull(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) tolerance = std::stod(argv[++i]);
        else if (arg == "-threads" && i + 1 < argc) num_threads = std::stoi(argv[++i]);
    }
    omp_set_num_threads(num_threads);

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Error: Apple Metal GPU device not available.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:METAL_SHADER_SRC]
                                                      options:nil
                                                        error:&error];
        if (!library) {
            std::cerr << "Metal compilation error: " << [[error localizedDescription] UTF8String] << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        id<MTLFunction> kernelFunc = [library newFunctionWithName:@"kmeans_assign_kernel"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFunc error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        std::vector<double> full_data, local_data, centroids;
        if (rank == 0) {
            if (!load_csv_data(input_file, full_data, N, d)) MPI_Abort(MPI_COMM_WORLD, 1);
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

        if (rank == 0) {
            std::cout << "====================================================================\n"
                      << "  Hybrid MPI + GPU (Apple Metal Shader + OpenMP) K-Means            \n"
                      << "====================================================================\n"
                      << "  GPU Device       : " << [[device name] UTF8String] << "\n"
                      << "  MPI Ranks (P)    : " << num_procs << " | OMP Threads: " << num_threads << "\n"
                      << "  Total Points (N) : " << N << " (" << local_N << " pts/rank)\n"
                      << "  Features (d)     : " << d << " | Clusters (K): " << K << "\n\n";
        }

        // Allocate Unified Shared Memory GPU buffers (zero-copy between CPU and M-series GPU)
        id<MTLBuffer> bufData = [device newBufferWithLength:local_N * d * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufCentroids = [device newBufferWithLength:K * d * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufAssign = [device newBufferWithLength:local_N * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];

        float* gpu_data_ptr = static_cast<float*>([bufData contents]);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < local_N * d; ++i) {
            gpu_data_ptr[i] = static_cast<float>(local_data[i]);
        }

        float* gpu_cent_ptr = static_cast<float*>([bufCentroids contents]);
        uint32_t* gpu_assign_ptr = static_cast<uint32_t*>([bufAssign contents]);

        uint32_t u_local_N = static_cast<uint32_t>(local_N);
        uint32_t u_d       = static_cast<uint32_t>(d);
        uint32_t u_K       = static_cast<uint32_t>(K);

        std::vector<double> local_sums(K * d, 0.0), global_sums(K * d, 0.0);
        std::vector<unsigned long long> local_counts(K, 0), global_counts(K, 0);

        const size_t stride_sums = ((K * d + 7) / 8) * 8;
        const size_t stride_cnts = ((K + 7) / 8) * 8;
        std::vector<double> all_thread_sums(num_threads * stride_sums, 0.0);
        std::vector<unsigned long long> all_thread_counts(num_threads * stride_cnts, 0ULL);

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();
        double total_gpu_kernel_time = 0.0, total_accum_time = 0.0, total_comm_time = 0.0;
        size_t iter = 0;

        for (iter = 0; iter < max_iters; ++iter) {
            double t0 = MPI_Wtime();
            for (size_t i = 0; i < K * d; ++i) {
                gpu_cent_ptr[i] = static_cast<float>(centroids[i]);
            }

            // 1. Dispatch GPU compute kernel across local_N points
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:bufData offset:0 atIndex:0];
            [encoder setBuffer:bufCentroids offset:0 atIndex:1];
            [encoder setBuffer:bufAssign offset:0 atIndex:2];
            [encoder setBytes:&u_local_N length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&u_d length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&u_K length:sizeof(uint32_t) atIndex:5];

            NSUInteger tgSize = 256;
            NSUInteger gridSize = ((local_N + tgSize - 1) / tgSize) * tgSize;
            [encoder dispatchThreads:MTLSizeMake(gridSize, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [encoder endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            double t1 = MPI_Wtime();

            // 2. Multi-threaded double-precision feature accumulation from Unified Memory assignments
            std::fill(local_sums.begin(), local_sums.end(), 0.0);
            std::fill(local_counts.begin(), local_counts.end(), 0ULL);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double* t_sums = &all_thread_sums[tid * stride_sums];
                unsigned long long* t_cnts = &all_thread_counts[tid * stride_cnts];
                std::fill(t_sums, t_sums + K * d, 0.0);
                std::fill(t_cnts, t_cnts + K, 0ULL);

                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < local_N; ++i) {
                    uint32_t best_k = gpu_assign_ptr[i];
                    t_cnts[best_k]++;
                    const double* pt = &local_data[i * d];
                    double* sum_ptr = &t_sums[best_k * d];
                    for (size_t j = 0; j < d; ++j) {
                        sum_ptr[j] += pt[j];
                    }
                }
            }

            for (int t = 0; t < num_threads; ++t) {
                const double* t_sums = &all_thread_sums[t * stride_sums];
                const unsigned long long* t_cnts = &all_thread_counts[t * stride_cnts];
                for (size_t k = 0; k < K; ++k) {
                    local_counts[k] += t_cnts[k];
                    for (size_t j = 0; j < d; ++j) {
                        local_sums[k * d + j] += t_sums[k * d + j];
                    }
                }
            }
            double t2 = MPI_Wtime();

            // 3. MPI Allreduce across ranks
            MPI_Allreduce(local_sums.data(), global_sums.data(), static_cast<int>(K * d), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(local_counts.data(), global_counts.data(), static_cast<int>(K), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            double t3 = MPI_Wtime();

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
            double t4 = MPI_Wtime();

            total_gpu_kernel_time += (t1 - t0);
            total_accum_time      += (t2 - t1) + (t4 - t3);
            total_comm_time       += (t3 - t2);

            if (rank == 0 && (iter < 3 || (iter + 1) % 10 == 0 || max_movement < tolerance)) {
                std::cout << "Iteration " << std::setw(2) << (iter + 1)
                          << " | Max Shift: " << std::scientific << std::setprecision(4) << max_movement
                          << " | GPU Kernel: " << std::fixed << std::setprecision(2) << ((t1 - t0) * 1000.0) << " ms"
                          << " | Host Accum: " << std::fixed << std::setprecision(2) << ((t2 - t1) * 1000.0) << " ms\n";
            }

            if (max_movement < tolerance) { iter++; break; }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double total_sec = MPI_Wtime() - t_start;

        if (rank == 0) {
            std::cout << "\n================ Hybrid MPI + GPU Summary ================\n"
                      << "GPU Hardware     : " << [[device name] UTF8String] << "\n"
                      << "Total Iterations : " << iter << "\n"
                      << "Total Time       : " << std::fixed << std::setprecision(4) << total_sec << " seconds\n"
                      << "GPU Distance Tm  : " << std::fixed << std::setprecision(4) << total_gpu_kernel_time << " seconds ("
                      << std::setprecision(1) << (100.0 * total_gpu_kernel_time / total_sec) << "%)\n"
                      << "OMP Accum Time   : " << std::fixed << std::setprecision(4) << total_accum_time << " seconds ("
                      << std::setprecision(1) << (100.0 * total_accum_time / total_sec) << "%)\n"
                      << "MPI_Allreduce Tm : " << std::fixed << std::setprecision(4) << total_comm_time << " seconds ("
                      << std::setprecision(1) << (100.0 * total_comm_time / total_sec) << "%)\n"
                      << "Avg Time/Iter    : " << std::fixed << std::setprecision(2) << (total_sec * 1000.0 / iter) << " ms\n"
                      << "Throughput       : " << std::fixed << std::setprecision(2)
                      << ((static_cast<double>(N) * iter) / (total_sec * 1e6)) << " M points/sec\n"
                      << "==========================================================\n";
            validate_centroids(ref_file, centroids, K, d);
        }
    }

    MPI_Finalize();
    return 0;
}
