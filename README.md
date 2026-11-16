# DSA5208 Project 3: Scalable Hybrid Parallel K-Means Clustering (`MPI + OpenMP + CUDA + Metal`)

**Course:** DSA5208 — Scalable Distributed Computing for Data Science (NUS)  
**Authors:** Nithillen Jayaseelan (`A0351473E`), Ashwin Sasikumar (`A0354333J`), Amitej Singh Datta (`A0352547Y`)  
**Platforms:** NUS Vanda Supercomputer (`CN-111`: Dual-Socket 72-Core Xeon Platinum 8452Y; `GN-A40-044`: NVIDIA A40 48 GB GPU) & Apple M5 (Local Verification / Metal Shader)  
**Algorithm:** Lloyd's K-Means Clustering (Iterative Expectation–Maximization)  
**Dataset:** Full [UCI Machine Learning Repository — HIGGS Boson Dataset (ID #280)](https://archive.ics.uci.edu/dataset/280/higgs) ($N = 11,000,000$ collision events $\times$ $d = 28$ continuous physics features $= 308,000,000$ IEEE-754 doubles, $2.464\text{ GB}$ in RAM) + Multi-Million Point Synthetic Gaussian Generator for Weak Scaling up to $12,800,000$ Points.

### Key Project Documents & Artifacts in This Repository
- **[`DSA5208_Project3_Report.pdf`](DSA5208_Project3_Report.pdf)** (source: [`report/DSA5208_Project3_Report.typ`](report/DSA5208_Project3_Report.typ)): Complete publication-grade academic PDF report with mathematical formulations, all 7 log-log scaling figures, 3 verbatim runtime terminal screenshots, validation proofs, HIGGS physics analysis on all 11,000,000 rows, and AI usage declaration.
- **[`kmeans_cpu_benchmarks.o1563432`](kmeans_cpu_benchmarks.o1563432)** & **[`kmeans_gpu_benchmark.o1563433`](kmeans_gpu_benchmark.o1563433)**: Raw, unmodified NUS Vanda PBS job outputs covering the complete 11,000,000-row CPU and NVIDIA A40 benchmark suites.
- **[`STEP_BY_STEP_WORKFLOW_README.md`](STEP_BY_STEP_WORKFLOW_README.md)**: Complete command log covering UCI dataset streaming, binary caching, compilation, local testing, Vanda PBS execution, and figure/PDF generation.
- **[`PROJECT_EXPLANATION_AND_RESULTS.txt`](PROJECT_EXPLANATION_AND_RESULTS.txt)**: Plain-text technical summary of all implementations, hardware platforms, and measured runtime comparisons.

---

## 1. Why This Algorithm & Dataset Were Chosen

1. **Iterative Global Synchronization (`MPI_Allreduce`):** Unlike embarrassingly parallel tasks, Lloyd's K-Means requires global synchronization across all ranks at the end of **every** iteration ($K \times d = 224$ sums, $K = 8$ counts, scalar SSE) before the next assignment pass can begin.
2. **Computation vs. Communication & NUMA Effects:** With $\mathcal{O}(N \cdot K \cdot d / P)$ computation per rank and $\mathcal{O}(K \cdot d \cdot \log P)$ collective latency, scaling from $P = 1$ to $72$ CPU cores exposes superlinear NUMA cache benefits ($P=2 \to 8$), Amdahl's Law barrier saturation ($P=16 \to 64$), and socket-aligned Hybrid `MPI + OpenMP` topology trade-offs (`1x72` vs. `2x36` vs. `72x1`).
3. **Shared-Memory & GPU Write Contention:** Comparing 64-byte cache-padded thread-private OpenMP accumulators against `#pragma omp atomic`, and comparing NVIDIA CUDA (`__shared__` centroid caching), OpenMP target offload (`-mp=gpu`), and Apple Metal shaders yields massive measurable speedups ($13.52\times$ OpenMP design speedup, $9.67\times$ A40 CUDA vs. serial, $50.65\times$ 64-rank MPI speedup).
4. **Full 11,000,000-Row Real Physics Workload:** Every primary CPU, MPI, Hybrid, and NVIDIA A40 GPU experiment runs on **all 11,000,000 collision events** (`0` skipped rows) from the UCI HIGGS dataset (`2.62 GB` compressed, `7.48 GB` uncompressed).

---

## 2. Repository Structure

```text
project3_scalable/
├── .gitignore                             # Tracks raw Vanda .o* logs & PDFs; excludes >100 MB data & .bin caches
├── Makefile                               # Builds Serial, MPI, Standalone/Hybrid OpenMP, Metal GPU, and CUDA targets
├── DSA5208_Project3_Report.pdf            # Compiled 13-page academic PDF report
├── README.md                              # Project documentation & verified 11M benchmark summary
├── PROJECT_EXPLANATION_AND_RESULTS.txt    # Plain-text summary & Vanda workflow guide
├── STEP_BY_STEP_WORKFLOW_README.md        # Step-by-step reproducible command & result log
├── kmeans_cpu_benchmarks.o1563432         # Verbatim Vanda CN-111 (72-core Xeon) 11M CPU log
├── kmeans_gpu_benchmark.o1563433          # Verbatim Vanda GN-A40-044 (NVIDIA A40) 11M GPU log
├── results/
│   ├── centroids_serial.txt               # Converged 8 x 28 11M reference centroids
│   ├── centroids_mpi.txt                  # Converged 8 x 28 Pure MPI centroids
│   ├── centroids_omp.txt                  # Converged 8 x 28 Hybrid MPI+OpenMP centroids
│   ├── centroids_omp_gpu.txt              # Converged 8 x 28 OpenMP -mp=gpu centroids
│   ├── centroids_cuda.txt                 # Converged 8 x 28 NVIDIA A40 CUDA centroids
│   ├── serial_scaling_N.csv               # Vanda 1-core serial scaling vs. N (100k -> 11M)
│   ├── serial_scaling_K.csv               # Vanda 1-core serial scaling vs. K (2 -> 32 on 11M)
│   ├── mpi_strong_scaling.csv             # Vanda Pure MPI strong scaling (P=1..64 on 11M)
│   ├── mpi_allreduce_comparison.csv       # Vanda separate (3-call) vs. packed (1-call) Allreduce
│   ├── mpi_weak_scaling.csv               # Vanda Pure MPI weak scaling (N=200k -> 12.8M)
│   ├── omp_design_choices.csv             # Vanda private vs. atomic & loop schedules (11M)
│   ├── hybrid_mpi_omp_grid.csv            # Vanda full 21-point (P x T) NUMA grid (1..72 cores, 11M)
│   ├── gpu_benchmark.csv                  # Vanda NVIDIA A40 CUDA vs. Xeon CPU (K={8,16,32}, 11M)
│   └── gpu_benchmark_apple_metal.csv      # Local Apple M5 Metal GPU vs. CPU (K={8,16,32}, 597,638)
├── report/
│   ├── DSA5208_Project3_Report.typ        # Typst report source
│   └── figures/                           # 7 Matplotlib log-log figures + 3 terminal screenshots
├── scripts/
│   ├── prepare_higgs_data.py              # Streaming UCI HIGGS extractor (--all for all 11M rows)
│   ├── test_correctness.py                # 16-test automated correctness & SSE parity suite
│   ├── run_serial_scaling.py              # Step 1 benchmark runner
│   ├── run_mpi_scaling.py                 # Step 2 benchmark runner
│   ├── run_hybrid_and_gpu_benchmarks.py   # Step 3 & 4 benchmark runner (--cpu-only / --gpu-only)
│   ├── generate_report_figures.py         # Generates fig1..fig7 from results/*.csv
│   ├── vanda_cpu_benchmarks.pbs           # Vanda 72-core CPU PBS job script (11M rows)
│   ├── vanda_gpu_benchmarks.pbs           # Vanda NVIDIA A40 GPU PBS job script (11M rows)
│   └── vanda_mpi_256cores.pbs             # Optional multi-node Vanda 256-core Pure MPI script
└── src/
    ├── kmeans_common.hpp                  # Fast strtod CSV loader, atomic .bin cache, SSE validator
    ├── kmeans_serial.cpp                  # Serial baseline + 11M HIGGS physics analyzer
    ├── kmeans_mpi.cpp                     # Pure MPI (MPI_Scatterv / Parallel I/O + Packed Allreduce)
    ├── kmeans_omp.cpp                     # Hybrid MPI + OpenMP (private vs. atomic, -mp=gpu offload)
    ├── kmeans_cuda.cu                     # Hybrid MPI + CUDA (__shared__ centroid cache + atomicAddDouble)
    └── kmeans_metal_gpu.mm                # Hybrid MPI + Apple Metal GPU shader
```

---

## 3. Verified NUS Vanda Benchmark Results (Full $N = 11,000,000$ UCI HIGGS Dataset)

### A. 1-Node Serial Baseline & Scaling (`CN-111`, Intel Xeon Platinum 8452Y)
50-iteration reference run on all $11,000,000$ rows ($K=8, d=28$): **`49.0870 s`** (`981.74 ms/iter`, `11.20 M pts/s`, crossed `tol=1e-5` at iteration `47`, `Final SSE = 1.584590e+08`, CSV initial load `8.32 s`).

- **Scaling vs. $N$ (`results/serial_scaling_N.csv`, $K=8, 30$ Iters):** `100k` (`0.4079 s`, `7.35 M/s`) $\to$ `500k` (`1.2805 s`, `11.71 M/s`) $\to$ `1M` (`2.7431 s`, `10.94 M/s`) $\to$ `2.5M` (`6.4020 s`, `11.72 M/s`) $\to$ `5M` (`12.8151 s`, `11.70 M/s`) $\to$ **`11,000,000` (`28.0452 s`, `934.84 ms/iter`, `11.77 M pts/s`)**.
- **Scaling vs. $K$ (`results/serial_scaling_K.csv`, $N=11\text{M}, 30$ Iters):** `K=2` (`10.5133 s`, `31.39 M/s`) $\to$ `K=4` (`15.8274 s`, `20.85 M/s`) $\to$ `K=8` (`28.1074 s`, `11.74 M/s`) $\to$ `K=16` (`60.0651 s`, `5.49 M/s`) $\to$ **`K=32` (`112.5194 s`, `3750.65 ms/iter`, `2.93 M/s`)**.

### B. Pure MPI Strong & Weak Scaling (`results/mpi_strong_scaling.csv`, `results/mpi_weak_scaling.csv`)
Strong scaling on all $N = 11,000,000$ rows ($d=28, K=8, 50$ iters):

| Ranks $P$ | Points / Rank | Total Time | Comp (s) | Comm (s) | Comm % | Throughput | Speedup $S(P)$ | Efficiency $E(P)$ | Max Err / SSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11,000,000 | 31.4583 s | 31.4576 | 0.0005 | 0.0% | 17.48 M/s | 1.00x | 100.0% | `0.000e+00` / `1.584590e+08` |
| 2 | 5,500,000 | 14.4035 s | 14.2815 | 1.3279 | 9.2% | 38.19 M/s | 2.18x | 109.2% | `0.000e+00` / `1.584590e+08` |
| 4 | 2,750,000 | 7.8956 s | 7.5634 | 1.2932 | 16.4% | 69.66 M/s | 3.98x | 99.6% | `0.000e+00` / `1.584590e+08` |
| 8 | 1,375,000 | 3.9428 s | 3.8821 | 0.7755 | 19.7% | 139.49 M/s | 7.98x | 99.7% | `0.000e+00` / `1.584590e+08` |
| 16 | 687,500 | 2.4039 s | 2.2922 | 0.7380 | 30.7% | 228.80 M/s | 13.09x | 81.8% | `0.000e+00` / `1.584590e+08` |
| 32 | 343,750 | 1.3659 s | 1.2615 | 0.5703 | 41.8% | 402.68 M/s | 23.03x | 72.0% | `0.000e+00` / `1.584590e+08` |
| **64** | **171,875** | **0.6211 s** | **0.6191** | **0.1323** | **21.3%** | **885.49 M/s** | **50.65x** | **79.1%** | **`0.000e+00` / `1.584590e+08`** |

- **Weak Scaling ($N_0 = 200,000$ pts/rank, 30 iters, $P=1 \to 64$):** Scales from $200,000$ points (`0.2983 s`, `20.11 M/s`, `100.0%`) to $12,800,000$ points (`0.5491 s`, `699.36 M/s`, `54.3%` weak efficiency — a `34.8x` throughput increase).
- **Packed vs. Separate `MPI_Allreduce`:** Packing all stats into 1 collective cuts $P=8$ wait time by $2.41\times$ (`1.6457 s` $\to$ `0.6824 s`, reducing total runtime from `4.8595 s` to `3.8170 s`) and yields the fastest packed run at $P=64$ (`0.6308 s` vs. `0.6579 s`).

### C. Hybrid `MPI + OpenMP` Design & 72-Core NUMA Grid (`results/omp_design_choices.csv`, `results/hybrid_mpi_omp_grid.csv`)
- **Accumulation & Schedule ($P=1, T=4, N=11\text{M}, K=8, 50$ iters):** Cache-padded `private` (`guided`: **`11.5415 s`**, `230.83 ms/iter`, `47.65 M/s`; `dynamic`: `11.5427 s`; `static`: `14.1680 s`; `private_standalone`: `14.6160 s`) outperforms `#pragma omp atomic` (`static`: `156.0935 s`, `3.52 M/s`) by **`13.52x`**, eliminating L1/L2 cache-line ping-pong.
- **NUMA Socket Alignment ($72$ Cores):** Aligning one rank per socket (`P=2, T=36`: **`1.3244 s`**, `3.3%` comm, `22.14x` vs. `1x1`) cuts cross-socket pure OpenMP runtime (`P=1, T=72`: `2.0441 s`) by **`35.2%`**, while flat MPI (`P=72, T=1`) reaches **`0.6448 s`** (`45.48x` speedup vs. `1x1`, **`853.03 M pts/s`**, `76.13x` faster than 1-core `kmeans_serial` at `49.0870 s`).

### D. Hardware GPU Acceleration: Vanda NVIDIA A40 ($N=11\text{M}$) & Apple M5 Metal ($N=597,638$)
- **Vanda `GN-A40-044` NVIDIA A40 (30 Iters, $N=11\text{M}$, `results/gpu_benchmark.csv`):**
  - $K=8$: CUDA **`3.0461 s`** (`108.33 M/s`, **`5.70x`** vs. 1C `17.3726 s`, `1.68x` vs. 4C `5.1064 s`, `RMSE = 1.509e-12`, `SSE = 1.584590e+08`).
  - $K=16$: CUDA **`5.7795 s`** (`57.10 M/s`, `5.66x` vs. 1C `32.7197 s`, **`1.75x`** vs. 4C `10.1117 s`, `RMSE = 1.521e-12`, `SSE = 1.441739e+08`).
  - $K=32$: CUDA **`12.4475 s`** (`26.51 M/s`, `4.69x` vs. 1C `58.3357 s`, `1.29x` vs. 4C `16.0496 s`, `RMSE = 1.562e-12`, `SSE = 1.318415e+08`).
- **Vanda 50-Iter Backend Comparison ($N=11\text{M}, K=8$, `kmeans_gpu_benchmark.o1563433`):** Native CUDA (`__shared__`, **`5.0759 s`**, **`108.36 M pts/s`**, **`9.67x`** vs. `49.0870 s` serial) beats OpenMP GPU target offload `-mp=gpu` (`15.7554 s`, `34.91 M pts/s`) by **`3.10x`**.
- **Apple M5 Metal GPU (`results/gpu_benchmark_apple_metal.csv`, $N=597,638, 30$ iters):** `K=8`: `0.1189 s` (`4.76x` vs. 1C), `K=16`: `0.1444 s` (`6.57x`), `K=32`: `0.2090 s` (**`8.10x`** vs. 1C `1.6926 s`, `2.61x` vs. 4C `0.5456 s`, FP32 `RMSE <= 5.296e-04`).

---

## 4. Quickstart: Reproducing Everything

```bash
# 1. Extract all 11,000,000 HIGGS rows into data/higgs_clean.csv & data/higgs_labels.txt
python3 scripts/prepare_higgs_data.py --all

# 2. Build all targets & verify 16/16 correctness tests
make all
python3 scripts/test_correctness.py

# 3. Submit full 11M Vanda HPC CPU (72 cores) & GPU (NVIDIA A40) batch jobs
qsub scripts/vanda_cpu_benchmarks.pbs
qsub scripts/vanda_gpu_benchmarks.pbs

# 4. Regenerate all 7 figures and compile the PDF report locally
/tmp/report_venv/bin/python3 scripts/generate_report_figures.py
typst compile report/DSA5208_Project3_Report.typ DSA5208_Project3_Report.pdf
```
