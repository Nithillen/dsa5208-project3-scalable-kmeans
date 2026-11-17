# DSA5208 Project 3: Scalable Hybrid Parallel K-Means Clustering (`MPI + OpenMP + CUDA + Metal`)

**Course:** DSA5208 — Scalable Distributed Computing for Data Science (NUS)  
**Authors:** Nithillen Jayaseelan (`A0351473E`), Ashwin Sasikumar (`A0354333J`), Amitej Singh Datta (`A0352547Y`)  
**Platforms:** NUS Vanda Supercomputer (`CN-048`: Dual-Socket 72-Core Xeon Platinum 8452Y; `GN-A40-023`: NVIDIA A40 48 GB GPU) & Apple M5 (Local Verification / Metal Shader)  
**Algorithm:** Lloyd's K-Means Clustering (Iterative Expectation–Maximization)  
**Dataset:** Full [UCI Machine Learning Repository — HIGGS Boson Dataset (ID #280)](https://archive.ics.uci.edu/dataset/280/higgs) ($N = 11,000,000$ collision events $\times$ $d = 28$ continuous physics features $= 308,000,000$ IEEE-754 doubles, $2.464\text{ GB}$ in RAM) + Multi-Million Point Synthetic Gaussian Generator for Weak Scaling up to $12,800,000$ Points.

### Key Project Documents & Artifacts in This Repository
- **[`DSA5208_Project3_Report.pdf`](DSA5208_Project3_Report.pdf)** (source: [`report/DSA5208_Project3_Report.typ`](report/DSA5208_Project3_Report.typ)): Complete publication-grade academic PDF report with mathematical formulations, all 7 log-log scaling figures, 3 verbatim runtime terminal screenshots, validation proofs, HIGGS physics analysis on all 11,000,000 rows, and AI usage declaration.
- **[`kmeans_cpu_benchmarks.o1565343`](kmeans_cpu_benchmarks.o1565343)** & **[`kmeans_gpu_benchmark.o1564235`](kmeans_gpu_benchmark.o1564235)**: Raw, unmodified NUS Vanda PBS job outputs covering the complete 11,000,000-row CPU and NVIDIA A40 benchmark suites (both passing 100% with zero aborts or warnings).
- **[`STEP_BY_STEP_WORKFLOW_README.md`](STEP_BY_STEP_WORKFLOW_README.md)**: Complete command log covering UCI dataset streaming, binary caching, compilation, local testing, Vanda PBS execution, and figure/PDF generation.
- **[`PROJECT_EXPLANATION_AND_RESULTS.txt`](PROJECT_EXPLANATION_AND_RESULTS.txt)**: Plain-text technical summary of all implementations, hardware platforms, and measured runtime comparisons.

---

## 1. Why This Algorithm & Dataset Were Chosen

1. **Iterative Global Synchronization (`MPI_Allreduce`):** Unlike embarrassingly parallel tasks, Lloyd's K-Means requires global synchronization across all ranks at the end of **every** iteration ($K \times d = 224$ sums, $K = 8$ counts, scalar SSE) before the next assignment pass can begin.
2. **Computation vs. Communication & NUMA Effects:** With $\mathcal{O}(N \cdot K \cdot d / P)$ computation per rank and $\mathcal{O}(K \cdot d \cdot \log P)$ collective latency, scaling from $P = 1$ to $72$ CPU cores exposes near-linear L3 cache scaling ($P=2 \to 4$), Amdahl's Law barrier saturation ($P=8 \to 64$), and close-pinned OpenMP vs. Hybrid `MPI + OpenMP` topology trade-offs (`1x72` vs. `2x36` vs. `72x1`).
3. **Shared-Memory & GPU Write Contention:** Comparing 64-byte cache-padded thread-private OpenMP accumulators against `#pragma omp atomic`, and comparing NVIDIA CUDA (`__shared__` centroid caching), OpenMP target offload (`-mp=gpu`), and Apple Metal shaders yields massive measurable speedups ($10.01\times$ OpenMP design speedup, $9.29\times$ A40 CUDA vs. `CN-048` serial / $9.50\times$ vs. `GN-A40-023` host serial, $41.40\times$ 64-rank MPI speedup).
4. **Full 11,000,000-Row Real Physics Workload:** Every primary CPU, MPI, Hybrid, and NVIDIA A40 GPU experiment runs on **all 11,000,000 collision events** (`0` skipped rows) from the UCI HIGGS dataset (`2.62 GB` compressed, `7.48 GB` uncompressed).

---

## 2. Repository Structure

```text
project3_scalable/
├── .gitignore                             # Tracks raw Vanda .o* logs & PDFs; excludes >100 MB data, .bin, & .tar.gz
├── Makefile                               # Builds Serial, MPI, Standalone/Hybrid OpenMP, Metal GPU, and CUDA targets
├── DSA5208_Project3_Report.pdf            # Compiled 13-page academic PDF report
├── README.md                              # Project documentation & verified 11M benchmark summary
├── PROJECT_EXPLANATION_AND_RESULTS.txt    # Plain-text summary & Vanda workflow guide
├── STEP_BY_STEP_WORKFLOW_README.md        # Step-by-step reproducible command & result log
├── kmeans_cpu_benchmarks.o1565343         # Verbatim Vanda CN-048 (72-core Xeon) 11M CPU log (ALL PASSED)
├── kmeans_gpu_benchmark.o1564235          # Verbatim Vanda GN-A40-023 (NVIDIA A40) 11M GPU log (ALL PASSED)
├── results/
│   ├── centroids_serial.txt               # Serial scaling final output (32 x 28 from K=32 11M run)
│   ├── centroids_mpi.txt                  # Pure MPI weak-scaling final output (8 x 28 from 12.8M run)
│   ├── centroids_omp.txt                  # Converged 8 x 28 11M Hybrid MPI+OpenMP centroids
│   ├── centroids_metal.txt                # Converged 32 x 28 Apple M5 Metal centroids (N=597,638)
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
│   ├── vanda_full_benchmark.pbs           # Early combined Vanda CPU benchmark script
│   ├── vanda_hybrid_gpu.pbs               # Standalone Vanda hybrid GPU test script
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

### A. 1-Node Serial Baseline & Scaling (`CN-048`, Intel Xeon Platinum 8452Y)
50-iteration reference run on all $11,000,000$ rows ($K=8, d=28$): **`47.1513 s`** (`943.03 ms/iter`, `11.66 M pts/s`, crossed `tol=1e-5` at iteration `47`, `Final SSE = 1.584590e+08`, initial CSV parse `47.95 s`).

- **Scaling vs. $N$ (`results/serial_scaling_N.csv`, $K=8, 30$ Iters):** `100k` (`0.4091 s`, `7.33 M/s`) $\to$ `500k` (`1.2828 s`, `11.69 M/s`) $\to$ `1M` (`3.7381 s`, `8.03 M/s`) $\to$ `2.5M` (`8.0281 s`, `9.34 M/s`) $\to$ `5M` (`14.7397 s`, `10.18 M/s`) $\to$ **`11,000,000` (`30.0148 s`, `1000.49 ms/iter`, `10.99 M pts/s`)**.
- **Scaling vs. $K$ (`results/serial_scaling_K.csv`, $N=11\text{M}, 30$ Iters):** `K=2` (`10.5269 s`, `31.35 M/s`) $\to$ `K=4` (`15.8202 s`, `20.86 M/s`) $\to$ `K=8` (`28.1127 s`, `11.74 M/s`) $\to$ `K=16` (`60.5340 s`, `5.45 M/s`) $\to$ **`K=32` (`113.1340 s`, `3771.13 ms/iter`, `2.92 M/s`)**.

### B. Pure MPI Strong & Weak Scaling (`results/mpi_strong_scaling.csv`, `results/mpi_weak_scaling.csv`)
Strong scaling on all $N = 11,000,000$ rows ($d=28, K=8, 50$ iters):

| Ranks $P$ | Points / Rank | Total Time | Comp (s) | Comm (s) | Comm % | Throughput | Speedup $S(P)$ | Efficiency $E(P)$ | Max Err / SSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11,000,000 | 26.7964 s | 26.7955 | 0.0006 | 0.0% | 20.53 M/s | 1.00x | 100.0% | `0.000e+00` / `1.584590e+08` |
| 2 | 5,500,000 | 13.8943 s | 13.3829 | 0.6830 | 4.9% | 39.58 M/s | 1.93x | 96.4% | `0.000e+00` / `1.584590e+08` |
| 4 | 2,750,000 | 7.3715 s | 7.3041 | 0.4959 | 6.7% | 74.61 M/s | 3.64x | 90.9% | `0.000e+00` / `1.584590e+08` |
| 8 | 1,375,000 | 4.6739 s | 4.5584 | 1.5139 | 32.4% | 117.67 M/s | 5.73x | 71.7% | `0.000e+00` / `1.584590e+08` |
| 16 | 687,500 | 2.2648 s | 2.1407 | 0.6844 | 30.2% | 242.84 M/s | 11.83x | 73.9% | `0.000e+00` / `1.584590e+08` |
| 32 | 343,750 | 1.4536 s | 1.3135 | 0.6572 | 45.2% | 378.37 M/s | 18.43x | 57.6% | `0.000e+00` / `1.584590e+08` |
| **64** | **171,875** | **0.6473 s** | **0.6413** | **0.1629** | **25.2%** | **849.74 M/s** | **41.40x** | **64.7%** | **`0.000e+00` / `1.584590e+08`** |

- **Weak Scaling ($N_0 = 200,000$ pts/rank, 30 iters, $P=1 \to 64$):** Scales from $200,000$ points (`0.2692 s`, `22.28 M/s`, `100.0%`) to $12,800,000$ points (`0.4814 s`, `797.72 M/s`, `55.9%` weak efficiency — a `35.8x` throughput increase).
- **Packed vs. Separate `MPI_Allreduce`:** Packing all stats into 1 collective cuts $P=8$ wait time by $2.58\times$ (`1.5610 s` $\to$ `0.6041 s`, reducing total runtime from `4.6781 s` to `3.8375 s`) and cuts $P=32$ runtime from `1.2995 s` to `1.1664 s`.

### C. Hybrid `MPI + OpenMP` Design & 72-Core NUMA Grid (`results/omp_design_choices.csv`, `results/hybrid_mpi_omp_grid.csv`)
- **Accumulation & Schedule ($P=1, T=4, N=11\text{M}, K=8, 50$ iters):** Cache-padded `private` (`dynamic`: **`12.0817 s`**, `241.63 ms/iter`, `45.52 M/s`; `guided`: `12.1257 s`; `static`: `13.8990 s`; `private_standalone`: `18.9681 s`) outperforms `#pragma omp atomic` (`static`: `120.9895 s`, `4.55 M/s`) by **`10.01x`**, eliminating L1/L2 cache-line ping-pong.
- **Core Pinning & Flat MPI ($72$ Cores):** With `P==1` close pinning (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`), `P=1, T=72` finishes in **`1.4163 s`** (`388.33 M/s`, `0.0%` comm, faster than unpinned hybrid `P=2, T=36` at `1.8289 s` / `39.0%` barrier skew), while flat MPI (`P=72, T=1`) achieves **`0.6243 s`** (`48.76x` speedup vs. `1x1` `30.4406 s`, **`880.94 M pts/s`**, `75.53x` faster than 1-core `kmeans_serial` at `47.1513 s`).

### D. Hardware GPU Acceleration: Vanda NVIDIA A40 ($N=11\text{M}$) & Apple M5 Metal ($N=597,638$)
- **Vanda `GN-A40-023` NVIDIA A40 (30 Iters, $N=11\text{M}$, `results/gpu_benchmark.csv`):**
  - $K=8$: CUDA **`3.0485 s`** (`108.25 M/s`, **`7.12x`** vs. 1C `21.7097 s`, `2.03x` vs. 4C `6.1889 s`, `RMSE = 1.509e-12`, `SSE = 1.584590e+08`).
  - $K=16$: CUDA **`5.7699 s`** (`57.19 M/s`, `6.23x` vs. 1C `35.9229 s`, **`2.12x`** vs. 4C `12.2053 s`, `RMSE = 1.521e-12`, `SSE = 1.441739e+08`).
  - $K=32$: CUDA **`12.4459 s`** (`26.51 M/s`, `4.97x` vs. 1C `61.8688 s`, `1.37x` vs. 4C `17.0752 s`, `RMSE = 1.562e-12`, `SSE = 1.318415e+08`).
- **Vanda 50-Iter Backend Comparison ($N=11\text{M}, K=8$, `kmeans_gpu_benchmark.o1564235`):** Native CUDA (`__shared__`, **`5.0769 s`**, **`108.33 M pts/s`**, **`9.29x`** vs. `CN-048` `47.1513 s` serial, `9.50x` vs. `GN-A40-023` host `48.2479 s`, `MaxErr = 4.9776e-12`, **`PASSED`**) beats OpenMP GPU target offload `-mp=gpu` (`16.9227 s`, `32.50 M pts/s`, `MaxErr = 4.9776e-12`, **`PASSED`**) by **`3.33x`**.
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

# 4. Regenerate all 7 figures and compile the PDF report locally (requires matplotlib)
python3 scripts/generate_report_figures.py
typst compile report/DSA5208_Project3_Report.typ DSA5208_Project3_Report.pdf
```
