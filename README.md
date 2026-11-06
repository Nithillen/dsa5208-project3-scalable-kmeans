# DSA5208 Project 3: Scalable Hybrid Parallel K-Means Clustering (`MPI + OpenMP + GPU`)

**Course:** DSA5208 — Scalable Distributed Computing for Data Science (NUS)  
**Target Platforms:** NUS Vanda Supercomputer (72-core CPU nodes & NVIDIA GPUs), NUS Atlas8, and Apple M5 Local Development  
**Algorithm:** Lloyd's K-Means Clustering (Iterative Expectation–Maximization)  
**Dataset:** [UCI Machine Learning Repository — HIGGS Dataset (ID #280)](https://archive.ics.uci.edu/dataset/280/higgs) ($N = 597,638$ collision events $\times$ $d = 28$ kinematic features) + Multi-Million Point Synthetic Gaussian Generator for Weak Scaling up to 256 CPU Cores.

---

## 1. Why This Algorithm & Dataset Were Chosen

### A. Why Lloyd's K-Means Clustering?
The project requires an algorithm that is **not embarrassingly parallel** and reveals rich parallel scaling trade-offs across distributed memory (`MPI`), shared memory (`OpenMP`), and accelerators (`GPU / CUDA / Metal`):
1. **Iterative Global Synchronization (`MPI_Allreduce`):** Unlike embarrassingly parallel tasks (such as Monte Carlo sampling), K-Means is an iterative Expectation–Maximization (EM) algorithm. In every single iteration $t$, each worker computes distances from its local data partition to all $K$ centroids, accumulates partial feature sums $\mathbf{S}_k^{(p)} \in \mathbb{R}^d$ and counts $C_k^{(p)} \in \mathbb{Z}$, and **must synchronize globally** via `MPI_Allreduce` before iteration $t+1$ can begin.
2. **Computation-to-Communication Ratio:**
   - **Per-iteration Computation:** $\mathcal{O}\!\left(\frac{N \cdot K \cdot d}{P}\right)$ floating-point operations per rank.
   - **Per-iteration Communication:** $\mathcal{O}(K \cdot d \cdot \log P)$ message reduction complexity across $P$ MPI ranks.
   - As $P$ scales up to **256 CPU cores**, the local computation per rank shrinks by $256\times$, exposing the `MPI_Allreduce` latency floor (Amdahl's Law in strong scaling vs. Gustafson's Law in weak scaling).
3. **Shared-Memory & GPU Write Contention:** During the accumulation step, thousands of threads attempt to add $d$-dimensional vectors into the same $K$ centroid buckets simultaneously. Comparing **cache-padded thread-private accumulators** vs. **atomic updates** (`#pragma omp atomic` / CUDA `atomicAdd` / shared-memory block reductions) provides a clear, measurable design discussion for the report.
4. **Exact Deterministic Validation:** When initialized deterministically across ranks, the Serial, Pure MPI, `MPI + OpenMP`, and GPU implementations converge to the **exact same centroids** within floating-point precision ($\sim 10^{-12}$ for FP64 CPU/CUDA).

### B. Why the UCI HIGGS Boson Dataset (ID #280)?
- **Source:** [https://archive.ics.uci.edu/dataset/280/higgs](https://archive.ics.uci.edu/dataset/280/higgs) (Baldi, Sadowski, & Whiteson, *Nature Communications*, 2014).
- **Scale & Dimensionality:** Each collision event has **$d = 28$ continuous physics features**:
  - **21 low-level kinematic features:** Transverse momenta ($p_T$), pseudorapidities ($\eta$), azimuthal angles ($\phi$) of leptons and up to 4 particle jets, plus Missing Transverse Energy ($\text{MET}$).
  - **7 high-level invariant mass features:** $m_{jj}, m_{jjj}, m_{\ell\nu}, m_{j\ell\nu}, m_{bb}, m_{wbb}, m_{wwbb}$ constructed by particle physicists to distinguish **Higgs boson signal events ($y=1$, $H^0 \to W^+W^- \to b\bar{b}\ell\nu$)** from **background events ($y=0$, top-quark pair production $t\bar{t}$)**.

---

## 2. Repository Structure

```text
project3_scalable/
├── .gitignore                             # Excludes >100 MB raw CSVs & compiled binaries from Git
├── Makefile                               # Builds Serial, MPI, OpenMP, Metal GPU, and CUDA targets
├── README.md                              # Complete project documentation & benchmark results
├── bin/                                   # Compiled executables
├── data/                                  # Generated via scripts/prepare_higgs_data.py
│   ├── higgs_clean.csv                    # 597,638 x 28 cleaned HIGGS features (131 MB)
│   └── higgs_labels.txt                   # Ground-truth labels (1 = Higgs signal, 0 = background)
├── results/
│   ├── centroids_serial.txt               # Converged 8 x 28 reference centroids for validation
│   ├── centroids_mpi.txt                  # Converged 8 x 28 Pure MPI centroids
│   ├── centroids_omp.txt                  # Converged 8 x 28 Hybrid MPI+OpenMP centroids
│   ├── serial_scaling_N.csv               # 1-node serial runtime scaling vs. problem size N
│   ├── serial_scaling_K.csv               # 1-node serial runtime scaling vs. cluster count K
│   ├── mpi_strong_scaling.csv             # Pure MPI strong scaling on UCI HIGGS
│   ├── mpi_weak_scaling.csv               # Pure MPI weak scaling (200k pts/rank)
│   ├── omp_design_choices.csv             # OpenMP private vs. atomic & schedule comparisons
│   ├── hybrid_mpi_omp_grid.csv            # Hybrid MPI ranks (P) x OpenMP threads (T) trade-offs
│   └── gpu_benchmark.csv                  # Hardware GPU vs. CPU scaling across K={8, 16, 32}
├── scripts/
│   ├── prepare_higgs_data.py              # Fast HTTP streaming downloader & cleaner for UCI HIGGS
│   ├── run_serial_scaling.py              # Step 1: 1-node serial scaling benchmark suite
│   ├── run_mpi_scaling.py                 # Step 2: Pure MPI strong & weak scaling benchmark suite
│   ├── run_hybrid_and_gpu_benchmarks.py   # Step 3 & 4: Hybrid MPI+OpenMP & GPU benchmark suite
│   ├── vanda_mpi_256cores.pbs             # Vanda PBS script for 1 to 256 CPU cores (Pure MPI)
│   └── vanda_hybrid_gpu.pbs               # Vanda PBS script for 72-core Hybrid OMP & NVIDIA GPU
└── src/
    ├── kmeans_serial.cpp                  # Step 1: 1-core reference baseline + HIGGS physics analyzer
    ├── kmeans_mpi.cpp                     # Step 2: Pure MPI (MPI_Scatterv + MPI_Allreduce)
    ├── kmeans_omp.cpp                     # Step 3: Hybrid MPI + OpenMP (CPU + GPU Target Offload)
    ├── kmeans_cuda.cu                     # Step 4A: Hybrid MPI + CUDA (for Vanda NVIDIA GPUs)
    └── kmeans_metal_gpu.mm                # Step 4B: Hybrid MPI + Apple Metal GPU Shader (M5 GPU)
```

---

## 3. Step 1 Completed: 1-Node Serial Baseline & Scaling Analysis on UCI HIGGS

### A. Runtime Scaling vs. Problem Size $N$ (Fixed $K = 8, d = 28, \text{Iterations} = 30$)
Empirical scaling on the UCI HIGGS dataset (`results/serial_scaling_N.csv`) confirms exact linear $\mathcal{O}(N)$ complexity per iteration, maintaining a steady throughput of **~22.6 million points/sec**:

| Points ($N$) | Total Floats ($N \times d$) | Total Time ($30$ Iters) | Avg Time / Iter | Throughput | Scaling Factor vs. $25\text{k}$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| $25,000$ | $700,000$ | $0.0338\text{ s}$ | $1.13\text{ ms}$ | $22.18\text{ M pts/s}$ | $1.00\times$ |
| $50,000$ | $1,400,000$ | $0.0651\text{ s}$ | $2.17\text{ ms}$ | $23.05\text{ M pts/s}$ | $1.93\times$ |
| $100,000$ | $2,800,000$ | $0.1307\text{ s}$ | $4.36\text{ ms}$ | $22.95\text{ M pts/s}$ | $3.87\times$ |
| $200,000$ | $5,600,000$ | $0.2633\text{ s}$ | $8.78\text{ ms}$ | $22.79\text{ M pts/s}$ | $7.79\times$ |
| $400,000$ | $11,200,000$ | $0.5311\text{ s}$ | $17.70\text{ ms}$ | $22.59\text{ M pts/s}$ | $15.71\times$ |
| $597,638$ | $16,733,864$ | $0.7943\text{ s}$ | $26.48\text{ ms}$ | $22.57\text{ M pts/s}$ | $23.50\times$ |

### B. Runtime Scaling vs. Cluster Count $K$ (Fixed $N = 597,638, d = 28, \text{Iterations} = 30$)
| Clusters ($K$) | Distance Evals / Iter ($N \times K$) | Total Time ($30$ Iters) | Avg Time / Iter | Throughput |
| ---: | ---: | ---: | ---: | ---: |
| $2$ | $1,195,276$ | $0.2663\text{ s}$ | $8.88\text{ ms}$ | $67.34\text{ M pts/s}$ |
| $4$ | $2,390,552$ | $0.4917\text{ s}$ | $16.39\text{ ms}$ | $36.46\text{ M pts/s}$ |
| $8$ | $4,781,104$ | $0.8023\text{ s}$ | $26.74\text{ ms}$ | $22.35\text{ M pts/s}$ |
| $16$ | $9,562,208$ | $1.4189\text{ s}$ | $47.30\text{ ms}$ | $12.64\text{ M pts/s}$ |
| $32$ | $19,124,416$ | $2.6691\text{ s}$ | $88.97\text{ ms}$ | $6.72\text{ M pts/s}$ |

---

## 4. Step 2 Completed: Pure MPI Parallelization (`src/kmeans_mpi.cpp`)

### A. Strong Scaling on UCI HIGGS ($N = 597,638, d = 28, K = 8, 50\text{ Iters}$)
| MPI Ranks ($P$) | Points / Rank | Total Time (s) | Comp Time (s) | `MPI_Allreduce` Comm (s) | Comm % | Speedup $S(P)$ | Efficiency $E(P)$ | Max Centroid Error vs. Serial |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $1$ | $597,638$ | $1.3101\text{ s}$ | $1.3100\text{ s}$ | $0.0000\text{ s}$ | $0.0\%$ | $1.00\times$ | $100.0\%$ | $4.977 \times 10^{-12}$ |
| $2$ | $298,819$ | $0.6802\text{ s}$ | $0.6761\text{ s}$ | $0.0049\text{ s}$ | $0.7\%$ | $1.93\times$ | $96.3\%$ | $5.484 \times 10^{-12}$ |
| $4$ | $149,409$ | $0.4031\text{ s}$ | $0.3983\text{ s}$ | $0.0108\text{ s}$ | $2.7\%$ | $3.25\times$ | $81.3\%$ | $5.810 \times 10^{-12}$ |
| $8$ | $74,704$ | $0.4403\text{ s}$ | $0.3174\text{ s}$ | $0.1442\text{ s}$ | $32.8\%$ | $2.98\times$ | $37.2\%$ | $6.763 \times 10^{-12}$ |

### B. Weak Scaling ($N_{\text{per\_rank}} = 200,000\text{ pts/rank}, d = 28, K = 8, 30\text{ Iters}$)
| MPI Ranks ($P$) | Total Points ($N = 200\text{k} \times P$) | Total Time (s) | Comp Time (s) | Comm Time (s) | Comm % | Weak Efficiency $T(1)/T(P)$ | Aggregate Throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $1$ | $200,000$ | $0.2927\text{ s}$ | $0.2926\text{ s}$ | $0.0000\text{ s}$ | $0.0\%$ | $100.0\%$ | $20.50\text{ M pts/s}$ |
| $2$ | $400,000$ | $0.2733\text{ s}$ | $0.2702\text{ s}$ | $0.0032\text{ s}$ | $1.2\%$ | $107.1\%$ | $43.91\text{ M pts/s}$ |
| $4$ | $800,000$ | $0.3695\text{ s}$ | $0.3569\text{ s}$ | $0.0324\text{ s}$ | $8.8\%$ | $79.2\%$ | $64.94\text{ M pts/s}$ |
| $8$ | $1,600,000$ | $0.5518\text{ s}$ | $0.4662\text{ s}$ | $0.1173\text{ s}$ | $21.2\%$ | $53.0\%$ | $86.98\text{ M pts/s}$ |

---

## 5. Step 3 Completed: Hybrid `MPI + OpenMP` Parallelization (`src/kmeans_omp.cpp`)

### A. Parallel Design Decisions: Cache-Padded Thread-Private Accumulators vs. Atomic Updates
In shared-memory OpenMP, multiple threads simultaneously accumulate $28$-dimensional vectors into $K=8$ cluster centroids. We implemented and benchmarked two accumulation designs plus three OpenMP loop schedules (`results/omp_design_choices.csv` at $P=1, T=4$ on UCI HIGGS):

| Accumulation Strategy | OpenMP Schedule | Total Time ($50$ Iters) | Avg Time / Iter | Throughput | Speedup vs. Atomic | Max Error vs. Serial |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| **Cache-Padded Thread-Private (`private`)** | `static` | $0.3265\text{ s}$ | $6.53\text{ ms}$ | $91.51\text{ M pts/s}$ | **$5.65\times$** | $5.810 \times 10^{-12}$ |
| **Cache-Padded Thread-Private (`private`)** | `dynamic,1024` | $0.3054\text{ s}$ | $6.11\text{ ms}$ | $97.85\text{ M pts/s}$ | **$6.04\times$** | $5.810 \times 10^{-12}$ |
| **Cache-Padded Thread-Private (`private`)** | `guided` | **$0.2947\text{ s}$** | **$5.89\text{ ms}$** | **$101.40\text{ M pts/s}$** | **$6.26\times$** | $5.813 \times 10^{-12}$ |
| **Shared Atomic Updates (`#pragma omp atomic`)** | `static` | $1.8447\text{ s}$ | $36.89\text{ ms}$ | $16.20\text{ M pts/s}$ | $1.00\times$ | $4.982 \times 10^{-12}$ |

- **Why Thread-Private Wins by $6.26\times$:** Using `#pragma omp atomic` forces all threads to execute $597,638 \times 29 = 17.3\text{ million}$ atomic read-modify-write operations per iteration on the same $224$ memory addresses, causing severe L1/L2 cache-line invalidation ("cache-line ping-pong"). Allocating 64-byte aligned thread-private buffers (`stride_sums`) eliminates all lock contention during the loop and reduces deterministically across threads at the end of each iteration.

### B. Hybrid MPI Ranks ($P$) $\times$ OpenMP Threads ($T$) Trade-Off (`results/hybrid_mpi_omp_grid.csv`)
| Total Cores ($P \times T$) | MPI Ranks ($P$) | OMP Threads ($T$) | Total Time (s) | Comp Time (s) | `MPI_Allreduce` Comm (s) | Speedup $S(P,T)$ | Efficiency | Max Error vs. Serial |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | $1$ | $1$ | $0.9088\text{ s}$ | $0.9087\text{ s}$ | $0.0000\text{ s}$ | $1.00\times$ | $100.0\%$ | $4.98 \times 10^{-12}$ |
| **2** | $1$ | $2$ | $0.4890\text{ s}$ | $0.4889\text{ s}$ | $0.0000\text{ s}$ | $1.86\times$ | $92.9\%$ | $5.48 \times 10^{-12}$ |
| **2** | $2$ | $1$ | $0.4658\text{ s}$ | $0.4590\text{ s}$ | $0.0119\text{ s}$ | $1.95\times$ | $97.6\%$ | $5.48 \times 10^{-12}$ |
| **4** | $1$ | $4$ | $0.3047\text{ s}$ | $0.3046\text{ s}$ | $0.0000\text{ s}$ | $2.98\times$ | $74.6\%$ | $5.81 \times 10^{-12}$ |
| **4** | $2$ | $2$ | $0.2972\text{ s}$ | $0.2943\text{ s}$ | $0.0112\text{ s}$ | $3.06\times$ | $76.4\%$ | $5.81 \times 10^{-12}$ |
| **4** | $4$ | $1$ | $0.2943\text{ s}$ | $0.2853\text{ s}$ | $0.0136\text{ s}$ | $3.09\times$ | $77.2\%$ | $5.81 \times 10^{-12}$ |
| **8** | **$1$** | **$8$** | **$0.2500\text{ s}$** | $0.2498\text{ s}$ | **$0.0000\text{ s}$** | **$3.64\times$** | **$45.4\%$** | $6.76 \times 10^{-12}$ |
| **8** | $2$ | $4$ | $0.2661\text{ s}$ | $0.2631\text{ s}$ | $0.0096\text{ s}$ | $3.42\times$ | $42.7\%$ | $6.76 \times 10^{-12}$ |
| **8** | $4$ | $2$ | $0.2660\text{ s}$ | $0.2555\text{ s}$ | $0.0200\text{ s}$ | $3.42\times$ | $42.7\%$ | $6.76 \times 10^{-12}$ |
| **8** | $8$ | $1$ | $0.2624\text{ s}$ | $0.1969\text{ s}$ | $0.0711\text{ s}$ | $3.46\times$ | $43.3\%$ | $6.76 \times 10^{-12}$ |

- **Key Hybrid Insight:** At $8$ total cores, Pure MPI (`P=8, T=1`) spends **$0.0711\text{ s}$ ($27.1\%$)** in `MPI_Allreduce` communication/synchronization. Switching to Hybrid (`P=2, T=4` or `P=1, T=8`) cuts `MPI_Allreduce` overhead by **$7.4\times$** (down to $0.0096\text{ s}$ or $0.0000\text{ s}$) and achieves the fastest overall CPU runtime (**$0.2500\text{ s}$, $3.64\times$ speedup**).

---

## 6. Step 4 Completed: GPU Acceleration (`MPI + CUDA` & `MPI + Metal GPU`)

Per **Lecture 12 Slide 37** (*"If you choose X=OpenMP, then you also have to test on at least one GPU"*), we implemented three GPU acceleration paths:
1. **`src/kmeans_cuda.cu` (Hybrid `MPI + CUDA` for Vanda NVIDIA GPUs):** Caches centroids in on-chip `__shared__` memory (`s_centroids`), accumulates partial sums in block shared memory (`s_sums`), and flushes once per thread block to global memory before `MPI_Allreduce`.
2. **`src/kmeans_omp.cpp -g` (OpenMP Target GPU Offload):** Uses `#pragma omp target teams distribute parallel for` for NVIDIA `nvc++ -mp=gpu` compilation on Vanda.
3. **`src/kmeans_metal_gpu.mm` (Hybrid `MPI + Apple Metal GPU Shader` for Apple M5 GPU):** Uses on-chip `threadgroup float s_centroids[1024]` shared memory and zero-copy Unified Shared Memory (`MTLResourceStorageModeShared`) to offload the $\mathcal{O}(N \cdot K \cdot d)$ distance evaluations to the Apple M5 GPU.

### Hardware GPU Benchmark on UCI HIGGS ($N = 597,638, d = 28, 30\text{ Iters}$, `results/gpu_benchmark.csv`)
| Clusters ($K$) | Distance Evals / Iter | 1-Core CPU (s) | 4-Core OpenMP (s) | Apple M5 GPU Total (s) | GPU Kernel Time (s) | GPU Speedup vs. 1-Core CPU | GPU Speedup vs. 4-Core OMP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **$8$** | $4.78\text{ M}$ | $0.5621\text{ s}$ | $0.1871\text{ s}$ | $0.2022\text{ s}$ | $0.1489\text{ s}$ | **$2.78\times$** | $0.93\times$ |
| **$16$** | $9.56\text{ M}$ | $0.9243\text{ s}$ | $0.3100\text{ s}$ | **$0.1459\text{ s}$** | **$0.0945\text{ s}$** | **$6.34\times$** | **$2.12\times$** |
| **$32$** | $19.12\text{ M}$ | $1.6754\text{ s}$ | $0.5563\text{ s}$ | **$0.2070\text{ s}$** | **$0.1557\text{ s}$** | **$8.09\times$** | **$2.69\times$** |

- **GPU Arithmetic Intensity Analysis:** At small cluster counts ($K=8$), GPU kernel dispatch overhead makes 4-core CPU OpenMP ($0.1871\text{ s}$) and M5 GPU ($0.2022\text{ s}$) nearly equal. As arithmetic intensity increases to $K=16$ and $K=32$ ($19.12\text{ million}$ 28D distance evaluations per iteration), the GPU's massive SIMD parallelism and on-chip `threadgroup` centroid cache dominate, reaching an **$8.09\times$ speedup over 1-core CPU** and **$2.69\times$ speedup over 4-core OpenMP**!

---

## 7. Quickstart: Building & Running All Implementations

```bash
# 1. Stream and clean the UCI HIGGS dataset (ID #280) into data/higgs_clean.csv
python3 scripts/prepare_higgs_data.py

# 2. Compile Serial, Pure MPI, Hybrid MPI+OpenMP, and Metal GPU binaries
make

# 3. Run Hybrid MPI+OpenMP (e.g. 2 MPI ranks x 4 OpenMP threads, guided schedule, padded private accumulators)
mpirun -np 2 ./bin/kmeans_omp -f data/higgs_clean.csv -r results/centroids_serial.txt -threads 4 -a private -s guided -k 8 -m 50

# 4. Run Hybrid MPI + GPU (Apple Metal Compute Shader on M5 GPU)
mpirun -np 1 ./bin/kmeans_metal_gpu -f data/higgs_clean.csv -r results/centroids_serial.txt -threads 4 -k 32 -m 30

# 5. Reproduce all Hybrid OpenMP & GPU benchmark CSV tables
python3 scripts/run_hybrid_and_gpu_benchmarks.py

# 6. Submit Vanda HPC 256-Core MPI & 72-Core/NVIDIA GPU jobs
qsub scripts/vanda_mpi_256cores.pbs
qsub scripts/vanda_hybrid_gpu.pbs
```
