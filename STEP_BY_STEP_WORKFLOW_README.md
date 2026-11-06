# Complete Step-by-Step Workflow, Terminal Commands, Dataset Details & Runtime Comparisons

**Course:** DSA5208 — Scalable Distributed Computing for Data Science (NUS)  
**GitHub Repository:** [https://github.com/Nithillen/dsa5208-project3-scalable-kmeans](https://github.com/Nithillen/dsa5208-project3-scalable-kmeans)  
**Dataset Used:** [UCI Machine Learning Repository — HIGGS Dataset (ID #280)](https://archive.ics.uci.edu/dataset/280/higgs)

This document records **every single step** performed from the beginning of the project, including how the dataset was downloaded and extracted, how many files were produced and their exact byte/megabyte sizes, the exact terminal commands executed at each stage, what each implementation does, and the exact runtime differences between every parallelization method.

---

## Step 1: Downloading, Streaming & Extracting the UCI HIGGS Dataset

### 1.1 Source Dataset & File Count / Size Breakdown
The dataset comes from the official UCI Machine Learning Repository:
- **Dataset Page:** `https://archive.ics.uci.edu/dataset/280/higgs`
- **Direct Archive URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz`
- **What is inside the raw UCI download?**
  - **1 single compressed file:** `HIGGS.csv.gz`
  - **Compressed Download Size:** **$2.62\text{ GB}$** ($2,816,407,858\text{ bytes}$)
  - **Uncompressed Full Size (`HIGGS.csv`):** **$7.48\text{ GB}$** ($11,000,000\text{ rows} \times 29\text{ columns}$)
  - **Column Layout:**
    - `Column 0`: Binary class label (`1.0` = Higgs Boson signal event $H^0 \to W^+W^- \to b\bar{b}\ell\nu$, `0.0` = background $t\bar{t}$ process).
    - `Columns 1–21`: 21 low-level particle detector kinematic features (lepton $p_T, \eta, \phi$, missing transverse energy $\text{MET}, \text{MET}_\phi$, and $p_T, \eta, \phi, b\text{-tag}$ for up to 4 particle jets).
    - `Columns 22–28`: 7 high-level invariant mass features ($m_{jj}, m_{jjj}, m_{\ell\nu}, m_{j\ell\nu}, m_{bb}, m_{wbb}, m_{wwbb}$) derived by physicists.

### 1.2 How We Extracted & Cleaned the Dataset (`scripts/prepare_higgs_data.py`)
Instead of storing an $8\text{ GB}$ uncompressed CSV on disk, we wrote `scripts/prepare_higgs_data.py`, which streams `HIGGS.csv.gz` in $1\text{ MB}$ compressed HTTP chunks, decompresses on the fly using `zlib`, filters out any corrupted/NaN lines, and splits the features and labels into **2 clean files** inside `data/`:

| Extracted File Path | Exact Byte Size | Size in MB | Exact Line (Row) Count | Columns per Row | Total Values | Purpose |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`data/higgs_clean.csv`** | `137,159,578 bytes` | **130.81 MB** | `597,638` rows | `28` float features | `16,733,864` floats | Unlabeled feature matrix $X \in \mathbb{R}^{597638 \times 28}$ fed into K-Means |
| **`data/higgs_labels.txt`** | `1,195,276 bytes` | **1.14 MB** | `597,638` rows | `1` binary integer (`0`/`1`) | `597,638` labels | Ground-truth labels used **only** for post-clustering physics evaluation |
| **TOTAL (2 Files)** | **`138,354,854 bytes`** | **131.95 MB** | **`597,638` events** | **`29` total columns** | **`17,331,502` values** | Complete local & HPC benchmark dataset |

### 1.3 Exact Terminal Commands Executed in Step 1
```bash
# 1. Run the streaming downloader & cleaner
python3 scripts/prepare_higgs_data.py

# 2. Verify extracted file count, exact byte sizes, and row counts
ls -lh data/
ls -l data/
wc -l data/*
```

---

## Step 2: 1-Node Serial Baseline & Problem-Size Scaling (`src/kmeans_serial.cpp`)

### 2.1 What We Built
We implemented Lloyd's K-Means algorithm in `src/kmeans_serial.cpp` on a single CPU core:
1. **Buffered Fast CSV Loader (`load_csv_data`):** Uses an $8\text{ MB}$ `std::ifstream` buffer and pointer-based `std::strtod` parsing to load all $16,733,864$ floats from `data/higgs_clean.csv` in **$0.29\text{ seconds}$**.
2. **Deterministic Centroid Initialization (`initialize_centroids`):** Selects $K$ evenly spaced points ($\text{idx} = k \cdot \lfloor N/K \rfloor$) so that Serial, Pure MPI, Hybrid OpenMP, and GPU runs start from the **exact same initial centroids**.
3. **Iterative Expectation–Maximization Loop:**
   - **Expectation Step:** For each point $i \in \{0, \dots, N-1\}$ and centroid $k \in \{0, \dots, K-1\}$, computes squared Euclidean distance $D_{ik} = \sum_{j=0}^{d-1} (x_{ij} - \mu_{kj})^2$ in $\mathcal{O}(N \cdot K \cdot d)$ time and assigns point $i$ to $k^* = \arg\min_k D_{ik}$.
   - **Maximization Step:** Updates $\mu_{kj} \leftarrow \frac{1}{|C_k|} \sum_{i \in C_k} x_{ij}$ and checks if $\max_k \|\boldsymbol{\mu}_k^{(t+1)} - \boldsymbol{\mu}_k^{(t)}\|_2 < \text{tolerance}$.
4. **Reference Validation Output:** Saves the converged $8 \times 28$ centroids to `results/centroids_serial.txt`.

### 2.2 Exact Terminal Commands Executed in Step 2
```bash
# 1. Compile the serial baseline
make bin/kmeans_serial

# 2. Run 50 iterations on the full UCI HIGGS dataset (N=597,638, d=28, K=8)
./bin/kmeans_serial -f data/higgs_clean.csv -l data/higgs_labels.txt -k 8 -m 50 -t 1e-5

# 3. Run the automated problem-size (N) and cluster-count (K) scaling benchmark
python3 scripts/run_serial_scaling.py
```

### 2.3 Step 2 Benchmark Results (`results/serial_scaling_N.csv` & `results/serial_scaling_K.csv`)
- **Full 50-Iteration Reference Run ($N = 597,638, d = 28, K = 8$):**
  - **Total Runtime:** **$1.3380\text{ seconds}$**
  - **Average Time per Iteration:** **$26.76\text{ ms}$**
  - **Throughput:** **$22.33\text{ million points/sec}$**
- **Problem Size $N$ Scaling (30 Iterations, $K = 8, d = 28$):**
  - $N = 25,000$: **$0.0338\text{ s}$** ($1.13\text{ ms/iter}$, $22.18\text{ M pts/s}$)
  - $N = 50,000$: **$0.0651\text{ s}$** ($2.17\text{ ms/iter}$, $23.05\text{ M pts/s}$, $1.93\times$ time vs. $25\text{k}$)
  - $N = 100,000$: **$0.1307\text{ s}$** ($4.36\text{ ms/iter}$, $22.95\text{ M pts/s}$, $3.87\times$ time vs. $25\text{k}$)
  - $N = 200,000$: **$0.2633\text{ s}$** ($8.78\text{ ms/iter}$, $22.79\text{ M pts/s}$, $7.79\times$ time vs. $25\text{k}$)
  - $N = 400,000$: **$0.5311\text{ s}$** ($17.70\text{ ms/iter}$, $22.59\text{ M pts/s}$, $15.71\times$ time vs. $25\text{k}$)
  - $N = 597,638$: **$0.7943\text{ s}$** ($26.48\text{ ms/iter}$, $22.57\text{ M pts/s}$, $23.50\times$ time vs. $25\text{k}$)
- **Cluster Count $K$ Scaling (30 Iterations, $N = 597,638, d = 28$):**
  - $K = 2$: **$0.2663\text{ s}$** ($8.88\text{ ms/iter}$, $67.34\text{ M pts/s}$)
  - $K = 4$: **$0.4917\text{ s}$** ($16.39\text{ ms/iter}$, $36.46\text{ M pts/s}$)
  - $K = 8$: **$0.8023\text{ s}$** ($26.74\text{ ms/iter}$, $22.35\text{ M pts/s}$)
  - $K = 16$: **$1.4189\text{ s}$** ($47.30\text{ ms/iter}$, $12.64\text{ M pts/s}$)
  - $K = 32$: **$2.6691\text{ s}$** ($88.97\text{ ms/iter}$, $6.72\text{ M pts/s}$)

---

## Step 3: Pure MPI Parallelization (`src/kmeans_mpi.cpp`)

### 3.1 What We Built
1. **Installed OpenMPI (`open-mpi 5.0.11`) and `libomp 23.1.5`** via Homebrew.
2. **1D Row-Block Domain Decomposition (`MPI_Scatterv`):** Rank 0 reads `data/higgs_clean.csv`, broadcasts initial centroids via `MPI_Bcast`, and scatters $N_p = \lfloor N/P \rfloor + \mathbb{I}(p < N \bmod P)$ rows to each MPI rank $p$.
3. **Per-Iteration `MPI_Allreduce` Global Synchronization:** Each rank computes local partial sums `local_sums` ($K \times d$ doubles) and `local_counts` ($K$ integers) and executes two `MPI_Allreduce(..., MPI_SUM)` calls per iteration.
4. **Separate Instrumentation (`MPI_Wtime`):** Measures `Computation Time` vs. `MPI_Allreduce Communication Time` separately and validates final centroids against `results/centroids_serial.txt`.

### 3.2 Exact Terminal Commands Executed in Step 3
```bash
# 1. Install OpenMPI and libomp on macOS
brew install open-mpi libomp

# 2. Compile Pure MPI implementation
make bin/kmeans_mpi

# 3. Run Pure MPI Strong & Weak Scaling benchmark suite
python3 scripts/run_mpi_scaling.py
```

### 3.3 Step 3 Benchmark Results (`results/mpi_strong_scaling.csv` & `results/mpi_weak_scaling.csv`)
- **Strong Scaling on UCI HIGGS ($N = 597,638, d = 28, K = 8, 50\text{ Iterations}$):**
  - $P = 1$: Total $= \mathbf{1.3101\text{ s}}$ (`Comp = 1.3100 s`, `Comm = 0.0000 s [0.0%]`, Speedup $= \mathbf{1.00\times}$, Efficiency $= \mathbf{100.0\%}$, Max Error $= 4.977\times 10^{-12}$)
  - $P = 2$: Total $= \mathbf{0.6802\text{ s}}$ (`Comp = 0.6761 s`, `Comm = 0.0049 s [0.7%]`, Speedup $= \mathbf{1.93\times}$, Efficiency $= \mathbf{96.3\%}$, Max Error $= 5.484\times 10^{-12}$)
  - $P = 4$: Total $= \mathbf{0.4031\text{ s}}$ (`Comp = 0.3983 s`, `Comm = 0.0108 s [2.7%]`, Speedup $= \mathbf{3.25\times}$, Efficiency $= \mathbf{81.3\%}$, Max Error $= 5.810\times 10^{-12}$)
  - $P = 8$: Total $= \mathbf{0.4403\text{ s}}$ (`Comp = 0.3174 s`, `Comm = 0.1442 s [32.8%]`, Speedup $= \mathbf{2.98\times}$, Efficiency $= \mathbf{37.2\%}$, Max Error $= 6.763\times 10^{-12}$)
- **Weak Scaling ($N_{\text{per\_rank}} = 200,000\text{ pts/rank}, d = 28, K = 8, 30\text{ Iterations}$):**
  - $P = 1$ ($N = 200,000$): Total $= \mathbf{0.2927\text{ s}}$ (`Comp = 0.2926 s`, `Comm = 0.0000 s [0.0%]`, Weak Efficiency $= \mathbf{100.0\%}$, Throughput $= \mathbf{20.50\text{ M pts/s}}$)
  - $P = 2$ ($N = 400,000$): Total $= \mathbf{0.2733\text{ s}}$ (`Comp = 0.2702 s`, `Comm = 0.0032 s [1.2%]`, Weak Efficiency $= \mathbf{107.1\%}$, Throughput $= \mathbf{43.91\text{ M pts/s}}$)
  - $P = 4$ ($N = 800,000$): Total $= \mathbf{0.3695\text{ s}}$ (`Comp = 0.3569 s`, `Comm = 0.0324 s [8.8%]`, Weak Efficiency $= \mathbf{79.2\%}$, Throughput $= \mathbf{64.94\text{ M pts/s}}$)
  - $P = 8$ ($N = 1,600,000$): Total $= \mathbf{0.5518\text{ s}}$ (`Comp = 0.4662 s`, `Comm = 0.1173 s [21.2%]`, Weak Efficiency $= \mathbf{53.0\%}$, Throughput $= \mathbf{86.98\text{ M pts/s}}$)

---

## Step 4: Hybrid `MPI + OpenMP` & Hardware GPU Acceleration (`src/kmeans_omp.cpp`, `src/kmeans_cuda.cu`, `src/kmeans_metal_gpu.mm`)

### 4.1 What We Built
1. **`src/kmeans_omp.cpp` (Hybrid `MPI + OpenMP` + OpenMP GPU Target Offload):**
   - Initializes MPI with `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)` and spawns $T$ OpenMP threads per MPI rank.
   - Implements `#pragma omp simd reduction(+:dist)` for vectorized inner distance loops.
   - Compares **Cache-Padded Thread-Private Accumulators (`-a private`)** vs. **Shared Atomic Updates (`-a atomic`)**.
   - Compares OpenMP scheduling policies (`-s static`, `-s dynamic`, `-s guided`).
   - Supports OpenMP GPU target offload (`-g` flag using `#pragma omp target teams distribute parallel for`).
2. **`src/kmeans_cuda.cu` (Hybrid `MPI + CUDA` for Vanda NVIDIA GPUs):**
   - Uses on-chip `__shared__ double s_centroids[]` memory to cache centroids per thread block and performs two-stage shared-memory reduction before `MPI_Allreduce`.
3. **`src/kmeans_metal_gpu.mm` (Hybrid `MPI + Apple Metal GPU Shader` for Apple M5 GPU):**
   - Dispatches a Metal GPU compute shader (`kmeans_assign_kernel`) on the Apple M5 GPU using `threadgroup float s_centroids[1024]` shared memory and zero-copy Unified Shared Memory (`MTLResourceStorageModeShared`).

### 4.2 Exact Terminal Commands Executed in Step 4
```bash
# 1. Compile Hybrid OpenMP and Metal GPU executables
make

# 2. Run the complete Hybrid MPI+OpenMP & GPU benchmark suite
python3 scripts/run_hybrid_and_gpu_benchmarks.py
```

### 4.3 Exact Time Differences Across All Methods
- **OpenMP Accumulation & Schedule Design Decisions ($P = 1, T = 4, K = 8, 50\text{ Iterations}$ on UCI HIGGS):**
  - `atomic` + `static`: **$1.8447\text{ s}$** ($36.89\text{ ms/iter}$, $16.20\text{ M pts/s}$)
  - `private` + `static`: **$0.3265\text{ s}$** ($6.53\text{ ms/iter}$, $91.51\text{ M pts/s}$ — **$5.65\times$ faster than `atomic`, saves $1.5182\text{ s}$**)
  - `private` + `dynamic`: **$0.3054\text{ s}$** ($6.11\text{ ms/iter}$, $97.85\text{ M pts/s}$ — **$6.04\times$ faster than `atomic`, saves $1.5393\text{ s}$**)
  - `private` + `guided`: **$0.2947\text{ s}$** ($5.89\text{ ms/iter}$, $101.40\text{ M pts/s}$ — **$6.26\times$ faster than `atomic`, saves $1.5500\text{ s}$**)
- **Hybrid MPI Ranks ($P$) $\times$ OpenMP Threads ($T$) Grid ($50\text{ Iterations}, K = 8$ on UCI HIGGS):**
  - $(P=1, T=1)$ [1 core]: **$0.9088\text{ s}$** (`Comp = 0.9087 s`, `Comm = 0.0000 s`, Speedup $= 1.00\times$)
  - $(P=1, T=2)$ [2 cores]: **$0.4890\text{ s}$** (`Comp = 0.4889 s`, `Comm = 0.0000 s`, Speedup $= 1.86\times$)
  - $(P=2, T=1)$ [2 cores]: **$0.4658\text{ s}$** (`Comp = 0.4590 s`, `Comm = 0.0119 s`, Speedup $= 1.95\times$)
  - $(P=1, T=4)$ [4 cores]: **$0.3047\text{ s}$** (`Comp = 0.3046 s`, `Comm = 0.0000 s`, Speedup $= 2.98\times$)
  - $(P=2, T=2)$ [4 cores]: **$0.2972\text{ s}$** (`Comp = 0.2943 s`, `Comm = 0.0112 s`, Speedup $= 3.06\times$)
  - $(P=4, T=1)$ [4 cores]: **$0.2943\text{ s}$** (`Comp = 0.2853 s`, `Comm = 0.0136 s`, Speedup $= 3.09\times$)
  - $(P=1, T=8)$ [8 cores]: **$0.2500\text{ s}$** (`Comp = 0.2498 s`, `Comm = 0.0000 s`, Speedup $= \mathbf{3.64\times}$ — **Fastest 8-core CPU configuration!**)
  - $(P=2, T=4)$ [8 cores]: **$0.2661\text{ s}$** (`Comp = 0.2631 s`, `Comm = 0.0096 s`, Speedup $= 3.42\times$ — **$7.4\times$ less MPI communication than $P=8, T=1$**)
  - $(P=4, T=2)$ [8 cores]: **$0.2660\text{ s}$** (`Comp = 0.2555 s`, `Comm = 0.0200 s`, Speedup $= 3.42\times$)
  - $(P=8, T=1)$ [8 cores]: **$0.2624\text{ s}$** (`Comp = 0.1969 s`, `Comm = 0.0711 s`, Speedup $= 3.46\times$)
- **Hardware GPU vs. CPU Benchmark ($30\text{ Iterations}$ on UCI HIGGS across $K \in \{8, 16, 32\}$):**
  - **$K = 8$:** 1-Core CPU $= \mathbf{0.5621\text{ s}}$ | 4-Core OMP $= \mathbf{0.1871\text{ s}}$ | M5 GPU Total $= \mathbf{0.2022\text{ s}}$ (GPU Kernel $= \mathbf{0.1489\text{ s}}$, Speedup vs. 1-Core $= \mathbf{2.78\times}$)
  - **$K = 16$:** 1-Core CPU $= \mathbf{0.9243\text{ s}}$ | 4-Core OMP $= \mathbf{0.3100\text{ s}}$ | M5 GPU Total $= \mathbf{0.1459\text{ s}}$ (GPU Kernel $= \mathbf{0.0945\text{ s}}$, Speedup vs. 1-Core $= \mathbf{6.34\times}$, vs. 4-Core OMP $= \mathbf{2.12\times}$)
  - **$K = 32$:** 1-Core CPU $= \mathbf{1.6754\text{ s}}$ | 4-Core OMP $= \mathbf{0.5563\text{ s}}$ | M5 GPU Total $= \mathbf{0.2070\text{ s}}$ (GPU Kernel $= \mathbf{0.1557\text{ s}}$, Speedup vs. 1-Core $= \mathbf{8.09\times}$, vs. 4-Core OMP $= \mathbf{2.69\times}$)

---

## Step 5: Generating Log-Log Figures & Compiling the Final PDF Report

### 5.1 Exact Terminal Commands Executed in Step 5
```bash
# 1. Generate all 6 publication-quality 300-DPI log-log plots directly from results/*.csv
/tmp/report_venv/bin/python3 scripts/generate_report_figures.py

# 2. Compile the academic PDF report using Typst
typst compile report/DSA5208_Project3_Report.typ DSA5208_Project3_Report.pdf
```
