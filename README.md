# DSA5208 Project 3: Scalable Hybrid Parallel K-Means Clustering (`MPI + OpenMP + CUDA`)

**Course:** DSA5208 — Scalable Distributed Computing for Data Science (NUS)  
**Target Platforms:** NUS Vanda Supercomputer (72-core CPU nodes & NVIDIA GPUs), NUS Atlas8, and macOS Local Development  
**Algorithm:** Lloyd's K-Means Clustering (Iterative Expectation–Maximization)  
**Real-World Dataset:** UCI HIGGS Boson High-Energy Physics Dataset ($N = 597,638$ collision events $\times$ $d = 28$ kinematic features) + Multi-Million Point Synthetic Gaussian Generator for Weak Scaling up to 256 CPU Cores.

---

## 1. Why This Algorithm & Dataset Were Chosen

### A. Why Lloyd's K-Means Clustering?
The project requires an algorithm that is **not embarrassingly parallel** and reveals rich parallel scaling trade-offs across distributed memory (`MPI`), shared memory (`OpenMP`), and accelerators (`GPU / CUDA`):
1. **Iterative Global Synchronization (`MPI_Allreduce`):** Unlike embarrassingly parallel tasks (such as Monte Carlo sampling), K-Means is an iterative Expectation–Maximization (EM) algorithm. In every single iteration $t$, each worker computes distances from its local data partition to all $K$ centroids, accumulates partial feature sums $\mathbf{S}_k^{(p)} \in \mathbb{R}^d$ and counts $C_k^{(p)} \in \mathbb{Z}$, and **must synchronize globally** via `MPI_Allreduce` before iteration $t+1$ can begin.
2. **Computation-to-Communication Ratio:**
   - **Per-iteration Computation:** $\mathcal{O}\!\left(\frac{N \cdot K \cdot d}{P}\right)$ floating-point operations per rank.
   - **Per-iteration Communication:** $\mathcal{O}(K \cdot d \cdot \log P)$ message reduction complexity across $P$ MPI ranks.
   - As $P$ scales up to **256 CPU cores**, the local computation per rank shrinks by $256\times$, exposing the `MPI_Allreduce` latency floor (Amdahl's Law in strong scaling vs. Gustafson's Law in weak scaling).
3. **Shared-Memory & GPU Write Contention:** During the accumulation step, thousands of threads attempt to add $d$-dimensional vectors into the same $K$ centroid buckets simultaneously. Comparing **thread-private accumulators** vs. **atomic updates** (`#pragma omp atomic` / CUDA `atomicAdd` / shared-memory block reductions) provides a clear, measurable design discussion for the report.
4. **Exact Deterministic Validation:** When initialized deterministically across ranks, the Serial, Pure MPI, `MPI + OpenMP`, and `MPI + CUDA` implementations converge to the **exact same centroids** within IEEE-754 double-precision floating-point tolerance ($\sim 10^{-12}$).

### B. Why the UCI HIGGS Boson Dataset?
- **Source:** [UCI Machine Learning Repository — HIGGS Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz) (Baldi, Sadowski, & Whiteson, *Nature Communications*, 2014).
- **Scale & Dimensionality:** Each collision event has **$d = 28$ continuous physics features**:
  - **21 low-level kinematic features:** Transverse momenta ($p_T$), pseudorapidities ($\eta$), azimuthal angles ($\phi$) of leptons and up to 4 particle jets, plus Missing Transverse Energy ($\text{MET}$).
  - **7 high-level invariant mass features:** $m_{jj}, m_{jjj}, m_{\ell\nu}, m_{j\ell\nu}, m_{bb}, m_{wbb}, m_{wwbb}$ constructed by particle physicists to distinguish **Higgs boson signal events ($y=1$, $H^0 \to W^+W^- \to b\bar{b}\ell\nu$)** from **background events ($y=0$, top-quark pair production $t\bar{t}$)**.
- **Real-World Scientific Conclusions:** Unsupervised K-Means clustering on the 28-dimensional feature space naturally separates collision topologies into distinct kinematic regimes—isolating high-$m_{bb}$ / high-$m_{wwbb}$ clusters vs. resonant $m_{bb} \approx 0.90$ clusters that significantly enrich the true Higgs boson signal percentage.

---

## 2. Repository Structure & Progress So Far

```text
project3_scalable/
├── .gitignore                 # Excludes >100 MB raw CSVs & compiled binaries from Git
├── Makefile                   # Build system for Serial, MPI, OpenMP, and CUDA targets
├── README.md                  # Complete project documentation & progress log
├── bin/                       # Compiled executables
├── data/                      # Generated locally via scripts/prepare_higgs_data.py
│   ├── higgs_clean.csv        # 597,638 x 28 cleaned HIGGS features (131 MB)
│   └── higgs_labels.txt       # Ground-truth labels (1 = Higgs signal, 0 = background)
├── results/
│   └── centroids_serial.txt   # Converged 8 x 28 reference centroids for validation
├── scripts/
│   └── prepare_higgs_data.py  # Fast HTTP streaming downloader & cleaner for UCI HIGGS
└── src/
    └── kmeans_serial.cpp      # Completed 1-core reference baseline + physics analyzer
```

### Completed Milestones:
1. **Streaming Data Pipeline (`scripts/prepare_higgs_data.py`):**
   - Streams the compressed UCI `HIGGS.csv.gz` archive over HTTP using chunked `zlib` decompression, strips invalid/NaN records, separates ground-truth binary labels (`data/higgs_labels.txt`) from the 28 continuous features (`data/higgs_clean.csv`, $597,638 \times 28 = 16,733,864$ floats, $131\text{ MB}$).
2. **High-Performance Serial Baseline (`src/kmeans_serial.cpp`):**
   - Implements Lloyd's K-Means in C++17 with an 8 MB buffered `std::strtod` CSV parser (loads 16.7M floats in ~1.2 seconds).
   - Supports both the real HIGGS dataset (`-f data/higgs_clean.csv`) and a deterministic multi-million point synthetic Gaussian cluster generator (`-n <N> -d <d> -k <K>`) for weak scaling benchmarks up to 256 cores.
   - Outputs per-iteration centroid shift, total runtime, average time per iteration, throughput (million points/sec), and saves converged reference centroids to `results/centroids_serial.txt`.
   - Includes a post-clustering physics evaluator (`analyze_higgs_clusters`) that correlates the discovered clusters against ground-truth Higgs boson labels and invariant mass features ($p_T^\ell, \text{MET}, m_{bb}, m_{wbb}, m_{wwbb}$).

### Baseline Serial Results on UCI HIGGS ($N = 597,638, d = 28, K = 8$):
| Cluster | Events | True Higgs Signal % | Lepton $p_T$ | $\text{MET}$ | $m_{bb}$ | $m_{wbb}$ | $m_{wwbb}$ | Physical Interpretation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **C1** | 40,860 | 44.32% | 1.158 | 1.186 | 0.961 | 1.113 | 1.078 | High-$p_T$ multi-jet background-enriched |
| **C2** | 49,334 | 54.09% | 1.114 | 1.106 | 0.896 | 0.996 | 0.958 | Moderate invariant mass signal-enriched |
| **C3** | 115,356 | 53.76% | 0.916 | 0.929 | 0.779 | 0.869 | 0.822 | Low-mass compact collision topology |
| **C4** | 109,017 | **55.64%** | 0.927 | 0.946 | 0.822 | 0.898 | 0.851 | **Highest Higgs signal enrichment ($>55.6\%$)** |
| **C5** | 38,476 | 47.25% | 1.095 | 1.080 | **2.221** | **1.663** | **1.545** | Heavy $b\bar{b}$ tail ($t\bar{t}$ background-dominated) |
| **C6** | 95,477 | 55.41% | 0.959 | 0.968 | 0.903 | 0.925 | 0.875 | Resonant $m_{bb} \approx 0.90$ Higgs decay channel |
| **C7** | 49,648 | 49.06% | 1.087 | 1.078 | 0.880 | 1.150 | 1.132 | Intermediate $W b\bar{b}$ continuum |
| **C8** | 99,470 | 53.68% | 0.967 | 0.963 | 0.848 | 0.910 | 0.862 | Central kinematic bulk |

---

## 3. Step-by-Step Roadmap (DSA5208 Rubric Alignment)

| Step | Component | Rubric Weight | Description |
| :--- | :--- | :---: | :--- |
| **Step 1** | **Serial Problem-Size Scaling** (`src/kmeans_serial.cpp`) | **5 pts** | Benchmark runtime vs. problem size $N \in \{10^5, 2.5\times 10^5, 5\times 10^5, 10^6, 2\times 10^6, 4\times 10^6\}$ and cluster count $K \in \{4, 8, 16, 32\}$ to verify empirical $\mathcal{O}(N \cdot K \cdot d)$ complexity on log-log axes. |
| **Step 2** | **Pure MPI Parallelization** (`src/kmeans_mpi.cpp`) | **10 pts** | 1D row-block domain decomposition across $P \in \{1, 2, 4, 8, 16, 32, 64, 128, 256\}$ CPU cores on Vanda. Instrument separate timers for **Computation Time** vs. **`MPI_Allreduce` Communication Time** under both **Strong Scaling** (fixed $N$) and **Weak Scaling** (fixed $N/P$ per core). |
| **Step 3** | **Hybrid `MPI + OpenMP` (CPU & GPU)** (`src/kmeans_omp.cpp`) | **10 pts** | Combine inter-node MPI with intra-node OpenMP threads on Vanda's 72-core nodes. Compare thread-private accumulators vs. atomic contention, OpenMP scheduling (`static`, `dynamic`, `guided`), and ratio of MPI ranks $\times$ OpenMP threads per node. |
| **Step 4** | **GPU Acceleration (`MPI + CUDA`)** (`src/kmeans_cuda.cu`) | **Required with OpenMP** | Satisfies the Lecture 12 Slide 37 requirement (*"If you choose X=OpenMP, then you also have to test on at least one GPU"*). Offloads point-to-centroid distance evaluation and partial reductions to NVIDIA GPUs on Vanda using shared-memory tiled centroids. |
| **Step 5** | **Validation, Plots & Final PDF Report** | **10 pts** | Verify $\max_{k,j} |C_{k,j}^{\text{parallel}} - C_{k,j}^{\text{serial}}| < 10^{-11}$, generate log-log scaling/efficiency plots, and compile the final report. |

---

## 4. Quickstart Guide

### Step 1: Prepare the HIGGS Dataset (if not already in `data/`)
```bash
python3 scripts/prepare_higgs_data.py
```

### Step 2: Compile & Run the Serial Baseline
```bash
make
# Run on real UCI HIGGS dataset (597,638 x 28):
./bin/kmeans_serial -f data/higgs_clean.csv -l data/higgs_labels.txt -k 8 -m 50 -t 1e-5

# Run on synthetic Gaussian data (e.g. N = 2,000,000 points):
./bin/kmeans_serial -n 2000000 -d 28 -k 8 -m 50 -t 1e-5
```
