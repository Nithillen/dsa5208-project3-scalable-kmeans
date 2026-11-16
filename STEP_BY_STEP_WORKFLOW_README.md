# Complete Step-by-Step Reproducible Workflow, Commands & Measured Results

**Course:** DSA5208 — Scalable Distributed Computing for Data Science (NUS)  
**Authors:** Nithillen Jayaseelan (`A0351473E`), Ashwin Sasikumar (`A0354333J`), Amitej Singh Datta (`A0352547Y`)  
**GitHub Repository:** [https://github.com/Nithillen/dsa5208-project3-scalable-kmeans](https://github.com/Nithillen/dsa5208-project3-scalable-kmeans)  
**Dataset:** [UCI Machine Learning Repository — HIGGS Boson Dataset (ID #280)](https://archive.ics.uci.edu/dataset/280/higgs) ($N = 11,000,000 \times d = 28$ features)

---

## Step 1: Streaming the Complete 11,000,000-Row UCI HIGGS Dataset (`scripts/prepare_higgs_data.py`)

`HIGGS.csv.gz` (`2.62 GB` compressed, `7.48 GB` uncompressed, `11,000,000` rows $\times$ `29` columns) is streamed over HTTP, decompressed on the fly via `zlib`, and written as two clean files (`0` skipped rows):
1. `data/higgs_clean.csv` — `11,000,000` rows $\times$ `28` continuous physics features (`308,000,000` IEEE-754 doubles $= 2.464\text{ GB}$ in RAM).
2. `data/higgs_labels.txt` — `11,000,000` binary ground-truth class labels (`1` = Higgs boson signal, `0` = background), used strictly **after** clustering for cluster purity analysis.

```bash
python3 scripts/prepare_higgs_data.py --all
wc -l data/higgs_clean.csv data/higgs_labels.txt
```

To eliminate redundant multi-second ASCII parsing across dozens of benchmark runs on 11M rows, `src/kmeans_common.hpp` atomically creates a binary cache `data/higgs_clean.csv.bin` (`KMEANSB1` header + source CSV byte-size verification) on the first CSV load (`8.32 s` on Vanda), cutting subsequent multi-rank loads by $>30\times$ while guaranteeing bitwise parity (`diff -s` verified).

---

## Step 2: Compilation & 16-Test Automated Correctness Verification

```bash
# On Vanda (interactive compute node with NVIDIA HPC SDK):
module load nvhpc-hpcx-cuda12/26.1
make all

# Local or Vanda automated verification suite (16/16 PASSED locally; GPU tests auto-guarded on CPU nodes):
python3 scripts/test_correctness.py
```

All 16 tests verify initialization parity, `MPI_Scatterv` vs. distributed parallel I/O (`-io parallel`), packed vs. separate `MPI_Allreduce`, OpenMP `private` vs. `atomic`, large-$K$ shared-memory bounds fallback ($K=40, d=32 \implies K d = 1280 > 1024$), and Apple Metal GPU accuracy (`MaxErr <= 1.000e-12` on FP64 HIGGS).

---

## Step 3: Submitting Full 11M-Row Vanda HPC Jobs (`scripts/vanda_cpu_benchmarks.pbs` & `scripts/vanda_gpu_benchmarks.pbs`)

```bash
qsub scripts/vanda_cpu_benchmarks.pbs   # Output saved verbatim: kmeans_cpu_benchmarks.o1563432 (CN-111, 72 cores)
qsub scripts/vanda_gpu_benchmarks.pbs   # Output saved verbatim: kmeans_gpu_benchmark.o1563433  (GN-A40-044, NVIDIA A40)
```

### Verified Vanda CPU Readings (`kmeans_cpu_benchmarks.o1563432`, Node `CN-111`, $N=11,000,000$)
- **1-Core 50-Iter Baseline:** `49.0870 s` (`981.74 ms/iter`, `11.20 M pts/s`, crossed `tol=1e-5` at iter `47`, `SSE = 1.584590e+08`).
- **Serial vs. $N$ (`serial_scaling_N.csv`, 30 iters):** `100k` `0.4079 s` $\to$ `500k` `1.2805 s` $\to$ `1M` `2.7431 s` $\to$ `2.5M` `6.4020 s` $\to$ `5M` `12.8151 s` $\to$ `11M` **`28.0452 s`** (`11.77 M pts/s`).
- **Serial vs. $K$ (`serial_scaling_K.csv`, 30 iters, 11M):** `K=2` `10.5133 s` $\to$ `K=4` `15.8274 s` $\to$ `K=8` `28.1074 s` $\to$ `K=16` `60.0651 s` $\to$ `K=32` **`112.5194 s`**.
- **Pure MPI Strong Scaling (`mpi_strong_scaling.csv`, 50 iters, 11M):** `P=1` `31.4583 s` $\to$ `P=2` `14.4035 s` (`2.18x`) $\to$ `P=4` `7.8956 s` (`3.98x`) $\to$ `P=8` `3.9428 s` (`7.98x`) $\to$ `P=16` `2.4039 s` (`13.09x`) $\to$ `P=32` `1.3659 s` (`23.03x`) $\to$ `P=64` **`0.6211 s`** (**`50.65x` speedup**, `79.1%` efficiency, **`885.49 M pts/s`**, `MaxErr = 0.000e+00`).
- **Pure MPI Weak Scaling (`mpi_weak_scaling.csv`, 200k pts/rank, 30 iters):** `P=1` (`200k`, `0.2983 s`, `20.11 M/s`) $\to$ `P=64` (`12.8M`, `0.5491 s`, `54.3%` efficiency, **`699.36 M pts/s`**).
- **OpenMP Design (`omp_design_choices.csv`, $P=1, T=4$, 50 iters):** `private + guided` **`11.5415 s`** (`47.65 M/s`, **`13.52x`** faster than `atomic + static` at `156.0935 s`).
- **72-Core Hybrid NUMA Grid (`hybrid_mpi_omp_grid.csv`, 50 iters, 11M):** `1x1` (`29.3235 s`) $\to$ `1x72` (`2.0441 s`, `14.35x`) vs. NUMA-aligned `2x36` (**`1.3244 s`**, `22.14x`, `3.3%` comm, `35.2%` faster than `1x72`) vs. flat `72x1` (**`0.6448 s`**, `45.48x`, **`853.03 M pts/s`**).

### Verified Vanda NVIDIA A40 GPU & Local Apple M5 Metal Readings
- **Vanda `GN-A40-044` 30-Iter Scaling (`gpu_benchmark.csv`, $N=11\text{M}$):** `K=8` **`3.0461 s`** (`5.70x` vs. 1C, `RMSE = 1.509e-12`), `K=16` **`5.7795 s`** (`5.66x`, `RMSE = 1.521e-12`), `K=32` **`12.4475 s`** (`4.69x`, `RMSE = 1.562e-12`).
- **Vanda `GN-A40-044` 50-Iter Validation (`kmeans_gpu_benchmark.o1563433`, $N=11\text{M}, K=8$):** Native CUDA **`5.0759 s`** (`108.36 M pts/s`, **`9.67x`** vs. serial `49.0870 s`) vs. OpenMP `-mp=gpu` `15.7554 s` (`34.91 M pts/s`) $\implies$ CUDA is **`3.10x` faster**.
- **Apple M5 Metal (`gpu_benchmark_apple_metal.csv`, $N=597,638, 30$ iters):** `K=8` `0.1189 s` (`4.76x`), `K=16` `0.1444 s` (`6.57x`), `K=32` `0.2090 s` (**`8.10x`** vs. 1C).

---

## Step 4: Plotting Figures & Building the Final PDF Report

```bash
MPLCONFIGDIR=/tmp /tmp/report_venv/bin/python3 scripts/generate_report_figures.py
typst compile report/DSA5208_Project3_Report.typ DSA5208_Project3_Report.pdf
```
