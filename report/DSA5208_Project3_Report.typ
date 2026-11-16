#set page(
  paper: "a4",
  margin: (top: 1.75cm, bottom: 1.75cm, left: 1.75cm, right: 1.75cm),
  header: align(right)[
    #text(size: 8.5pt, fill: rgb("#555555"))[
      *DSA5208: Scalable Distributed Computing for Data Science — Project 3 Report*
    ]
  ],
  footer: context [
    #set text(size: 8.5pt, fill: rgb("#555555"))
    #grid(
      columns: (1fr, 1fr),
      align(left)[Scalable Hybrid Parallel K-Means (`MPI + OpenMP + CUDA + Metal`)],
      align(right)[Page #counter(page).display("1 of 1", both: true)]
    )
  ]
)

#set text(font: ("New Computer Modern", "Times New Roman", "Georgia"), size: 9.6pt)
#set par(justify: true, leading: 0.54em)
#set heading(numbering: "1.1.")

#show heading.where(level: 1): it => block(above: 1.15em, below: 0.55em)[
  #set text(size: 11.8pt, weight: "bold", fill: rgb("#1b4f72"))
  #it
  #v(-0.4em)
  #line(length: 100%, stroke: 0.7pt + rgb("#1b4f72"))
]

#show heading.where(level: 2): it => block(above: 0.85em, below: 0.4em)[
  #set text(size: 10.3pt, weight: "bold", fill: rgb("#1f618d"))
  #it
]

// Title Block
#align(center)[
  #block(
    width: 100%,
    fill: rgb("#f4f8fb"),
    stroke: 1pt + rgb("#1b4f72"),
    radius: 4pt,
    inset: 11pt,
  )[
    #text(size: 13.8pt, weight: "bold", fill: rgb("#1b4f72"))[
      Scalable Hybrid Parallelization of Lloyd's K-Means Clustering (`MPI + OpenMP + CUDA + Metal`) on the Full 11-Million-Row UCI HIGGS Boson Dataset
    ]
    #v(0.4em)
    #text(size: 10.0pt, weight: "semibold")[
      DSA5208: Scalable Distributed Computing for Data Science (Project 3)
    ]
    #v(0.25em)
    #text(size: 8.8pt, fill: rgb("#333333"))[
      *Authors:* Nithillen Jayaseelan (`A0351473E`), Ashwin Sasikumar (`A0354333J`), Amitej Singh Datta (`A0352547Y`) \
      *Course Date:* November 2026
    ]
    #v(0.15em)
    #text(size: 8.8pt, fill: rgb("#333333"))[
      *Public GitHub Repository:* #link("https://github.com/Nithillen/dsa5208-project3-scalable-kmeans")[`https://github.com/Nithillen/dsa5208-project3-scalable-kmeans`]
    ]
  ]
]

#v(0.3em)

#block(
  fill: rgb("#fcfcfc"),
  stroke: (left: 3pt + rgb("#117864")),
  inset: (left: 10pt, top: 6pt, bottom: 6pt, right: 8pt)
)[
  *Executive Summary.* This report presents the design, parallel implementation, mathematical validation, and empirical performance analysis of Lloyd's K-Means clustering across four parallel computing backends: *(1) 1-Core Serial C++17*, *(2) Pure Distributed-Memory MPI*, *(3) Hybrid `MPI + OpenMP` (CPU shared-memory multithreading and GPU target offload)*, and *(4) Native GPU Acceleration (`MPI + CUDA` on NVIDIA A40 and `MPI + Apple Metal` on Apple M5)*. All primary cluster benchmarks execute on the *entire real-world UCI HIGGS Boson Dataset (ID \#280)* ($N = 11,000,000$ collision events $times$ $d = 28$ features $= 308,000,000$ IEEE-754 doubles) on the *NUS Vanda Supercomputer* (`CN-111`: 72-core dual-socket Intel Xeon Platinum 8452Y; `GN-A40-044`: NVIDIA A40 48 GB GPU). On Vanda, Pure MPI achieves a *$50.65 times$ strong-scaling speedup* ($79.1%$ parallel efficiency, $885.49 "M points/sec"$) at $P = 64$ physical cores, cutting 50-iteration runtime from $31.4583 "s"$ ($P=1$) to $0.6211 "s"$ with exact `0.000e+00` centroid parity and identical WCSS/SSE objective ($1.584590 times 10^8$). Eliminating shared-memory atomic contention via 64-byte cache-padded thread-private OpenMP accumulators speeds up 4-thread execution by *$13.52 times$* ($11.5415 "s"$ vs. $156.0935 "s"$), while aligning MPI ranks to NUMA sockets (`P=2, T=36`: $1.3244 "s"$) cuts cross-socket OpenMP runtime by $35.2%$ relative to `P=1, T=72` ($2.0441 "s"$), and flat `P=72, T=1` reaches $0.6448 "s"$ ($45.48 times$ speedup, $853.03 "M pts/s"$). Finally, offloading to an NVIDIA A40 GPU with `__shared__` memory centroid caching completes 50 iterations on all $11,000,000$ events in *$5.0759 "s"$* ($108.36 "M pts/s"$, *$9.67 times$* faster than 1-core serial C++ and *$3.10 times$* faster than OpenMP `-mp=gpu` target offload at $15.7554 "s"$), while complementary Apple Metal GPU shaders reach *$8.10 times$ speedup* on $K = 32$ clusters.
]

= Introduction to the Algorithm & Parallel Computing Terminology

== Mathematical Formulation of Lloyd's K-Means Clustering
Given a dataset $X = {bold(x)_1, bold(x)_2, dots, bold(x)_N} subset RR^d$ of $N$ observations each with $d$ continuous features, and a target number of clusters $K$, K-Means clustering partitions $X$ into $K$ disjoint clusters $C = {C_1, C_2, dots, C_K}$ with centroids $bold(mu)_1, dots, bold(mu)_K in RR^d$ to minimize the Within-Cluster Sum of Squares (WCSS / SSE) objective:
$ J(C, bold(mu)) = sum_(k=1)^K sum_(bold(x)_i in C_k) norm(bold(x)_i - bold(mu)_k)_2^2 = sum_(i=1)^N min_(k in {1,dots,K}) sum_(j=1)^d (x_(i,j) - mu_(k,j))^2 $

Because globally minimizing $J(C, bold(mu))$ is NP-hard, *Lloyd's Algorithm* (1982) alternates between two coupled phases across iterations $t = 0, 1, dots, T_("max")-1$:
1. *Expectation (Point Assignment) Step:* Each point $bold(x)_i$ is assigned to its nearest centroid in squared Euclidean distance, executing $N times K times d$ multiply-subtract-accumulate operations per iteration:
   $ z_i^((t)) = arg min_(k in {1, dots, K}) sum_(j=1)^d (x_(i,j) - mu_(k,j)^((t)))^2, quad forall i in {1, dots, N} $
2. *Maximization (Centroid Update) Step:* New centroids $bold(mu)_k^((t+1))$ are computed from the global cluster feature sums $bold(S)_k^((t)) in RR^d$ and integer cluster cardinalities $M_k^((t)) in ZZ_(>= 0)$:
   $ bold(S)_k^((t)) = sum_(i : z_i^((t)) = k) bold(x)_i, quad M_k^((t)) = sum_(i : z_i^((t)) = k) 1, quad bold(mu)_k^((t+1)) = bold(S)_k^((t)) / M_k^((t)) $
Termination occurs when the maximum Euclidean centroid shift satisfies $max_k norm(bold(mu)_k^((t+1)) - bold(mu)_k^((t)))_2 < epsilon$, or when a fixed iteration budget $T_("max")$ completes for controlled per-iteration timing benchmarks.

== Why Lloyd's K-Means Is Not Embarrassingly Parallel
Unlike embarrassingly parallel workloads (e.g., independent Monte Carlo trials) where workers execute completely independently until final aggregation, scalable K-Means introduces three coupled HPC challenges:
1. *Mandatory Per-Iteration Global Barrier Synchronization:* Evaluating assignments at iteration $t+1$ requires exact global centroids $bold(mu)^((t+1))$ aggregated across *all* distributed points. Every MPI rank must participate in blocking collective all-reductions (`MPI_Allreduce`) over $K times d$ doubles, $K$ counts, and scalar SSE at the end of *every single iteration*.
2. *Many-to-Few Shared-Memory & GPU Write Contention:* Although reads of $X$ and $bold(mu)^((t))$ are embarrassingly read-only, updating cluster sums maps $N = 11,000,000$ concurrent worker evaluations onto only $K = 8$ shared accumulator buckets. Naive atomic accumulation causes severe hardware *cache-line ping-pong* across CPU sockets and warp serialization on GPUs.
3. *NUMA Memory Bandwidth & Compute-to-Communication Ratio:* Because 28-dimensional squared Euclidean distance performs modest arithmetic per loaded byte, multi-socket cache locality (NUMA domain binding) and collective communication packing strongly govern real speedup at scale.

== Definitions of Parallel Metrics & Validation Terminology
Table 1 defines every performance and correctness metric used in this study (following Lecture 12 guidelines).

#figure(
  table(
    columns: (1.7fr, 4.3fr),
    inset: 4.8pt,
    align: (left, left),
    fill: (col, row) => if row == 0 { rgb("#1b4f72") } else if calc.odd(row) { rgb("#f8fbfd") } else { white },
    [#text(fill: white, weight: "bold")[Term / Symbol]], [#text(fill: white, weight: "bold")[Formal Definition & Mathematical Formula]],
    [*Strong Scaling*], [Measuring execution time $T(P)$ as processor count $P$ increases while *total dataset size $N$ stays fixed* ($N = 11,000,000$).],
    [*Weak Scaling*], [Measuring execution time $T(P, P dot N_0)$ as processor count $P$ increases while *work per processor $N_0 = N/P$ stays fixed* ($200,000$ points/rank).],
    [*Parallel Speedup $S(P)$*], [Ratio of 1-worker runtime $T(1)$ to $P$-worker runtime $T(P)$ on identical problem size: $S(P) = T(1) / T(P)$.],
    [*Strong Efficiency $E_("strong")(P)$*], [Fraction of linear speedup achieved per worker: $E_("strong")(P) = (S(P) / P) times 100% = (T(1) / (P dot T(P))) times 100%$.],
    [*Weak Efficiency $E_("weak")(P)$*], [Ratio of 1-worker time on size $N_0$ to $P$-worker time on size $P dot N_0$: $E_("weak")(P) = (T(1, N_0) / T(P, P dot N_0)) times 100%$.],
    [*Amdahl's & Gustafson's Laws*], [Strong-scaling bound $S(P) <= 1/(f_("seq") + (1-f_("seq"))/P)$ from sequential/comm overhead $f_("seq")$; scaled speedup $S_("scaled")(P) = f_("seq") + P(1-f_("seq"))$.],
    [*`MPI_Allreduce` (Separate vs Packed)*], [Tree-based $cal(O)(log_2 P)$ reduction summing local statistics across ranks. Packing sums, counts, and SSE into one array reduces multiple collective latencies to one.],
    [*Cache-Line Ping-Pong / False Sharing*], [Hardware MESI coherence invalidation caused when concurrent CPU threads write atomic updates or adjacent elements on the same 64-byte L1/L2 cache line.],
    [*On-Chip Shared Memory Cache*], [User-managed GPU SRAM (`__shared__` in CUDA, `threadgroup` in Apple Metal) shared by threads in a block to broadcast centroids without global DRAM stalls.],
    [*Max Abs Error ($L_infinity$) & $"RMSE"$*], [Centroid parity vs. serial reference $C^("ser")$: $E_("max") = max_(k,j) |C_(k,j)^("par") - C_(k,j)^("ser")|$ and $"RMSE" = sqrt(1/(K d) sum_(k,j)(C_(k,j)^("par") - C_(k,j)^("ser"))^2)$.]
  ),
  caption: [Definitions of all parallel performance metrics, hardware concepts, and validation terminology.]
)

= Dataset, Supercomputer Hardware & Serial Complexity Scaling

== Full 11,000,000-Row UCI HIGGS Boson Dataset Extraction & Caching
All primary experiments use the complete *UCI HIGGS Boson High-Energy Physics Dataset (ID \#280)* (`Baldi et al., Nature Communications, 2014`), simulating all $11,000,000$ proton–proton collisions ($52.99%$ weighted Higgs signal mean across clusters vs. $47.01%$ background, `0` skipped rows):
- *Input Cleaning (`scripts/prepare_higgs_data.py --all`):* Streams `HIGGS.csv.gz` ($2.62 "GB"$ compressed, $7.48 "GB"$ uncompressed, $11,000,000 times 29$ columns) and separates binary class labels (`data/higgs_labels.txt`, $11,000,000$ lines) from the 28 continuous physics features (`data/higgs_clean.csv`, $N = 11,000,000$ rows $times$ $d = 28$ features $= 308,000,000$ IEEE-754 doubles, occupying $2.464 "GB"$ in RAM).
- *Transparent Binary Cache (`src/kmeans_common.hpp`):* On the first full-dataset read ($8.32 "s"$ on Vanda), `load_csv_data` atomically writes a validated IEEE-754 binary image (`data/higgs_clean.csv.bin` with magic header + source byte-size guard), cutting subsequent multi-rank I/O load times across all ~45 benchmark runs by $>30 times$ while guaranteeing bitwise-identical input floats (`diff -s` verified).

== Experimental Supercomputer & Development Hardware Platforms
Table 2 summarizes all hardware platforms measured in this study.

#figure(
  table(
    columns: (1.5fr, 1.3fr, 2.2fr, 1.8fr),
    inset: 4.5pt,
    align: left,
    fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if calc.odd(r) { rgb("#f8fbfd") } else { white },
    [#text(fill: white, weight: "bold")[Platform]],
    [#text(fill: white, weight: "bold")[Node / Host]],
    [#text(fill: white, weight: "bold")[Processor / Accelerator Architecture]],
    [#text(fill: white, weight: "bold")[Compiler / Stack & Workload]],
    [*Vanda HPC CPU*], [`CN-111` (NUS Vanda)], [Dual-socket Intel Xeon Platinum 8452Y (2 NUMA nodes, 36 cores/socket $= 72$ physical cores, $240 "GB"$ RAM)], [`g++`/`mpicxx` (`nvhpc-hpcx-cuda12/26.1`); full $11,000,000$ rows],
    [*Vanda HPC GPU*], [`GN-A40-044` (Vanda)], [1$times$ NVIDIA A40 (`sm_86`, $46,068 "MiB"$ VRAM, Driver `575.57.08`) + 36 host CPU cores], [`nvcc` CUDA `12.9` & `nvc++ -mp=gpu`; full $11,000,000$ rows],
    [*Local Laptop GPU*], [Apple MacBook Air], [Apple M5 SoC (10 CPU cores, Unified Memory Metal GPU, $16 "GiB"$ RAM)], [`mpicxx` + `libomp` + Metal shaders; CI & $N=597,638$ Metal test]
  ),
  caption: [Compute platforms evaluated across Serial C++, Pure MPI, Hybrid `MPI + OpenMP`, CUDA, and Apple Metal.]
)

== Serial Baseline & Complexity Scaling on Full 11M Dataset (`src/kmeans_serial.cpp`)
Running `kmeans_serial` for 50 iterations ($K = 8, d = 28$) on all $11,000,000$ HIGGS events on Vanda node `CN-111` completes in *$49.0870 "seconds"$* ($981.74 "ms/iter"$, *$11.20 "M points/sec"$*, Final SSE $= 1.584590 times 10^8$, crossing convergence tolerance $epsilon = 10^(-5)$ at iteration 47 while completing all 50 iterations cleanly for exact fixed-iteration validation). Figure 1 and Tables 3–4 report the empirical serial scaling across $N in [100"k", 11"M"]$ (`results/serial_scaling_N.csv`) and $K in \{2, 4, 8, 16, 32\}$ on all $N = 11,000,000$ events (`results/serial_scaling_K.csv`).

#figure(
  image("figures/fig1_serial_scaling.png", width: 100%),
  caption: [Log-log runtime scaling of 1-core serial `kmeans_serial` on Vanda (`CN-111`, Intel Xeon Platinum 8452Y, $d=28, 30 "iterations"$): (a) Problem size $N$ ($100"k" dots.h 11"M"$, $K=8$); (b) Cluster count $K in \{2 dots.h 32\}$ on all $11,000,000$ rows.]
)

#grid(
  columns: (1.14fr, 0.86fr),
  gutter: 8pt,
  figure(
    {
      set text(size: 8.2pt)
      table(
        columns: (1.25fr, 1.35fr, 1.0fr, 0.95fr, 0.95fr),
        inset: 3.5pt,
        align: right,
        fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if calc.odd(r) { rgb("#f8fbfd") } else { white },
        [#text(fill: white, weight: "bold")[Points $N$]],
        [#text(fill: white, weight: "bold")[Floats $N d$]],
        [#text(fill: white, weight: "bold")[Time (s)]],
        [#text(fill: white, weight: "bold")[Iter (ms)]],
        [#text(fill: white, weight: "bold")[M pts/s]],
        [100,000], [2,800,000], [0.4079], [13.60], [7.35],
        [500,000], [14,000,000], [1.2805], [42.68], [11.71],
        [1,000,000], [28,000,000], [2.7431], [91.44], [10.94],
        [2,500,000], [70,000,000], [6.4020], [213.40], [11.72],
        [5,000,000], [140,000,000], [12.8151], [427.17], [11.70],
        [*11,000,000*], [*308,000,000*], [*28.0452*], [*934.84*], [*11.77*],
      )
    },
    caption: [Vanda serial scaling vs. $N$ (`serial_scaling_N.csv`, $K=8, 30 "iters"$).]
  ),
  figure(
    {
      set text(size: 8.2pt)
      table(
        columns: (0.65fr, 1.45fr, 1.05fr, 1.0fr, 0.95fr),
        inset: 3.5pt,
        align: right,
        fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if calc.odd(r) { rgb("#f8fbfd") } else { white },
        [#text(fill: white, weight: "bold")[$K$]],
        [#text(fill: white, weight: "bold")[Evals $N K$]],
        [#text(fill: white, weight: "bold")[Time (s)]],
        [#text(fill: white, weight: "bold")[Iter (ms)]],
        [#text(fill: white, weight: "bold")[M pts/s]],
        [2], [22,000,000], [10.5133], [350.44], [31.39],
        [4], [44,000,000], [15.8274], [527.58], [20.85],
        [8], [88,000,000], [28.1074], [936.91], [11.74],
        [16], [176,000,000], [60.0651], [2002.17], [5.49],
        [*32*], [*352,000,000*], [*112.5194*], [*3750.65*], [*2.93*],
      )
    },
    caption: [Vanda serial scaling vs. $K$ (`serial_scaling_K.csv`, $N=11"M", 30 "iters"$).]
  )
)

*Serial Scaling Takeaways:* From $N = 500,000$ to $N = 11,000,000$ ($22 times$ more collision events), serial runtime scales by $28.0452 / 1.2805 = 21.90 times$, locking steady-state single-core Xeon throughput between $10.94$ and $11.77 "M points/sec"$ across all $308 "million"$ out-of-cache floats and confirming exact $cal(O)(N)$ complexity. Varying $K$ from $2$ to $32$ ($16 times$ more centroids) increases runtime by $112.5194 / 10.5133 = 10.70 times$ as fixed centroid accumulation cost is amortized at low $K$ before distance math ($cal(O)(N K d)$) dominates for $K >= 8$ (nearly exact doubling from $K=8 -> 16 -> 32$: $28.11 "s" -> 60.07 "s" -> 112.52 "s"$).

= Pure MPI Parallelization on 72-Core Vanda (`src/kmeans_mpi.cpp`)

== Distributed-Memory Design & Communication Packing
In `src/kmeans_mpi.cpp`, rows are partitioned evenly via 1D row-block decomposition ($N_p = floor(N/P) + II(p < N mod P)$) using either `MPI_Scatterv` (with 64-bit send/recv chunking for multi-billion-element safety) or distributed chunked parallel I/O (`-io parallel`) directly from the binary cache. After local assignment on $N_p$ points, each iteration invokes collective reduction (`MPI_Allreduce`) over $K times d = 224$ `MPI_DOUBLE` sums (`1,792` bytes), $K = 8$ `MPI_UNSIGNED_LONG_LONG` counts (`64` bytes), and scalar WCSS (`8` bytes). We compare default 3-call separate reductions against `-allreduce packed` (`-p`), which packs sums, counts, and SSE into a single contiguous `233`-double collective buffer to eliminate $2/3$ of MPI tree-synchronization barriers.

== Strong Scaling & Impact of Communication on Full 11M HIGGS Rows
Figure 2 and Tables 5–6 show the official Vanda strong-scaling and collective comparison results across $P in {1, 2, 4, 8, 16, 32, 64}$ physical CPU cores on all $N = 11,000,000$ events (`results/mpi_strong_scaling.csv`, `results/mpi_allreduce_comparison.csv`).

#figure(
  image("figures/fig2_mpi_strong_scaling.png", width: 100%),
  caption: [Log-log Pure MPI Strong Scaling on NUS Vanda (`CN-111`, Intel Xeon Platinum 8452Y, $N = 11,000,000, d = 28, K = 8, 50 "iterations"$): (a) Total, Computation, and `MPI_Allreduce` Communication time across $P=1..64$; (b) Parallel Speedup $S(P)$ reaching $50.65 times$ ($79.1%$ efficiency) and $885.49 "M pts/s"$ at $P=64$.]
)

#figure(
  {
    set text(size: 8.0pt)
    table(
      columns: (0.45fr, 1.15fr, 0.85fr, 0.85fr, 0.85fr, 0.72fr, 0.82fr, 0.75fr, 0.75fr, 1.12fr, 1.32fr),
      inset: 3.6pt,
      align: right,
      fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if r == 7 { rgb("#e8f8f5") } else if calc.odd(r) { rgb("#f8fbfd") } else { white },
      [#text(fill: white, weight: "bold")[$P$]],
      [#text(fill: white, weight: "bold")[Pts / Rank]],
      [#text(fill: white, weight: "bold")[Total (s)]],
      [#text(fill: white, weight: "bold")[Max Comp (s)]],
      [#text(fill: white, weight: "bold")[Max Comm (s)]],
      [#text(fill: white, weight: "bold")[Comm %]],
      [#text(fill: white, weight: "bold")[M pts/s]],
      [#text(fill: white, weight: "bold")[$S(P)$]],
      [#text(fill: white, weight: "bold")[$E(P)$]],
      [#text(fill: white, weight: "bold")[Max Err]],
      [#text(fill: white, weight: "bold")[Final SSE]],
      [1], [11,000,000], [31.4583], [31.4576], [0.0005], [0.0%], [17.48], [1.00x], [100.0%], [`0.000e+00`], [`1.584590e+08`],
      [2], [5,500,000], [14.4035], [14.2815], [1.3279], [9.2%], [38.19], [2.18x], [109.2%], [`0.000e+00`], [`1.584590e+08`],
      [4], [2,750,000], [7.8956], [7.5634], [1.2932], [16.4%], [69.66], [3.98x], [99.6%], [`0.000e+00`], [`1.584590e+08`],
      [8], [1,375,000], [3.9428], [3.8821], [0.7755], [19.7%], [139.49], [7.98x], [99.7%], [`0.000e+00`], [`1.584590e+08`],
      [16], [687,500], [2.4039], [2.2922], [0.7380], [30.7%], [228.80], [13.09x], [81.8%], [`0.000e+00`], [`1.584590e+08`],
      [32], [343,750], [1.3659], [1.2615], [0.5703], [41.8%], [402.68], [23.03x], [72.0%], [`0.000e+00`], [`1.584590e+08`],
      [*64*], [*171,875*], [*0.6211*], [*0.6191*], [*0.1323*], [*21.3%*], [*885.49*], [*50.65x*], [*79.1%*], [*`0.000e+00`*], [*`1.584590e+08`*],
    )
  },
  caption: [Measured NUS Vanda Pure MPI Strong Scaling (`results/mpi_strong_scaling.csv`, $N=11,000,000, d=28, K=8, 50 "iterations"$).]
)

#figure(
  table(
    columns: (0.7fr, 1.3fr, 1.3fr, 1.3fr, 1.3fr, 1.3fr, 1.3fr, 1.3fr),
    inset: 4.0pt,
    align: right,
    fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if calc.odd(r) { rgb("#f8fbfd") } else { white },
    [#text(fill: white, weight: "bold")[Ranks $P$]],
    [#text(fill: white, weight: "bold")[Separate Total (s)]],
    [#text(fill: white, weight: "bold")[Separate Comm (s)]],
    [#text(fill: white, weight: "bold")[Separate Comm %]],
    [#text(fill: white, weight: "bold")[Packed Total (s)]],
    [#text(fill: white, weight: "bold")[Packed Comm (s)]],
    [#text(fill: white, weight: "bold")[Packed Comm %]],
    [#text(fill: white, weight: "bold")[Max Err / SSE Parity]],
    [1], [27.9578], [0.0004], [0.0%], [27.1864], [0.0005], [0.0%], [`0.000e+00`],
    [2], [12.7625], [0.2708], [2.1%], [12.8766], [0.2505], [1.9%], [`0.000e+00`],
    [4], [7.1514], [0.6237], [8.7%], [7.3244], [0.8413], [11.5%], [`0.000e+00`],
    [8], [4.8595], [1.6457], [33.9%], [*3.8170*], [*0.6824*], [*17.9%*], [`0.000e+00`],
    [16], [2.2522], [0.6979], [31.0%], [2.2817], [0.7215], [31.6%], [`0.000e+00`],
    [32], [1.2595], [0.4586], [36.4%], [1.2709], [0.4729], [37.2%], [`0.000e+00`],
    [64], [0.6579], [0.1758], [26.7%], [*0.6308*], [*0.1455*], [*23.1%*], [`0.000e+00`],
  ),
  caption: [Separate (3-call) vs. Packed (1-call) `MPI_Allreduce` comparison on Vanda (`results/mpi_allreduce_comparison.csv`, $N=11"M"$).]
)

*Analysis of Vanda MPI Scaling & Communication Behavior:*
1. *Superlinear Cache Scaling ($P=2 -> 8$):* At $P = 2$, runtime drops from $31.4583 "s"$ to $14.4035 "s"$ (*$2.18 times$ speedup, $109.2%$ efficiency*) as splitting $2.464 "GB"$ across both NUMA sockets doubles aggregate L3 cache and memory channels. Near-ideal scaling holds through $P = 4$ ($3.98 times$, $99.6%$) and $P = 8$ ($7.98 times$, $99.7%$).
2. *Amdahl's Law Collective Saturation ($P=16 -> 64$):* As per-rank data shrinks to $343,750$ points ($P=32$), per-rank compute falls to $1.2615 "s"$ while blocking `MPI_Allreduce` barrier waiting time (`comp` + OS skew across ranks) rises to $0.5703 "s"$ (*$41.8%$ of total runtime*). At $P = 64$, balanced division yields $0.6211 "s"$ ($12.42 "ms/iter"$), achieving *$50.65 times$ speedup ($79.1%$ strong efficiency)* and *$885.49 "M points/sec"$* aggregate throughput.
3. *Packed Single-Call `MPI_Allreduce`:* Under transient rank arrival skew (such as $P = 8$, where 3 separate barriers magnified wait jitter to $1.6457 "s"$ / $33.9%$), packing all statistics into 1 `MPI_Allreduce` reduces communication wait by *$2.41 times$* to $0.6824 "s"$ ($17.9%$), speeding up total execution by $21.5%$ ($3.8170 "s"$ vs. $4.8595 "s"$), and also delivers the fastest packed MPI runtime at $P = 64$ ($0.6308 "s"$ vs. $0.6579 "s"$).
4. *Per-Rank `MPI_MAX` Timing Accounting:* In Tables 5, 7, and 9, `Max Comp (s)` and `Max Comm (s)` record independent `MPI_Reduce(..., MPI_MAX)` reductions across all ranks over the 50 iterations (the slowest rank's compute phase plus the longest rank's blocking barrier-wait + `MPI_Allreduce` time, with `Comm % = 100 * Max Comm / Total`). Consequently, `Max Comp + Max Comm` can exceed wall-clock `Total (s)` when different ranks experience peak compute vs. barrier arrival skew.

== Pure MPI Weak Scaling on Vanda ($N = 200"k" -> 12.8"M"$ Points)
To verify Gustafson's Law, Figure 3 and Table 7 hold $N_0 = 200,000$ synthetic Gaussian points per rank constant across $P = 1 dots.h 64$ ranks (`results/mpi_weak_scaling.csv`), scaling total problem size from $200,000$ to $12,800,000$ points.

#figure(
  image("figures/fig3_mpi_weak_scaling.png", width: 100%),
  caption: [Log-log Pure MPI Weak Scaling on NUS Vanda (`CN-111`, $N_0 = 200,000$ points/rank, $d=28, K=8, 30 "iterations"$): (a) Runtime breakdown ($200"k" -> 12.8"M"$ points); (b) Aggregate throughput scaling from $20.11 "M/s"$ ($P=1$) to $699.36 "M/s"$ ($P=64$).]
)

#figure(
  {
    set text(size: 8.3pt)
    table(
      columns: (0.6fr, 1.25fr, 1.0fr, 0.9fr, 0.9fr, 0.9fr, 0.85fr, 1.0fr, 1.6fr),
      inset: 3.8pt,
      align: right,
      fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if calc.odd(r) { rgb("#f8fbfd") } else { white },
      [#text(fill: white, weight: "bold")[Ranks $P$]],
      [#text(fill: white, weight: "bold")[Total Points $N$]],
      [#text(fill: white, weight: "bold")[Pts / Rank]],
      [#text(fill: white, weight: "bold")[Total (s)]],
      [#text(fill: white, weight: "bold")[Max Comp (s)]],
      [#text(fill: white, weight: "bold")[Max Comm (s)]],
      [#text(fill: white, weight: "bold")[Comm %]],
      [#text(fill: white, weight: "bold")[Weak Eff]],
      [#text(fill: white, weight: "bold")[Throughput]],
      [1], [200,000], [200,000], [0.2983], [0.2982], [0.0000], [0.0%], [100.0%], [20.11 M pts/s],
      [2], [400,000], [200,000], [0.2820], [0.2816], [0.0079], [2.8%], [*105.8%*], [42.56 M pts/s],
      [4], [800,000], [200,000], [0.4223], [0.4213], [0.1438], [34.1%], [70.6%], [56.83 M pts/s],
      [8], [1,600,000], [200,000], [0.4253], [0.4246], [0.1510], [35.5%], [70.1%], [112.85 M pts/s],
      [16], [3,200,000], [200,000], [0.4788], [0.4763], [0.2021], [42.2%], [62.3%], [200.50 M pts/s],
      [32], [6,400,000], [200,000], [0.5832], [0.5822], [0.3049], [52.3%], [51.1%], [329.24 M pts/s],
      [*64*], [*12,800,000*], [*200,000*], [*0.5491*], [*0.5360*], [*0.2068*], [*37.7%*], [*54.3%*], [*699.36 M pts/s*],
    )
  },
  caption: [Vanda Pure MPI Weak Scaling (`results/mpi_weak_scaling.csv`, fixed $200,000 "pts/rank", 30 "iterations"$).]
)

Expanding the total workload $64 times$ from $200,000$ to $12,800,000$ collision points increases wall-clock runtime by less than $1.85 times$ ($0.2983 "s" -> 0.5491 "s"$), maintaining $54.3%$ weak efficiency and scaling throughput *$34.8 times$* from $20.11 "M pts/s"$ ($P=1$) to *$699.36 "M pts/s"$* ($P=64$).

= Hybrid `MPI + OpenMP` Design Choices & NUMA Scaling (`src/kmeans_omp.cpp`)

== OpenMP Accumulation Modes & Loop Scheduling on 11M Rows
In `src/kmeans_omp.cpp` (`MPI_THREAD_FUNNELED`), we evaluate three design choices on Vanda node `CN-111` ($P=1$ rank, $T=4$ threads, $N=11,000,000, K=8, 50 "iterations"$, `results/omp_design_choices.csv`):
1. *Cache-Line Padded Thread-Private Buffers vs. `#pragma omp atomic`:* In `private` mode, each thread accumulates into its own row inside a single contiguous buffer padded to 64-byte L1 cache-line boundaries (`stride_sums = ((K * d + 7) / 8) * 8`) with `#pragma omp for schedule(runtime) nowait`, merging across $T$ buffers once per iteration with zero atomics.
2. *Dynamic/Guided vs. Static Scheduling:* `static` divides rows equally, whereas `dynamic,1024` and `guided` rebalance work chunks across cores dynamically.
3. *Standalone OpenMP vs. MPI+OpenMP (`P=1`):* Compiling `-DSTANDALONE_OMP` eliminates MPI runtime overhead while sharing identical `#pragma omp simd` vectorization.

#figure(
  image("figures/fig4_omp_design_choices.png", width: 100%),
  caption: [Impact of OpenMP parallel design decisions on full 11,000,000-row UCI HIGGS (`CN-111`, $P=1, T=4, K=8, 50 "iterations"$): (a) Log-scale runtime showing `atomic` contention ($156.0935 "s"$) vs. cache-padded `private` (`11.5415 s` with `guided`); (b) Log-scale throughput ($13.54 times$ higher throughput).]
)

#figure(
  table(
    columns: (1.9fr, 1.0fr, 1.0fr, 1.0fr, 1.2fr, 1.1fr, 1.1fr),
    inset: 4.2pt,
    align: (left, center, right, right, right, right, right),
    fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if r == 3 { rgb("#e8f8f5") } else if r == 4 { rgb("#fdedec") } else { white },
    [#text(fill: white, weight: "bold")[Accumulation Mode]],
    [#text(fill: white, weight: "bold")[Schedule]],
    [#text(fill: white, weight: "bold")[Total (s)]],
    [#text(fill: white, weight: "bold")[Iter (ms)]],
    [#text(fill: white, weight: "bold")[Throughput]],
    [#text(fill: white, weight: "bold")[Speedup vs Atomic]],
    [#text(fill: white, weight: "bold")[Max Err]],
    [Cache-Padded Private (MPI `P=1`)], [`static`], [14.1680], [283.36], [38.82 M/s], [11.02x], [`0.000e+00`],
    [Cache-Padded Private (MPI `P=1`)], [`dynamic`], [11.5427], [230.85], [47.65 M/s], [13.52x], [`0.000e+00`],
    [*Cache-Padded Private (MPI `P=1`)*], [*`guided`*], [*11.5415*], [*230.83*], [*47.65 M/s*], [*13.52x*], [*`0.000e+00`*],
    [Shared Atomic (`#pragma omp atomic`)], [`static`], [156.0935], [3121.87], [3.52 M/s], [1.00x], [`0.000e+00`],
    [Standalone OpenMP (No MPI)], [`static`], [14.6160], [292.32], [37.63 M/s], [10.68x], [`0.000e+00`],
  ),
  caption: [OpenMP parallel design decisions on full 11,000,000 HIGGS rows (`results/omp_design_choices.csv`, $P=1, T=4$).]
)

*Why Cache-Padded Private Accumulation Wins by $13.52 times$:* In `atomic` mode, 4 Xeon cores execute $11,000,000 times 29 = 319 "million"$ atomic read-modify-write operations *per iteration* on just $224$ shared array elements, triggering relentless MESI cache-line invalidations that inflate runtime to *$156.0935 "s"$* ($3,121.87 "ms/iter"$, $3.18 times$ slower than 1-core serial!). Padded thread-private buffers cut runtime to $11.5415 "s"$ (*$13.52 times$ faster wall-clock time*, *$13.54 times$ higher throughput*), and `guided` / `dynamic` scheduling beat `static` by $18.5%$ ($11.5415 "s"$ vs. $14.1680 "s"$) by smoothing OS scheduling jitter across physical cores.

== Full 72-Core Hybrid `MPI Ranks (P) x OpenMP Threads (T)` Grid (`results/hybrid_mpi_omp_grid.csv`)
Figure 5 and Table 9 present all 21 measured $(P, T)$ hybrid combinations across $C = P times T in {1, 8, 16, 36, 72}$ physical cores on dual-socket Vanda node `CN-111` for $N = 11,000,000, K = 8, 50 "iterations"$.

#figure(
  image("figures/fig5_hybrid_mpi_omp_grid.png", width: 100%),
  caption: [Hybrid `MPI + OpenMP` topology scaling across 72 cores on Vanda (`CN-111`, $N=11,000,000, K=8, 50 "iterations"$): (a) Log-log runtime comparing Pure OpenMP (`1xC`), NUMA-aligned Hybrid (`2xT`), and Flat MPI (`Cx1`); (b) Full 72-core configuration trade-off.]
)

#figure(
  table(
    columns: (0.7fr, 0.55fr, 0.55fr, 0.95fr, 0.95fr, 0.9fr, 0.85fr, 0.9fr, 1.0fr, 0.85fr, 0.85fr, 1.0fr),
    inset: 3.4pt,
    align: right,
    fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if r == 16 or r == 21 { rgb("#e8f8f5") } else if calc.odd(r) { rgb("#f8fbfd") } else { white },
    [#text(fill: white, weight: "bold")[Cores]],
    [#text(fill: white, weight: "bold")[$P$]],
    [#text(fill: white, weight: "bold")[$T$]],
    [#text(fill: white, weight: "bold")[Total (s)]],
    [#text(fill: white, weight: "bold")[Max Comp (s)]],
    [#text(fill: white, weight: "bold")[Max Comm (s)]],
    [#text(fill: white, weight: "bold")[Comm %]],
    [#text(fill: white, weight: "bold")[Iter (ms)]],
    [#text(fill: white, weight: "bold")[M pts/s]],
    [#text(fill: white, weight: "bold")[$S$]],
    [#text(fill: white, weight: "bold")[$E$]],
    [#text(fill: white, weight: "bold")[Max Err]],
    [1], [1], [1], [29.3235], [29.3227], [0.0005], [0.0%], [586.47], [18.76], [1.00x], [100.0%], [`0.000e+00`],
    [8], [1], [8], [10.7194], [10.7190], [0.0002], [0.0%], [214.39], [51.31], [2.74x], [34.2%], [`0.000e+00`],
    [8], [8], [1], [4.3865], [4.2980], [0.6348], [14.5%], [87.73], [125.38], [6.68x], [83.6%], [`0.000e+00`],
    [16], [1], [16], [7.3621], [7.3618], [0.0002], [0.0%], [147.24], [74.71], [3.98x], [24.9%], [`0.000e+00`],
    [16], [2], [8], [5.2241], [5.1241], [0.2322], [4.4%], [104.48], [105.28], [5.61x], [35.1%], [`0.000e+00`],
    [16], [4], [4], [5.2148], [5.2093], [1.9061], [36.6%], [104.30], [105.47], [5.62x], [35.1%], [`0.000e+00`],
    [16], [8], [2], [4.2113], [4.1192], [1.1634], [27.6%], [84.23], [130.60], [6.96x], [43.5%], [`0.000e+00`],
    [16], [16], [1], [2.6927], [2.6358], [0.9290], [34.5%], [53.85], [204.26], [10.89x], [68.1%], [`0.000e+00`],
    [36], [1], [36], [3.4593], [3.4589], [0.0002], [0.0%], [69.19], [158.99], [8.48x], [23.5%], [`0.000e+00`],
    [36], [2], [18], [2.3079], [2.2916], [0.1264], [5.5%], [46.16], [238.31], [12.71x], [35.3%], [`0.000e+00`],
    [36], [4], [9], [2.8700], [2.8690], [1.3679], [47.7%], [57.40], [191.64], [10.22x], [28.4%], [`0.000e+00`],
    [36], [9], [4], [2.5158], [2.4758], [1.0486], [41.7%], [50.32], [218.62], [11.66x], [32.4%], [`0.000e+00`],
    [36], [18], [2], [2.7040], [2.6253], [1.3904], [51.4%], [54.08], [203.40], [10.84x], [30.1%], [`0.000e+00`],
    [36], [36], [1], [1.1727], [1.0564], [0.3586], [30.6%], [23.45], [469.01], [25.01x], [69.5%], [`0.000e+00`],
    [72], [1], [72], [2.0441], [2.0430], [0.0003], [0.0%], [40.88], [269.07], [14.35x], [19.9%], [`0.000e+00`],
    [*72*], [*2*], [*36*], [*1.3244*], [*1.3222*], [*0.0431*], [*3.3%*], [*26.49*], [*415.29*], [*22.14x*], [*30.8%*], [*`0.000e+00`*],
    [72], [4], [18], [1.9089], [1.9070], [0.7734], [40.5%], [38.18], [288.12], [15.36x], [21.3%], [`0.000e+00`],
    [72], [8], [9], [1.4987], [1.4906], [0.3309], [22.1%], [29.97], [366.99], [19.57x], [27.2%], [`0.000e+00`],
    [72], [18], [4], [1.5197], [1.5128], [0.5723], [37.7%], [30.39], [361.92], [19.30x], [26.8%], [`0.000e+00`],
    [72], [36], [2], [1.3566], [1.3354], [0.6123], [45.1%], [27.13], [405.42], [21.62x], [30.0%], [`0.000e+00`],
    [*72*], [*72*], [*1*], [*0.6448*], [*0.6430*], [*0.1479*], [*22.9%*], [*12.90*], [*853.03*], [*45.48x*], [*63.2%*], [*`0.000e+00`*],
  ),
  caption: [Full 72-core Hybrid `MPI + OpenMP` scaling grid on NUS Vanda (`results/hybrid_mpi_omp_grid.csv`, $N=11,000,000, K=8, 50 "iters"$).]
)

*NUMA & Topology Analysis:* Notice two striking supercomputer insights from Table 9:
1. *NUMA Socket Alignment (`1x72` vs. `2x36`):* On dual-socket `CN-111` (36 cores per NUMA node), pure OpenMP (`1x72`, $2.0441 "s"$) allocates `local_data` in one socket's memory (`first-touch` serial load by Rank 0), forcing Socket 1 threads to fetch points across the UPI interconnect and merging $72$ thread-private buffers serially. Placing *one MPI rank per NUMA socket* (`2x36`, $1.3244 "s"$) splits `local_data` cleanly across local DRAM controllers with only $3.3%$ MPI overhead ($0.0431 "s"$), cutting runtime by *$35.2%$ ($1.54 times$ speedup over `1x72`)*! The identical NUMA win appears at 36 cores (`2x18`: $2.3079 "s"$ vs. `1x36`: $3.4593 "s"$).
2. *First-Touch Memory & Serial Thread-Reduction Avoided by `72x1`:* At 72 cores, flat MPI (`72x1`) achieves *$0.6448 "s"$* ($45.48 times$ speedup, $63.2%$ efficiency, *$853.03 "M pts/s"$*, or *$76.13 times$ faster than 1-core `kmeans_serial` at $49.0870 "s"$*) because each rank owns local pages in its core's closest LLC slice and tree-based `MPI_Allreduce` merges all 72 ranks in $cal(O)(log_2 P)$ steps rather than serial loop reduction.

= Hardware GPU Acceleration (`MPI + CUDA`, OpenMP `-mp=gpu`, `MPI + Metal`)

== GPU Parallel Architecture & Shared-Memory Bounds Guard
1. *`src/kmeans_cuda.cu` (`MPI + CUDA` on Vanda NVIDIA A40):* Uses 256-thread blocks (`cudaMalloc`/`cudaMemcpy` of the $2.464 "GB"$ feature matrix staged once outside `t_start`, analogous to CPU CSV/binary loading, so `Total Time` captures iterative centroid uploads, CUDA kernel compute, and reduction downloads) and dynamically caches both centroids (`K*d` doubles) and block-local partial sums/counts inside on-chip `__shared__` memory whenever `2*K*d*8 + K*8 <= cudaDevAttrMaxSharedMemoryPerBlock`, falling back cleanly to global DRAM for huge $K$ (`Test Suite 3` verified at $K=40, d=32$). Our `atomicAddDouble` wrapper invokes native Ampere hardware `atomicAdd(double*, double)` on `sm_60+` (`-arch=sm_86`) and an `atomicCAS` loop fallback on older architectures, avoiding CUDA 12.9 header overload conflicts.
2. *`src/kmeans_omp.cpp -g` (OpenMP GPU Target Offload):* Compiles with `nvc++ -mp=gpu` using `#pragma omp target teams distribute parallel for` for portable directive-based GPU offloading, while host OpenMP benchmarks default to close core pinning (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`).
3. *`src/kmeans_metal_gpu.mm` (`MPI + Apple Metal GPU Shader`):* Maps unified host/GPU buffers (`MTLResourceStorageModeShared`, zero PCIe copy overhead) on Apple Silicon M5, caches centroids in `threadgroup float s_centroids[1024]`, and executes fused multiply-add (`fma`) FP32 distance evaluations across 256-thread threadgroups with host FP64 accumulation.

== Empirical NVIDIA A40 (`N = 11M`) & Apple M5 Metal GPU Results
Figure 6 and Tables 10–12 distinguish two complementary evaluation protocols: *30-iteration cluster scaling* across $K in \{8, 16, 32\}$ (Table 10 for Vanda A40; Table 12 for Apple M5 Metal) and the *50-iteration full convergence-budget comparison* at $K = 8$ across all Vanda CPU and GPU backends (Table 11).

#figure(
  image("figures/fig6_gpu_scaling.png", width: 100%),
  caption: [Vanda NVIDIA A40 GPU Acceleration on full UCI HIGGS ($N=11,000,000, d=28$): (a) Log-log scaling across $K in {8, 16, 32}$ ($30 "iters"$) vs. 1-core and 4-core Xeon CPUs; (b) Complete 50-iteration runtime comparison across Vanda execution backends ($K=8$, native CUDA achieves $9.67 times$ speedup over Serial C++ and $3.10 times$ over OpenMP GPU offload).]
)

#figure(
  table(
    columns: (0.6fr, 1.3fr, 1.0fr, 1.0fr, 1.0fr, 1.0fr, 1.0fr, 1.0fr, 1.1fr, 1.1fr),
    inset: 3.8pt,
    align: right,
    fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if r == 1 { rgb("#e8f8f5") } else if calc.odd(r) { rgb("#f8fbfd") } else { white },
    [#text(fill: white, weight: "bold")[$K$]],
    [#text(fill: white, weight: "bold")[GPU Device]],
    [#text(fill: white, weight: "bold")[1C CPU (s)]],
    [#text(fill: white, weight: "bold")[4C OMP (s)]],
    [#text(fill: white, weight: "bold")[GPU Tot (s)]],
    [#text(fill: white, weight: "bold")[GPU Kern (s)]],
    [#text(fill: white, weight: "bold")[Speedup vs 1C]],
    [#text(fill: white, weight: "bold")[Speedup vs 4C]],
    [#text(fill: white, weight: "bold")[M pts/s]],
    [#text(fill: white, weight: "bold")[Centroid RMSE]],
    [*8*], [NVIDIA A40], [17.3726], [5.1064], [*3.0461*], [3.0453], [*5.70x*], [1.68x], [*108.33*], [`1.509e-12`],
    [16], [NVIDIA A40], [32.7197], [10.1117], [5.7795], [5.7787], [5.66x], [*1.75x*], [57.10], [`1.521e-12`],
    [32], [NVIDIA A40], [58.3357], [16.0496], [12.4475], [12.4466], [4.69x], [1.29x], [26.51], [`1.562e-12`],
  ),
  caption: [30-iteration Vanda NVIDIA A40 CUDA GPU scaling across $K in \{8, 16, 32\}$ on all 11,000,000 HIGGS rows (`results/gpu_benchmark.csv`).]
)

#grid(
  columns: (0.98fr, 1.02fr),
  gutter: 8pt,
  figure(
    {
      set text(size: 8.2pt)
      table(
        columns: (1.9fr, 0.95fr, 1.05fr, 0.9fr),
        inset: 3.6pt,
        align: (left, right, right, right),
        fill: (c, r) => if r == 0 { rgb("#117864") } else if r == 4 { rgb("#e8f8f5") } else { white },
        [#text(fill: white, weight: "bold")[Vanda 50-Iter Backend ($N=11"M", K=8$)]],
        [#text(fill: white, weight: "bold")[Time (s)]],
        [#text(fill: white, weight: "bold")[M pts/s]],
        [#text(fill: white, weight: "bold")[Speedup]],
        [1-Core Serial C++ (`CN-111`)], [49.0870], [11.20], [1.00x],
        [OpenMP CPU SIMD (`P=1, T=1`)], [29.3235], [18.76], [1.67x],
        [OpenMP GPU Target (`-mp=gpu`, A40)], [15.7554], [34.91], [3.12x],
        [*Native CUDA (`__shared__`, NVIDIA A40)*], [*5.0759*], [*108.36*], [*9.67x*],
      )
    },
    caption: [50-iteration Vanda backend comparison at fixed $K=8$ ($N=11"M"$).]
  ),
  figure(
    {
      set text(size: 8.2pt)
      table(
        columns: (0.48fr, 0.98fr, 0.98fr, 0.98fr, 0.85fr, 1.35fr),
        inset: 3.6pt,
        align: right,
        fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if r == 3 { rgb("#e8f8f5") } else { white },
        [#text(fill: white, weight: "bold")[$K$]],
        [#text(fill: white, weight: "bold")[1C CPU (s)]],
        [#text(fill: white, weight: "bold")[4C OMP (s)]],
        [#text(fill: white, weight: "bold")[Metal Tot (s)]],
        [#text(fill: white, weight: "bold")[vs 1C]],
        [#text(fill: white, weight: "bold")[RMSE (FP32)]],
        [8], [0.5657], [0.1769], [0.1189], [4.76x], [`1.671e-04`],
        [16], [0.9493], [0.3007], [0.1444], [6.57x], [`1.029e-04`],
        [*32*], [*1.6926*], [*0.5456*], [*0.2090*], [*8.10x*], [`5.296e-04`],
      )
    },
    caption: [30-iteration Apple M5 Metal GPU (`gpu_benchmark_apple_metal.csv`, $N=597,638$).]
  )
)

*GPU Architectural Trade-Off Discussion:* On Vanda's NVIDIA A40 (`N = 11,000,000`), native CUDA (`5.0759 s`, $108.36 "M pts/s"$, *$9.67 times$ vs. Serial*) beats compiler-generated OpenMP target offload (`15.7554 s`, $34.91 "M pts/s"$) by *$3.10 times$* because explicit CUDA `__shared__` memory caches centroids on-chip and coalesces warp reads. Crucially, notice two distinct scaling trends across Table 10 and Table 12: in CUDA FP64 (`kmeans_cuda.cu`), intra-block shared atomic accumulation over $K times d$ doubles slightly serializes warps at larger $K$ ($5.70 times$ down to $4.69 times$ at $K=32$), whereas decoupling assignment on the GPU shader from CPU thread-private reduction in Apple Metal lets speedup climb monotonically with arithmetic intensity from *$4.76 times$ ($K=8$)* to *$6.57 times$ ($K=16$)* and *$8.10 times$ ($K=32$)* (`2.61x` over 4-core CPU).

= Mathematical Validation & Runtime Execution Proof

== Validation That Parallel Implementations Preserve Mathematical Correctness
Every parallel backend compares its converged centroids and WCSS/SSE objective against the serial baseline (`results/centroids_serial.txt`), supplemented by our 16-case automated test harness (`scripts/test_correctness.py`). Table 13 summarizes the validation accuracy across all architectures.

#figure(
  table(
    columns: (2.2fr, 1.5fr, 1.2fr, 1.3fr, 1.6fr),
    inset: 4.2pt,
    align: (left, center, right, right, center),
    fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if calc.odd(r) { rgb("#f8fbfd") } else { white },
    [#text(fill: white, weight: "bold")[Parallel Implementation]],
    [#text(fill: white, weight: "bold")[Benchmark Configuration]],
    [#text(fill: white, weight: "bold")[Max Abs Error $L_infinity$]],
    [#text(fill: white, weight: "bold")[Final WCSS (SSE)]],
    [#text(fill: white, weight: "bold")[Validation Status]],
    [Automated Test Suite (`test_correctness.py`)], [16 Tests (Synth + HIGGS + $K=40$)], [`0.000` – `1.000e-12`], [`5.375590e+07` / `7.216756e+05`], [*16/16 PASSED*],
    [Vanda Pure MPI (`kmeans_mpi.cpp`)], [$P = 1..64$ ($N=11"M"$), Packed/Sep], [`0.000e+00`], [`1.584590e+08`], [*PASSED (Exact Match)*],
    [Vanda Hybrid `MPI + OpenMP` (`kmeans_omp`)], [21 Grids ($1..72$ cores) + Designs], [`0.000e+00`], [`1.584590e+08`], [*PASSED (Exact Match)*],
    [Vanda NVIDIA A40 CUDA (`kmeans_cuda.cu`)], [$K=8, 16, 32$ ($N=11"M", 30 "it"$)], [RMSE `1.509e-12`–`1.562e-12`], [`1.584590e+08` / `1.441739e+08` / `1.318415e+08`], [*PASSED (FP64 Match)*],
    [Apple M5 Metal GPU (`kmeans_metal_gpu`)], [$K=8, 16, 32$ ($N=597"k"$, FP32)], [RMSE `1.029e-04`–`5.296e-04`], [`8.744180e+06` – `7.138754e+06`], [*PASSED (FP32 Match)*],
  ),
  caption: [Numerical validation of parallel converged centroids and WCSS objective against serial baselines.]
)

*(Provenance Notes on Raw Committed Vanda Logs: (1) In `kmeans_cpu_benchmarks.o1563432`, Step 2 ran `test_correctness.py` prior to adding our `has_cuda_gpu()` CPU-node guard, causing test 8 to abort on CPU-only `CN-111` while Steps 3–5 completed; `test_correctness.py` is now patched and passes `16/16` locally. (2) In `kmeans_gpu_benchmark.o1563433`, Step 2 validated CUDA against a fresh matching-iteration serial baseline at `1.509e-12` RMSE, while Step 3 read a stale 597k `centroids_serial.txt` (`Ref SSE: 8.742847e+06`) on `/scratch` before CPU job `1563432` overwrote it; `scripts/vanda_gpu_benchmarks.pbs` is now patched to regenerate the 11M baseline before Step 2, and both `kmeans_cuda` and `kmeans_omp_gpu` match the exact 50-iteration 11M serial SSE of `1.584590e+08`.)*

== Verbatim Runtime Terminal Captures (NUS Vanda HPC & Local Workstation)
Terminal Captures 1–3 preserve direct verbatim runtime execution logs from Vanda CPU job `1563432` (`CN-111`), Vanda GPU job `1563433` (`GN-A40-044`), and local Apple M5 validation.

#figure(
  image("figures/screenshot_vanda_cpu_72core.png", width: 96%),
  kind: "screenshot",
  supplement: [Terminal Capture],
  caption: [Runtime terminal capture of NUS Vanda job `kmeans_cpu_benchmarks.o1563432` on dual-socket Xeon Platinum 8452Y node `CN-111` executing all $N = 11,000,000$ UCI HIGGS rows across 72 cores.]
)

#figure(
  image("figures/screenshot_vanda_gpu_a40.png", width: 96%),
  kind: "screenshot",
  supplement: [Terminal Capture],
  caption: [Runtime terminal capture of NUS Vanda GPU job `kmeans_gpu_benchmark.o1563433` on `GN-A40-044` (`nvidia-smi` NVIDIA A40 48 GB) showing $K in \{8, 16, 32\}$ scaling and 50-iteration CUDA (`5.0759 s`) vs. OpenMP `-mp=gpu` (`15.7554 s`) on $11,000,000$ rows.]
)

#figure(
  image("figures/screenshot_local_validation.png", width: 96%),
  kind: "screenshot",
  supplement: [Terminal Capture],
  caption: [Runtime terminal capture on Apple Silicon M5 executing `python3 scripts/test_correctness.py` (16/16 PASSED) and `kmeans_metal_gpu` benchmarks (`4.76x` to `8.10x` speedup).]
)

= Physical Conclusions on the Full 11,000,000-Row UCI HIGGS Dataset

After unsupervised convergence over all $N = 11,000,000$ unlabeled 28-dimensional collider events into $K = 8$ clusters (`kmeans_cpu_benchmarks.o1563432`), `analyze_higgs_clusters` evaluates the converged cluster assignments against ground-truth signal/background labels in `data/higgs_labels.txt` (overall $11,000,000$-event weighted mean: $52.99%$ Higgs boson signal vs. $47.01%$ top-quark pair $t overline(t)$ background). Figure 7 and Table 14 report the exact 11,000,000-event physical profiles.

#figure(
  image("figures/fig7_higgs_physics_clusters.png", width: 100%),
  caption: [Physical cluster profiles discovered on all $N = 11,000,000$ UCI HIGGS events ($K = 8, 50 "iterations"$): (a) True Higgs boson signal enrichment rate (%) vs. $52.99%$ mean; (b) Converged centroid values of invariant mass coordinates ($m_(b b), m_(w b b), m_(w w b b)$).]
)

#figure(
  table(
    columns: (0.6fr, 1.0fr, 0.85fr, 0.85fr, 0.75fr, 0.75fr, 0.75fr, 0.75fr, 2.5fr),
    inset: 3.8pt,
    align: (center, right, right, right, right, right, right, right, left),
    fill: (c, r) => if r == 0 { rgb("#1b4f72") } else if r == 7 or r == 1 { rgb("#e8f8f5") } else if r == 6 or r == 8 { rgb("#fdedec") } else { white },
    [#text(fill: white, weight: "bold")[Cluster]],
    [#text(fill: white, weight: "bold")[Events]],
    [#text(fill: white, weight: "bold")[Signal %]],
    [#text(fill: white, weight: "bold")[Lep $p_T$]],
    [#text(fill: white, weight: "bold")[$"MET"$]],
    [#text(fill: white, weight: "bold")[$m_(b b)$]],
    [#text(fill: white, weight: "bold")[$m_(w b b)$]],
    [#text(fill: white, weight: "bold")[$m_(w w b b)$]],
    [#text(fill: white, weight: "bold")[High-Energy Physics Interpretation]],
    [*C1*], [1,560,788], [*53.90%*], [0.990], [1.015], [0.784], [0.982], [0.957], [Resonant low-$m_(b b)$ vector/Higgs regime (signal-enriched)],
    [C2], [1,729,399], [52.77%], [0.990], [0.990], [1.120], [1.056], [0.950], [High-$m_(b b)$ multi-jet topology (partner to C4)],
    [C3], [1,120,799], [52.57%], [0.992], [1.008], [0.903], [1.014], [0.966], [Central transition $W b overline(b)$ kinematic bulk (partner to C5)],
    [C4], [1,732,201], [52.64%], [0.990], [0.989], [1.120], [1.056], [0.950], [Symmetric high-$m_(b b)$ $t overline(t)$/mixed hadronic branch],
    [C5], [1,121,982], [52.46%], [0.993], [1.008], [0.905], [1.014], [0.967], [Symmetric central $W b overline(b)$ kinematic bulk],
    [*C6*], [923,472], [*49.72%*], [1.007], [*1.039*], [0.754], [0.992], [*0.983*], [*Highest $"MET"$ & $m_(w w b b)$; background-dominated ($t overline(t)$)*],
    [*C7*], [1,263,796], [*57.56%*], [0.983], [0.975], [0.927], [1.042], [0.952], [*Highest Higgs signal purity ($57.56%, +4.57%$ above mean)*],
    [C8], [1,547,563], [51.62%], [0.992], [0.984], [1.103], [1.079], [0.969], [Heavy $m_(b b) = 1.103$, $m_(w b b) = 1.079$ background-enriched],
  ),
  caption: [Physical cluster profiles across all $N = 11,000,000$ HIGGS events (`kmeans_cpu_benchmarks.o1563432`, sum $= 11,000,000$).]
)

*Key Physics Findings Across All 11,000,000 Events:*
1. *Unsupervised Signal Enrichment (`C7` & `C1` vs. `C6` & `C8`):* Without ever reading binary class labels during clustering, Lloyd's K-Means isolates a strongly signal-enriched resonant regime in *Cluster C7* ($1,263,796$ events, *$57.56%$ Higgs signal*, $+4.57$ percentage points above the $52.99%$ baseline) characterized by balanced low $"MET"$ ($0.975$) and resonant invariant mass ($m_(b b) = 0.927, m_(w b b) = 1.042$). Conversely, *Cluster C6* ($923,472$ events) captures high missing transverse energy ($"MET" = 1.039$) and elevated $m_(w w b b) = 0.983$ with compressed $m_(b b) = 0.754$, falling to *$49.72%$ signal* (majority $t overline(t)$ leptonic background, a *$7.84$ percentage-point separation* vs. C7), while *Cluster C8* ($1,547,563$ events, $m_(b b) = 1.103, m_(w b b) = 1.079$) captures the heavy invariant-mass background tail ($51.62%$ signal).
2. *Azimuthal & Pseudorapidity Detector Symmetry (`C2`/`C4` and `C3`/`C5`):* Notice that pairs (`C2`, `C4`) and (`C3`, `C5`) have nearly identical invariant masses ($m_(b b) = 1.120$ and $0.903–0.905$) and near-equal cluster sizes ($approx 1.73"M"$ and $approx 1.12"M"$ events each). Because LHC proton–proton collisions are symmetric across forward/backward detector pseudorapidity ($eta -> -eta$) and azimuthal angle ($phi$), K-Means naturally bifurcates rotationally equivalent kinematic topologies along the low-level detector angle coordinates while preserving identical physical invariant-mass signatures.

= Declaration of AI Usage & Source Code Repository
In accordance with the DSA5208 Project 3 guidelines, AI coding assistance (Google Antigravity) was used as a pair-programming assistant to: (1) structure C++17 command-line argument parsing, fast `strtod`/binary cache I/O boilerplate, and CUDA/Metal device memory error checks across `src/`; (2) author the automated 16-case verification test harness (`scripts/test_correctness.py`), PBS batch submission scripts (`scripts/vanda_*.pbs`), Python scaling harnesses, and Matplotlib plotting utilities (`scripts/generate_report_figures.py`); and (3) format the Typst layout and cross-check numerical consistency against empirical outputs. All reported measurements were executed on real hardware (NUS Vanda nodes `CN-111` and `GN-A40-044` over the full $11,000,000$-row UCI HIGGS dataset, plus Apple Silicon M5), and all source code, complete raw Vanda job logs (`kmeans_cpu_benchmarks.o1563432`, `kmeans_gpu_benchmark.o1563433`), CSV tables, and build instructions (`README.md`) are published in our public GitHub repository: #link("https://github.com/Nithillen/dsa5208-project3-scalable-kmeans")[`https://github.com/Nithillen/dsa5208-project3-scalable-kmeans`].
