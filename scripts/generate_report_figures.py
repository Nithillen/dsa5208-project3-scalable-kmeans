#!/usr/bin/env python3
"""
Generates all 7 publication-quality 300-DPI figures for the DSA5208 Project 3 Report.
Reads directly from official 11,000,000-row NUS Vanda results/*.csv (and Vanda job logs)
so EVERY value on EVERY figure matches the report tables and text to the exact decimal place.
Follows Lecture 12 Slide 38:
  "Time, problem size, and CPU count should be plotted on a logarithmic axis."
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def read_csv_dict(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def generate_fig1_serial_scaling():
    rows_n = read_csv_dict("serial_scaling_N.csv")
    rows_k = read_csv_dict("serial_scaling_K.csv")

    N_vals = [int(r["N"]) for r in rows_n]
    T_n = [float(r["total_time_s"]) for r in rows_n]

    K_vals = [int(r["K"]) for r in rows_k]
    T_k = [float(r["total_time_s"]) for r in rows_k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9))

    # Panel A: Log-Log Time vs Problem Size N
    ax1.loglog(N_vals, T_n, "o-", color="#1f4e78", linewidth=2.2, markersize=7,
               label="Measured Serial Runtime T(N) on Xeon 8452Y")
    # Theoretical linear reference anchored at N=500,000 (steady-state out-of-cache throughput)
    anchor_idx = 1 if len(N_vals) > 1 else 0
    ideal_n = [T_n[anchor_idx] * (n / N_vals[anchor_idx]) for n in N_vals]
    ax1.loglog(N_vals, ideal_n, "--", color="#d95f02", linewidth=1.6,
               label="Theoretical Linear O(N) Reference")

    offsets_n = [(-16, 12), (-22, 10), (22, -16), (-22, 10), (-22, 10), (-24, 10)]
    for idx, (n, t) in enumerate(zip(N_vals, T_n)):
        ox, oy = offsets_n[idx] if idx < len(offsets_n) else (0, 10)
        lbl = f"{t:.4f}s\n(11M)" if n == 11000000 else f"{t:.4f}s\n({n//1000}k)"
        ax1.annotate(lbl, (n, t), textcoords="offset points", xytext=(ox, oy),
                     ha="center", fontsize=7.4, fontweight="bold", color="#1f4e78")

    ax1.set_title("(a) Serial Runtime vs. Problem Size N (Log-Log)\n[Vanda Xeon 8452Y, Fixed K = 8, d = 28, 30 Iterations]")
    ax1.set_xlabel("Problem Size N (HIGGS Events) [Log Scale]")
    ax1.set_ylabel("Total Runtime (seconds) [Log Scale]")
    ax1.set_xlim(6e4, 1.9e7)
    ax1.set_ylim(0.22, 65.0)
    ax1.legend(loc="upper left")

    # Panel B: Log-Log Time vs Cluster Count K on Full 11,000,000 Rows
    ax2.loglog(K_vals, T_k, "s-", color="#2b5c8f", linewidth=2.2, markersize=7,
               label="Measured Serial Runtime T(K) (N=11M)")
    ideal_k = [T_k[0] * (k / K_vals[0]) for k in K_vals]
    ax2.loglog(K_vals, ideal_k, "--", color="#d95f02", linewidth=1.6,
               label="Theoretical Linear O(K) Slope")

    offsets_k = [(18, -16), (18, -16), (18, -16), (20, -18), (22, -20)]
    for idx, (k, t) in enumerate(zip(K_vals, T_k)):
        ox, oy = offsets_k[idx] if idx < len(offsets_k) else (0, 10)
        ax2.annotate(f"K={k}\n{t:.4f}s", (k, t), textcoords="offset points",
                     xytext=(ox, oy), ha="center", fontsize=7.5, fontweight="bold", color="#2b5c8f")

    ax2.set_xscale("log", base=2)
    ax2.set_xticks(K_vals)
    ax2.set_xlim(1.5, 52)
    ax2.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.set_title("(b) Serial Runtime vs. Cluster Count K (Log-Log)\n[Full UCI HIGGS: N = 11,000,000, d = 28, 30 Iterations]")
    ax2.set_xlabel("Number of Clusters K [Log2 Scale]")
    ax2.set_ylabel("Total Runtime (seconds) [Log Scale]")
    ax2.set_xlim(1.5, 44)
    ax2.set_ylim(6.0, 220.0)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_serial_scaling.png"))
    plt.close()


def generate_fig2_mpi_strong_scaling():
    rows = read_csv_dict("mpi_strong_scaling.csv")
    P_meas = [int(r["P"]) for r in rows]
    T_tot  = [float(r["total_time_s"]) for r in rows]
    T_comp = [float(r["comp_time_s"]) for r in rows]
    T_comm = [max(float(r["comm_time_s"]), 1e-4) for r in rows]
    S_meas = [float(r["speedup"]) for r in rows]
    E_meas = [float(r["efficiency_pct"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # Panel A: Runtime Decomposition on Log-Log Axes (1..64 Vanda Cores on 11M Rows)
    ax1.loglog(P_meas, T_tot, "o-", color="#1b4f72", linewidth=2.4, markersize=7.5,
               label="Measured Total Time T(P)")
    ax1.loglog(P_meas, T_comp, "s--", color="#2874a6", linewidth=1.8, markersize=6,
               label="Computation Time")
    ax1.loglog(P_meas[1:], T_comm[1:], "^-", color="#c0392b", linewidth=2.0, markersize=6.5,
               label="MPI_Allreduce Comm Time")

    for p, t in zip(P_meas, T_tot):
        ax1.annotate(f"{t:.4f}s", (p, t), textcoords="offset points", xytext=(8 if p==1 else 0, 9),
                     ha="center", fontsize=7.5, fontweight="bold", color="#1b4f72")
    for p, c in zip(P_meas[1:], T_comm[1:]):
        ox = -14 if p == 32 else 0
        dy = -14 if p != 32 else -16
        ax1.annotate(f"{c:.4f}s", (p, c), textcoords="offset points", xytext=(ox, dy),
                     ha="center", fontsize=7.4, fontweight="bold", color="#c0392b")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(P_meas)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.set_title("(a) Pure MPI Strong Scaling Runtime Breakdown (Log-Log)\n[Vanda Xeon 8452Y, Full HIGGS N = 11,000,000, K = 8, 50 Iters]")
    ax1.set_xlabel("Number of MPI Processes / Physical Cores (P) [Log2 Scale]")
    ax1.set_ylabel("Execution Time (seconds) [Log10 Scale]")
    ax1.set_xlim(0.75, 92)
    ax1.set_ylim(0.05, 75.0)
    ax1.legend(loc="lower left", fontsize=8.0)

    # Panel B: Strong Scaling Speedup & Efficiency on Log-Log Axes
    ax2.loglog(P_meas, P_meas, "--", color="#7f8c8d", linewidth=1.6,
               label="Ideal Linear Speedup S(P) = P")
    ax2.loglog(P_meas, S_meas, "o-", color="#117864", linewidth=2.4, markersize=7.5,
               label="Measured Vanda Speedup S(P) (P=1..64)")

    offsets_s = {1: (24, -18), 2: (-20, 12), 4: (-22, 12), 8: (-26, 11), 16: (24, -18), 32: (24, -18), 64: (-30, 12)}
    for p, s, e in zip(P_meas, S_meas, E_meas):
        ox, oy = offsets_s.get(p, (0, 8))
        ax2.annotate(f"{s:.2f}x\n({e:.1f}%)", (p, s), textcoords="offset points",
                     xytext=(ox, oy), ha="center", fontsize=7.5, fontweight="bold", color="#0e6251")

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xticks(P_meas)
    ax2.set_yticks(P_meas)
    ax2.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.get_yaxis().set_major_formatter(ScalarFormatter())
    ax2.set_title("(b) Pure MPI Strong Scaling Speedup & Efficiency (Log-Log)\n[41.40x Speedup (64.7% Eff) at P=64 | 849.74 M pts/sec]")
    ax2.set_xlabel("Number of MPI Processes / Physical Cores (P) [Log2 Scale]")
    ax2.set_ylabel("Parallel Speedup S(P) = T(1) / T(P) [Log2 Scale]")
    ax2.set_xlim(0.75, 92)
    ax2.set_ylim(0.6, 105)
    ax2.legend(loc="upper left", fontsize=8.0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_mpi_strong_scaling.png"))
    plt.close()


def generate_fig3_mpi_weak_scaling():
    rows = read_csv_dict("mpi_weak_scaling.csv")
    P_meas = [int(r["P"]) for r in rows]
    T_tot  = [float(r["total_time_s"]) for r in rows]
    T_comp = [float(r["comp_time_s"]) for r in rows]
    T_comm = [max(float(r["comm_time_s"]), 1e-4) for r in rows]
    Thr    = [float(r["throughput_mpts_s"]) for r in rows]
    Eff    = [float(r["weak_efficiency_pct"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.9))

    # Panel A: Weak Scaling Runtime (Log-Log)
    ax1.loglog(P_meas, T_tot, "o-", color="#1f618d", linewidth=2.3, markersize=7.2,
               label="Total Runtime T(P, P*N0)")
    ax1.loglog(P_meas, T_comp, "s--", color="#2e86c1", linewidth=1.8, markersize=5.8,
               label="Computation Time")
    ax1.loglog(P_meas[1:], T_comm[1:], "^-", color="#c0392b", linewidth=2.0, markersize=6.5,
               label="MPI_Allreduce Comm Time")

    dy_tot = {1: 10, 2: -22, 4: 10, 8: 10, 16: 10, 32: 10, 64: 10}
    for p, t, e in zip(P_meas, T_tot, Eff):
        ax1.annotate(f"{t:.4f}s\n({e:.1f}%)", (p, t), textcoords="offset points",
                     xytext=(0, dy_tot.get(p, 10)), ha="center", fontsize=7.3, fontweight="bold", color="#1f618d")
    for p, c in zip(P_meas[1:], T_comm[1:]):
        ox = 16 if p == 4 else 0
        dy = -13 if p != 64 else -15
        ax1.annotate(f"{c:.4f}s", (p, c), textcoords="offset points",
                     xytext=(ox, dy), ha="center", fontsize=7.2, fontweight="bold", color="#c0392b")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(P_meas)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.set_title("(a) Pure MPI Weak Scaling Runtime Breakdown (Log-Log)\n[Vanda: 200,000 Pts/Rank -> Total N = 200k to 12.8M]")
    ax1.set_xlabel("Number of MPI Processes (P) [Log2 Scale]")
    ax1.set_ylabel("Execution Time (seconds) [Log10 Scale]")
    ax1.set_xlim(0.75, 92)
    ax1.set_ylim(0.004, 1.45)
    ax1.legend(loc="lower right")

    # Panel B: Weak Scaling Aggregate Throughput (Log-Log)
    ideal_thr = [Thr[0] * p for p in P_meas]
    ax2.loglog(P_meas, ideal_thr, "--", color="#7f8c8d", linewidth=1.6,
               label="Ideal Gustafson Linear Throughput")
    ax2.loglog(P_meas, Thr, "o-", color="#117864", linewidth=2.3, markersize=7.2,
               label="Measured Aggregate Throughput (M pts/s)")

    offs_th = {1: (16, -22), 2: (14, -22), 4: (14, -22), 8: (14, -22), 16: (14, -22), 32: (14, -22), 64: (0, -26)}
    for p, th in zip(P_meas, Thr):
        ox, oy = offs_th.get(p, (0, 10))
        ax2.annotate(f"{th:.2f}\nM/s", (p, th), textcoords="offset points",
                     xytext=(ox, oy), ha="center", fontsize=7.3, fontweight="bold", color="#117864")

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xticks(P_meas)
    ax2.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.get_yaxis().set_major_formatter(ScalarFormatter())
    ax2.set_title("(b) Pure MPI Weak Scaling Aggregate Throughput (Log-Log)\n[Scales from 22.28 M/s (P=1) to 797.72 M/s (P=64, N=12.8M)]")
    ax2.set_xlabel("Number of MPI Processes (P) [Log2 Scale]")
    ax2.set_ylabel("Throughput (Million Points / Sec) [Log2 Scale]")
    ax2.set_xlim(0.75, 90)
    ax2.set_ylim(12, 1800)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_mpi_weak_scaling.png"))
    plt.close()


def generate_fig4_omp_design_choices():
    rows = read_csv_dict("omp_design_choices.csv")
    label_map = {
        ("private", "static"): "private\n(static, MPI)",
        ("private", "dynamic"): "private\n(dynamic, MPI)",
        ("private", "guided"): "private\n(guided, MPI)",
        ("atomic", "static"): "atomic\n(static, MPI)",
        ("private_standalone", "static"): "private\n(standalone)",
    }
    labels = [label_map.get((r["accum_mode"], r["schedule"]), f"{r['accum_mode']}\n({r['schedule']})") for r in rows]
    times  = [float(r["total_time_s"]) for r in rows]
    thrs   = [float(r["throughput_mpts_s"]) for r in rows]
    colors = ["#2b5c8f", "#1f78b4", "#117864", "#c0392b", "#5b2c6f"]

    atomic_thr = next(float(r["throughput_mpts_s"]) for r in rows if r["accum_mode"] == "atomic")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # Panel A: Total Runtime on Log Scale
    bars1 = ax1.bar(labels, times, color=colors, width=0.52, edgecolor="black", linewidth=0.8)
    ax1.set_yscale("log")
    for bar, t in zip(bars1, times):
        ax1.annotate(f"{t:.4f}s\n({t*1000/50:.1f} ms/it)",
                     (bar.get_x() + bar.get_width() / 2, t),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7.6, fontweight="bold")

    ax1.set_title("(a) OpenMP Accumulation & Schedule Runtime (Log Scale)\n[Vanda CN-048: P = 1, T = 4 Threads, N = 11,000,000, K = 8, 50 Iters]")
    ax1.set_ylabel("Total Execution Time (seconds) [Log Scale]")
    ax1.set_ylim(5.0, 420.0)
    ax1.tick_params(axis="x", labelsize=8.0)

    # Panel B: Throughput on Log Scale
    bars2 = ax2.bar(labels, thrs, color=colors, width=0.52, edgecolor="black", linewidth=0.8)
    ax2.set_yscale("log")
    for bar, th in zip(bars2, thrs):
        sp = th / atomic_thr
        ax2.annotate(f"{th:.2f} M/s\n({sp:.2f}x)",
                     (bar.get_x() + bar.get_width() / 2, th),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7.6, fontweight="bold")

    ax2.set_title("(b) OpenMP Throughput Comparison (Log Scale)\n[Cache-Padded Private Buffers Beat Atomic Contention by 10.00x]")
    ax2.set_ylabel("Throughput (Million Points / Sec) [Log Scale]")
    ax2.set_ylim(1.8, 120.0)
    ax2.tick_params(axis="x", labelsize=8.0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_omp_design_choices.png"))
    plt.close()


def generate_fig5_hybrid_mpi_omp_grid():
    rows = read_csv_dict("hybrid_mpi_omp_grid.csv")

    pure_omp = [(int(r["total_cores"]), float(r["total_time_s"])) for r in rows if int(r["P"]) == 1]
    pure_mpi = [(int(r["total_cores"]), float(r["total_time_s"])) for r in rows if int(r["T"]) == 1]
    dual_socket_hybrid = [
        (int(r["total_cores"]), float(r["total_time_s"]))
        for r in rows if int(r["P"]) == 2 and int(r["T"]) > 1
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # Panel A: Log-Log Runtime vs Total Cores (1..72 Vanda Cores on 11M Rows)
    ax1.loglog([x[0] for x in pure_omp], [x[1] for x in pure_omp], "o-", color="#c0392b",
               linewidth=2.1, markersize=7.0, label="Pure OpenMP (P=1 Rank, T=1..72)")
    ax1.loglog([x[0] for x in dual_socket_hybrid], [x[1] for x in dual_socket_hybrid], "D-", color="#d35400",
               linewidth=2.2, markersize=7.2, label="NUMA-Aligned Hybrid (P=2, T=8,18,36)")
    ax1.loglog([x[0] for x in pure_mpi], [x[1] for x in pure_mpi], "s-", color="#117864",
               linewidth=2.4, markersize=7.2, label="Flat MPI Inside Node (P=1..72, T=1)")

    omp_offsets = {1: (12, 8), 8: (0, 10), 16: (22, 2), 36: (22, 2), 72: (0, -19)}
    lbl_bbox = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88)
    for c, t in pure_omp:
        ox, oy = omp_offsets.get(c, (0, 10))
        ax1.annotate(f"1x{c}: {t:.2f}s", (c, t), textcoords="offset points",
                     xytext=(ox, oy), ha="center", fontsize=7.2, fontweight="bold",
                     color="#c0392b", bbox=lbl_bbox)
    for c, t in pure_mpi:
        if c > 1:
            ax1.annotate(f"{c}x1: {t:.2f}s", (c, t), textcoords="offset points",
                         xytext=(-18 if c in (16, 36, 72) else 0, -16), ha="center", fontsize=7.2,
                         fontweight="bold", color="#117864", bbox=lbl_bbox)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 8, 16, 36, 72])
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.set_title("(a) Hybrid MPI x OpenMP Runtime vs. Active Cores (Log-Log)\n[Vanda CN-048: N = 11,000,000, d = 28, K = 8, 50 Iterations]")
    ax1.set_xlabel("Total Active CPU Cores (C = P x T) [Log2 Scale]")
    ax1.set_ylabel("Total Execution Time (seconds) [Log10 Scale]")
    ax1.set_xlim(0.75, 105)
    ax1.set_ylim(0.32, 55.0)
    ax1.legend(loc="lower left", fontsize=7.8)

    # Panel B: Full-Node 72-Core Configuration Breakdown (P*T = 72)
    rows_72 = [r for r in rows if int(r["total_cores"]) == 72]
    cfg_labels = [f"{r['P']}x{r['T']}" for r in rows_72]
    comm_72 = [float(r["comm_time_s"]) for r in rows_72]
    pct_72  = [float(r["comm_pct"]) for r in rows_72]
    tot_72  = [float(r["total_time_s"]) for r in rows_72]

    # Visualize compute + comm share summing to total
    comm_share = [min(cm, tot) for cm, tot in zip(comm_72, tot_72)]
    comp_share = [max(tot - cs, 0.0) for tot, cs in zip(tot_72, comm_share)]

    x = np.arange(len(cfg_labels))
    ax2.bar(x, comp_share, width=0.56, color="#2e86c1", edgecolor="black", label="Compute Portion (s)")
    ax2.bar(x, comm_share, width=0.56, bottom=comp_share, color="#e74c3c", edgecolor="black", label="MPI_Allreduce Sync (s)")

    for i, (tot, pct) in enumerate(zip(tot_72, pct_72)):
        ax2.annotate(f"{tot:.4f}s\n({pct:.1f}%)", (x[i], tot), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=7.2, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(cfg_labels)
    ax2.set_title("(b) Full 72-Core Topology Trade-Off (P x T = 72 on N=11M)\n[Pinned 1x72 Hits 1.4163s; Flat MPI 72x1 Hits 0.6243s]")
    ax2.set_xlabel("Topology Configuration (MPI Ranks P x OpenMP Threads T)")
    ax2.set_ylabel("Total Execution Time (seconds)")
    ax2.set_ylim(0, 2.55)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_hybrid_mpi_omp_grid.png"))
    plt.close()


def generate_fig6_gpu_scaling():
    rows = read_csv_dict("gpu_benchmark.csv")
    K_vals   = [int(r["K"]) for r in rows]
    cpu1_s   = [float(r["cpu_1core_s"]) for r in rows]
    omp4_s   = [float(r["omp_4core_s"]) for r in rows]
    gpu_tot  = [float(r["gpu_total_s"]) for r in rows]
    sp_1c    = [float(r["gpu_speedup_vs_1core"]) for r in rows]
    sp_4c    = [omp4_s[i] / gpu_tot[i] for i in range(len(rows))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # Panel A: Vanda NVIDIA A40 CUDA vs 1C/4C CPU on Full N=11,000,000 (30 iters)
    ax1.loglog(K_vals, cpu1_s, "o-", color="#7f8c8d", linewidth=2.2, markersize=7,
               label="1-Core Xeon CPU (P=1, T=1)")
    ax1.loglog(K_vals, omp4_s, "s-", color="#1f618d", linewidth=2.2, markersize=7,
               label="4-Core Xeon OpenMP (P=1, T=4)")
    ax1.loglog(K_vals, gpu_tot, "D-", color="#117864", linewidth=2.5, markersize=7.5,
               label="NVIDIA A40 CUDA GPU (1 Rank)")

    for k, c1, o4, gt, s1 in zip(K_vals, cpu1_s, omp4_s, gpu_tot, sp_1c):
        ax1.annotate(f"{c1:.2f}s", (k, c1), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7.6, fontweight="bold", color="#555555")
        ax1.annotate(f"{o4:.2f}s", (k, o4), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7.6, fontweight="bold", color="#1f618d")
        ax1.annotate(f"A40: {gt:.2f}s ({s1:.2f}x)", (k, gt), textcoords="offset points", xytext=(0, -15),
                     ha="center", fontsize=7.5, fontweight="bold", color="#117864")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(K_vals)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.set_title("(a) Vanda NVIDIA A40 CUDA vs. CPU Scaling Across K (Log-Log)\n[Full UCI HIGGS: N = 11,000,000, d = 28, 30 Iterations]")
    ax1.set_xlabel("Number of Clusters K [Log2 Scale]")
    ax1.set_ylabel("Execution Time (seconds) [Log10 Scale]")
    ax1.set_xlim(6, 44)
    ax1.set_ylim(1.5, 110.0)
    ax1.legend(loc="upper left", fontsize=8.0)

    # Panel B: Complete 50-Iteration Backend Comparison on Vanda N=11,000,000, K=8
    backends = [
        "Serial C++\n(1 Xeon Core)",
        "OMP CPU SIMD\n(P=1, T=1)",
        "OMP GPU Offload\n(A40 -mp=gpu)",
        "Native CUDA\n(NVIDIA A40)"
    ]
    t_50 = [47.1513, 30.4406, 16.9227, 5.0769]
    b_colors = ["#7f8c8d", "#2874a6", "#d35400", "#117864"]

    bars = ax2.bar(backends, t_50, color=b_colors, width=0.52, edgecolor="black", linewidth=0.8)
    ax2.set_yscale("log")
    for bar, t in zip(bars, t_50):
        sp = t_50[0] / t
        ax2.annotate(f"{t:.4f}s\n({sp:.2f}x vs Serial)",
                     (bar.get_x() + bar.get_width() / 2, t),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7.6, fontweight="bold")

    ax2.set_title("(b) 50-Iteration Runtime Across Vanda Backends (Log Scale)\n[Full N = 11,000,000, K = 8: Native CUDA Hits 108.33 M pts/s]")
    ax2.set_ylabel("50-Iteration Runtime (seconds) [Log Scale]")
    ax2.set_ylim(2.5, 120.0)
    ax2.tick_params(axis="x", labelsize=8.2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_gpu_scaling.png"))
    plt.close()


def generate_fig7_higgs_physics():
    # Exact 11,000,000-row HIGGS Boson Physical Cluster Analysis from kmeans_cpu_benchmarks.o1565343
    clusters = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    events   = [1560788, 1729399, 1120799, 1732201, 1121982, 923472, 1263796, 1547563]
    sig_pct  = [53.90, 52.77, 52.57, 52.64, 52.46, 49.72, 57.56, 51.62]
    m_bb     = [0.784, 1.120, 0.903, 1.120, 0.905, 0.754, 0.927, 1.103]
    m_wbb    = [0.982, 1.056, 1.014, 1.056, 1.014, 0.992, 1.042, 1.079]
    m_wwbb   = [0.957, 0.950, 0.966, 0.950, 0.967, 0.983, 0.952, 0.969]

    assert sum(events) == 11000000, f"Expected 11,000,000 events, got {sum(events)}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.9))

    # Panel A: Signal Enrichment % vs Cluster
    colors = ["#117864" if s >= 53.5 else ("#c0392b" if s < 51.8 else "#2e86c1") for s in sig_pct]
    bars = ax1.bar(clusters, sig_pct, color=colors, width=0.58, edgecolor="black")
    ax1.axhline(52.99, color="black", linestyle="--", linewidth=1.3,
                label="Full 11M Dataset Mean Signal Rate (52.99%)")

    for bar, s, ev in zip(bars, sig_pct, events):
        if 52.0 <= s <= 53.3:
            ax1.annotate(f"{s:.2f}%",
                         (bar.get_x() + bar.get_width() / 2, s),
                         textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=8.0, fontweight="bold", color="white")
        else:
            ax1.annotate(f"{s:.2f}%",
                         (bar.get_x() + bar.get_width() / 2, s),
                         textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=8.0, fontweight="bold", color="black")
        ax1.text(bar.get_x() + bar.get_width() / 2, 48.2, f"{ev/1e6:.2f}M",
                 ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="white")

    ax1.set_title("(a) True Higgs Boson Signal Purity Across K=8 Clusters\n[Unsupervised Clustering on Full N = 11,000,000 Collision Events]")
    ax1.set_xlabel("Discovered K-Means Cluster ID")
    ax1.set_ylabel("True Higgs Signal Percentage (%)")
    ax1.set_ylim(47.5, 60.4)
    ax1.legend(loc="upper right")

    # Panel B: High-Level Invariant Mass Features per Cluster
    x = np.arange(len(clusters))
    w = 0.26
    ax2.bar(x - w, m_bb, width=w, color="#1f618d", edgecolor="black", label=r"$m_{bb}$ (b-quark pair mass)")
    ax2.bar(x,     m_wbb, width=w, color="#2e86c1", edgecolor="black", label=r"$m_{wbb}$ (W + b-pair mass)")
    ax2.bar(x + w, m_wwbb, width=w, color="#117864", edgecolor="black", label=r"$m_{wwbb}$ (WW + b-pair mass)")

    for i, val in enumerate(m_bb):
        ax2.annotate(f"{val:.3f}", (x[i] - w, val), textcoords="offset points", xytext=(-3, 4),
                     ha="center", fontsize=6.2, fontweight="bold", color="#1f618d")

    ax2.set_xticks(x)
    ax2.set_xticklabels(clusters)
    ax2.set_title(r"(b) Converged Centroid Invariant Masses ($m_{bb}, m_{wbb}, m_{wwbb}$)" +
                  "\n[C7 Enriches Higgs Signal (57.56%) vs. C6 Background (49.72%)]",
                  fontsize=10.2, fontweight="bold")
    ax2.set_xlabel("Discovered K-Means Cluster ID")
    ax2.set_ylabel("Standardized Feature Value")
    ax2.set_ylim(0.60, 1.28)
    ax2.legend(loc="upper left", fontsize=8.0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_higgs_physics_clusters.png"))
    plt.close()


if __name__ == "__main__":
    generate_fig1_serial_scaling()
    generate_fig2_mpi_strong_scaling()
    generate_fig3_mpi_weak_scaling()
    generate_fig4_omp_design_choices()
    generate_fig5_hybrid_mpi_omp_grid()
    generate_fig6_gpu_scaling()
    generate_fig7_higgs_physics()
    print("Successfully generated all 7 publication-grade figures in report/figures/.")
