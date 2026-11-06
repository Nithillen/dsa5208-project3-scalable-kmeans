#!/usr/bin/env python3
"""
Step 3 & 4: Hybrid MPI + OpenMP & GPU Acceleration Benchmark Suite on UCI HIGGS Dataset
Measures:
1. OpenMP Design Choices (Thread-Private Padded Accumulators vs. Atomic Contention, and static/dynamic/guided schedules)
   plus Standalone OpenMP (no MPI) vs. 1-Rank MPI+OpenMP.
2. Hybrid MPI Ranks (P) x OpenMP Threads (T) Trade-offs on UCI HIGGS Dataset (N=597,638, d=28, K=8, 50 Iters)
3. Hardware GPU Acceleration (NVIDIA CUDA GPU on Vanda or Apple Metal GPU on macOS) vs. CPU Baselines across K={8, 16, 32}
   - Runs strictly without --oversubscribe by default so measurements reflect physical hardware cores.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_OMP = os.path.join(BASE_DIR, "bin", "kmeans_omp")
BIN_OMP_STANDALONE = os.path.join(BASE_DIR, "bin", "kmeans_omp_standalone")
BIN_CUDA = os.path.join(BASE_DIR, "bin", "kmeans_cuda")
BIN_METAL = os.path.join(BASE_DIR, "bin", "kmeans_metal_gpu")
DATA_CSV = os.path.join(BASE_DIR, "data", "higgs_clean.csv")
REF_TXT = os.path.join(BASE_DIR, "results", "centroids_serial.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MPIRUN = shutil.which("mpirun") or "/opt/homebrew/bin/mpirun"


def run_omp(np, threads, accum="private", sched="static", K=8, iters=50, oversubscribe=False, standalone=False):
    if standalone and os.path.exists(BIN_OMP_STANDALONE):
        cmd = [
            BIN_OMP_STANDALONE, "-f", DATA_CSV, "-r", REF_TXT,
            "-threads", str(threads), "-a", accum, "-s", sched,
            "-k", str(K), "-m", str(iters), "-t", "0.0"
        ]
    else:
        cmd = [MPIRUN]
        if oversubscribe:
            cmd.append("--oversubscribe")
        cmd.extend([
            "-np", str(np), BIN_OMP, "-f", DATA_CSV, "-r", REF_TXT,
            "-threads", str(threads), "-a", accum, "-s", sched,
            "-k", str(K), "-m", str(iters), "-t", "0.0"
        ])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout

    total_s = float(re.search(r"Total Time\s*:\s*([\d.]+)", out).group(1))
    comp_s = float(re.search(r"Computation Time\s*:\s*([\d.]+)", out).group(1))
    comm_s = float(re.search(r"MPI_Allreduce Tm\s*:\s*([\d.]+)", out).group(1))
    comm_pct = float(re.search(r"MPI_Allreduce Tm\s*:\s*[\d.]+\s*seconds\s*\(([\d.]+)%\)", out).group(1))
    avg_iter_ms = float(re.search(r"Avg Time/Iter\s*:\s*([\d.]+)", out).group(1))
    throughput = float(re.search(r"Throughput\s*:\s*([\d.]+)", out).group(1))
    sse_match = re.search(r"Final SSE \(WCSS\)\s*:\s*([\d.eE+-]+)", out)
    final_sse = float(sse_match.group(1)) if sse_match else 0.0
    err_match = re.search(r"Max Abs Error\s*:\s*([\d.eE+-]+)", out)
    max_err = float(err_match.group(1)) if err_match else 0.0
    return total_s, comp_s, comm_s, comm_pct, avg_iter_ms, throughput, final_sse, max_err


def run_gpu(np, threads, K=8, iters=50, oversubscribe=False):
    cmd = [MPIRUN]
    if oversubscribe:
        cmd.append("--oversubscribe")

    if os.path.exists(BIN_CUDA):
        cmd.extend(["-np", str(np), BIN_CUDA, "-f", DATA_CSV, "-r", REF_TXT, "-k", str(K), "-m", str(iters), "-t", "0.0"])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = proc.stdout
        gpu_name = "NVIDIA CUDA GPU (Vanda)"
        total_s = float(re.search(r"Total Time\s*:\s*([\d.]+)", out).group(1))
        gpu_s = float(re.search(r"GPU Kernel\+Copy\s*:\s*([\d.]+)", out).group(1))
        accum_s = 0.0
        comm_s = float(re.search(r"MPI_Allreduce Tm\s*:\s*([\d.]+)", out).group(1))
    else:
        cmd.extend(["-np", str(np), BIN_METAL, "-f", DATA_CSV, "-r", REF_TXT, "-threads", str(threads), "-k", str(K), "-m", str(iters), "-t", "0.0"])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = proc.stdout
        gpu_name = re.search(r"GPU Hardware\s*:\s*(.+)", out).group(1).strip()
        total_s = float(re.search(r"Total Time\s*:\s*([\d.]+)", out).group(1))
        gpu_s = float(re.search(r"GPU Distance Tm\s*:\s*([\d.]+)", out).group(1))
        accum_s = float(re.search(r"OMP Accum Time\s*:\s*([\d.]+)", out).group(1))
        comm_s = float(re.search(r"MPI_Allreduce Tm\s*:\s*([\d.]+)", out).group(1))

    avg_iter_ms = float(re.search(r"Avg Time/Iter\s*:\s*([\d.]+)", out).group(1))
    throughput = float(re.search(r"Throughput\s*:\s*([\d.]+)", out).group(1))
    rmse_match = re.search(r"Centroid RMSE\s*:\s*([\d.eE+-]+)", out)
    rmse = float(rmse_match.group(1)) if rmse_match else 0.0
    sse_match = re.search(r"Final SSE \(WCSS\)\s*:\s*([\d.eE+-]+)", out)
    final_sse = float(sse_match.group(1)) if sse_match else 0.0
    return gpu_name, total_s, gpu_s, accum_s, comm_s, avg_iter_ms, throughput, rmse, final_sse


def main():
    parser = argparse.ArgumentParser(description="Run Hybrid MPI+OpenMP and GPU Benchmarks")
    parser.add_argument("--vanda", action="store_true", help="Use Vanda 72-core node P x T grid")
    parser.add_argument("--oversubscribe", action="store_true", help="Allow OpenMPI --oversubscribe")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. OpenMP Design Choices: Thread-Private vs Atomic & Schedules (P=1, T=4 on UCI HIGGS)
    print("=============================================================================================")
    print("  Step 3A: OpenMP Parallel Design Decisions on UCI HIGGS (P=1 Rank, T=4 Threads, K=8, 50 Iters)")
    print("=============================================================================================")
    print(f"{'Accumulation Mode':>18} | {'Schedule':>10} | {'Total (s)':>9} | {'Avg Iter (ms)':>13} | {'Throughput':>14} | {'Max Err vs Serial':>17}")
    print("-" * 92)

    design_csv = os.path.join(RESULTS_DIR, "omp_design_choices.csv")
    design_configs = [
        ("private", "static", False),
        ("private", "dynamic", False),
        ("private", "guided", False),
        ("atomic",  "static", False),
        ("private_standalone", "static", True),
    ]
    with open(design_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["P", "T", "accum_mode", "schedule", "total_time_s", "avg_iter_ms", "throughput_mpts_s", "max_abs_err", "final_sse"])
        for accum_label, sched, is_standalone in design_configs:
            actual_accum = "private" if "private" in accum_label else accum_label
            tot_s, comp_s, comm_s, comm_pct, iter_ms, thr, sse, max_err = run_omp(
                1, 4, accum=actual_accum, sched=sched, K=8, iters=50,
                oversubscribe=args.oversubscribe, standalone=is_standalone
            )
            writer.writerow([1, 4, accum_label, sched, f"{tot_s:.4f}", f"{iter_ms:.2f}", f"{thr:.2f}", f"{max_err:.3e}", f"{sse:.6e}"])
            print(f"{accum_label:>18} | {sched:>10} | {tot_s:9.4f} | {iter_ms:13.2f} | {thr:9.2f} M/s | {max_err:17.3e}")

    # 2. Hybrid MPI Ranks (P) x OpenMP Threads (T) Grid
    print("\n=============================================================================================")
    print("  Step 3B: Hybrid MPI Ranks (P) x OpenMP Threads (T) Scaling on UCI HIGGS (N=597,638, K=8)  ")
    print("=============================================================================================")
    print(f"{'Total Cores':>11} | {'MPI (P)':>7} | {'OMP (T)':>7} | {'Total (s)':>9} | {'Comp (s)':>8} | {'Comm (s)':>8} | {'Speedup':>7} | {'Efficiency':>10} | {'Max Err':>10}")
    print("-" * 96)

    grid_csv = os.path.join(RESULTS_DIR, "hybrid_mpi_omp_grid.csv")
    if args.vanda:
        grid_configs = [
            (1, 1),
            (1, 8), (8, 1),
            (1, 16), (2, 8), (4, 4), (8, 2), (16, 1),
            (1, 36), (2, 18), (4, 9), (9, 4), (18, 2), (36, 1),
            (1, 72), (2, 36), (4, 18), (8, 9), (18, 4), (36, 2), (72, 1),
        ]
    else:
        grid_configs = [
            (1, 1),
            (1, 2), (2, 1),
            (1, 4), (2, 2), (4, 1),
            (1, 8), (2, 4), (4, 2), (8, 1),
        ]

    t_base = None
    with open(grid_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "total_cores", "P", "T", "total_time_s", "comp_time_s", "comm_time_s",
            "comm_pct", "avg_iter_ms", "throughput_mpts_s", "speedup", "efficiency_pct", "max_abs_err", "final_sse"
        ])
        for p, t in grid_configs:
            tot_s, comp_s, comm_s, comm_pct, iter_ms, thr, sse, max_err = run_omp(
                p, t, accum="private", sched="static", K=8, iters=50, oversubscribe=args.oversubscribe
            )
            if t_base is None:
                t_base = tot_s
            total_cores = p * t
            speedup = t_base / tot_s
            eff_pct = (speedup / total_cores) * 100.0
            writer.writerow([
                total_cores, p, t, f"{tot_s:.4f}", f"{comp_s:.4f}", f"{comm_s:.4f}",
                f"{comm_pct:.1f}", f"{iter_ms:.2f}", f"{thr:.2f}", f"{speedup:.2f}",
                f"{eff_pct:.1f}", f"{max_err:.3e}", f"{sse:.6e}"
            ])
            print(f"{total_cores:11d} | {p:7d} | {t:7d} | {tot_s:9.4f} | {comp_s:8.4f} | {comm_s:8.4f} | {speedup:6.2f}x | {eff_pct:9.1f}% | {max_err:10.2e}")

    # 3. GPU Benchmark across K={8, 16, 32} on UCI HIGGS
    if os.path.exists(BIN_CUDA) or os.path.exists(BIN_METAL):
        print("\n=============================================================================================")
        print("  Step 4: GPU Acceleration (MPI + GPU vs. 1-Core CPU & 4-Core OMP on UCI HIGGS) ")
        print("=============================================================================================")
        print(f"{'Clusters K':>10} | {'1-Core CPU (s)':>14} | {'4-Core OMP (s)':>14} | {'GPU Total (s)':>13} | {'GPU Kernel (s)':>14} | {'GPU Speedup vs 1C':>17} | {'Centroid RMSE':>13}")
        print("-" * 108)

        gpu_csv = os.path.join(RESULTS_DIR, "gpu_benchmark.csv")
        with open(gpu_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "K", "gpu_device", "cpu_1core_s", "omp_4core_s", "gpu_total_s", "gpu_kernel_s",
                "host_accum_s", "gpu_speedup_vs_1core", "gpu_throughput_mpts_s", "centroid_rmse", "final_sse"
            ])
            for K in [8, 16, 32]:
                cpu1_s, _, _, _, _, _, _, _ = run_omp(1, 1, K=K, iters=30, oversubscribe=args.oversubscribe)
                omp4_s, _, _, _, _, _, _, _ = run_omp(1, 4, K=K, iters=30, oversubscribe=args.oversubscribe)
                gpu_name, gpu_tot_s, gpu_kern_s, accum_s, comm_s, iter_ms, thr, rmse, sse = run_gpu(1, 4, K=K, iters=30, oversubscribe=args.oversubscribe)
                speedup_1c = cpu1_s / gpu_tot_s
                writer.writerow([
                    K, gpu_name, f"{cpu1_s:.4f}", f"{omp4_s:.4f}", f"{gpu_tot_s:.4f}", f"{gpu_kern_s:.4f}",
                    f"{accum_s:.4f}", f"{speedup_1c:.2f}", f"{thr:.2f}", f"{rmse:.3e}", f"{sse:.6e}"
                ])
                print(f"{K:10d} | {cpu1_s:14.4f} | {omp4_s:14.4f} | {gpu_tot_s:13.4f} | {gpu_kern_s:14.4f} | {speedup_1c:16.2f}x | {rmse:13.3e}")

    print("\nSaved results to results/omp_design_choices.csv, results/hybrid_mpi_omp_grid.csv, and results/gpu_benchmark.csv.")


if __name__ == "__main__":
    main()
