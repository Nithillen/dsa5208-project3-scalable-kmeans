#!/usr/bin/env python3
"""
Step 2: Pure MPI Strong, Weak & Collective Reduction Scaling Benchmark Suite
1. Strong Scaling on UCI HIGGS Dataset (Fixed N=597,638 or full 11M, d=28, K=8, iters=50)
   - Measures Total Time, Computation Time, MPI_Allreduce Communication Time, Speedup,
     Parallel Efficiency, Final SSE (WCSS), and Max Centroid Error vs Serial.
2. Collective Reduction Comparison: Separate MPI_Allreduce vs. Packed Single-Call MPI_Allreduce (-p)
3. Weak Scaling on Synthetic Clustered Data (Fixed N_per_rank=200,000, d=28, K=8, iters=30)
   - Measures Weak Scaling Parallel Efficiency E_weak(P) = T(1) / T(P).
   - Runs strictly without --oversubscribe by default so measurements reflect physical hardware cores.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_MPI = os.path.join(BASE_DIR, "bin", "kmeans_mpi")
DATA_CSV = os.path.join(BASE_DIR, "data", "higgs_clean.csv")
REF_TXT = os.path.join(BASE_DIR, "results", "centroids_serial.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MPIRUN = shutil.which("mpirun") or "/opt/homebrew/bin/mpirun"


def run_mpi(np, args, oversubscribe=False):
    cmd = [MPIRUN]
    if oversubscribe:
        cmd.append("--oversubscribe")
    cmd.extend(["-np", str(np), BIN_MPI] + args)
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


def main():
    parser = argparse.ArgumentParser(description="Run Pure MPI Strong & Weak Scaling Benchmarks")
    parser.add_argument(
        "--ranks",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of MPI ranks (e.g., '1,2,4,8,16,32,64,128,256' on Vanda)",
    )
    parser.add_argument(
        "--oversubscribe",
        action="store_true",
        help="Allow OpenMPI --oversubscribe (leave disabled on Vanda HPC)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ranks_list = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]

    print("========================================================================================")
    print("  Step 2A: Pure MPI Strong Scaling on UCI HIGGS Dataset (N=597,638, d=28, K=8, 50 Iters)")
    print("========================================================================================")
    print(
        f"{'Ranks (P)':>9} | {'Total (s)':>9} | {'Comp (s)':>8} | {'Comm (s)':>8} | "
        f"{'Comm %':>6} | {'Speedup':>7} | {'Efficiency':>10} | {'Final SSE':>13} | {'Max Err':>10}"
    )
    print("-" * 106)

    strong_csv = os.path.join(RESULTS_DIR, "mpi_strong_scaling.csv")
    t1_strong = None

    with open(strong_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "P", "N_total", "N_per_rank", "total_time_s", "comp_time_s", "comm_time_s",
            "comm_pct", "avg_iter_ms", "throughput_mpts_s", "speedup", "efficiency_pct", "max_abs_err", "final_sse"
        ])

        for p in ranks_list:
            tot_s, comp_s, comm_s, comm_pct, iter_ms, thr, sse, max_err = run_mpi(
                p, ["-f", DATA_CSV, "-r", REF_TXT, "-k", "8", "-m", "50", "-t", "0.0"],
                oversubscribe=args.oversubscribe,
            )
            if t1_strong is None:
                t1_strong = tot_s
            speedup = t1_strong / tot_s
            eff_pct = (speedup / p) * 100.0
            n_per_rank = 597638 // p

            writer.writerow([
                p, 597638, n_per_rank, f"{tot_s:.4f}", f"{comp_s:.4f}", f"{comm_s:.4f}",
                f"{comm_pct:.1f}", f"{iter_ms:.2f}", f"{thr:.2f}", f"{speedup:.2f}",
                f"{eff_pct:.1f}", f"{max_err:.3e}", f"{sse:.6e}"
            ])
            print(
                f"{p:9d} | {tot_s:9.4f} | {comp_s:8.4f} | {comm_s:8.4f} | {comm_pct:5.1f}% | "
                f"{speedup:6.2f}x | {eff_pct:9.1f}% | {sse:13.6e} | {max_err:10.3e}"
            )

    print("\n========================================================================================")
    print("  Step 2B: Separate vs. Packed Single-Call MPI_Allreduce Benchmark")
    print("========================================================================================")
    allreduce_csv = os.path.join(RESULTS_DIR, "mpi_allreduce_comparison.csv")
    with open(allreduce_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["P", "allreduce_mode", "total_time_s", "comp_time_s", "comm_time_s", "comm_pct", "final_sse", "max_abs_err"])
        for p in ranks_list:
            for mode in ["separate", "packed"]:
                tot_s, comp_s, comm_s, comm_pct, _, _, sse, max_err = run_mpi(
                    p, ["-f", DATA_CSV, "-r", REF_TXT, "-k", "8", "-m", "50", "-t", "0.0", "-allreduce", mode],
                    oversubscribe=args.oversubscribe,
                )
                writer.writerow([p, mode, f"{tot_s:.4f}", f"{comp_s:.4f}", f"{comm_s:.4f}", f"{comm_pct:.1f}", f"{sse:.6e}", f"{max_err:.3e}"])
                print(f"  P={p:3d} | Mode={mode:<8} | Total={tot_s:.4f}s | Comm={comm_s:.4f}s ({comm_pct:.1f}%) | SSE={sse:.6e}")

    print("\n========================================================================================")
    print("  Step 2C: Pure MPI Weak Scaling (Fixed N_per_rank=200,000 pts/rank, d=28, K=8, 30 Iters)")
    print("========================================================================================")
    print(
        f"{'Ranks (P)':>9} | {'Total Points N':>14} | {'Total (s)':>9} | {'Comp (s)':>8} | "
        f"{'Comm (s)':>8} | {'Comm %':>6} | {'Weak Efficiency':>15} | {'Throughput':>13}"
    )
    print("-" * 100)

    weak_csv = os.path.join(RESULTS_DIR, "mpi_weak_scaling.csv")
    t1_weak = None
    n_per_rank_weak = 200000

    with open(weak_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "P", "N_total", "N_per_rank", "total_time_s", "comp_time_s", "comm_time_s",
            "comm_pct", "avg_iter_ms", "throughput_mpts_s", "weak_efficiency_pct", "final_sse"
        ])

        for p in ranks_list:
            tot_s, comp_s, comm_s, comm_pct, iter_ms, thr, sse, _ = run_mpi(
                p, ["-w", "-n", str(n_per_rank_weak), "-d", "28", "-k", "8", "-m", "30", "-t", "0.0"],
                oversubscribe=args.oversubscribe,
            )
            if t1_weak is None:
                t1_weak = tot_s
            weak_eff_pct = (t1_weak / tot_s) * 100.0
            n_total = n_per_rank_weak * p

            writer.writerow([
                p, n_total, n_per_rank_weak, f"{tot_s:.4f}", f"{comp_s:.4f}", f"{comm_s:.4f}",
                f"{comm_pct:.1f}", f"{iter_ms:.2f}", f"{thr:.2f}", f"{weak_eff_pct:.1f}", f"{sse:.6e}"
            ])
            print(
                f"{p:9d} | {n_total:14,d} | {tot_s:9.4f} | {comp_s:8.4f} | {comm_s:8.4f} | "
                f"{comm_pct:5.1f}% | {weak_eff_pct:14.1f}% | {thr:8.2f} M/s"
            )

    print("\nSaved MPI scaling results to results/mpi_strong_scaling.csv, results/mpi_allreduce_comparison.csv, and results/mpi_weak_scaling.csv.")


if __name__ == "__main__":
    main()
