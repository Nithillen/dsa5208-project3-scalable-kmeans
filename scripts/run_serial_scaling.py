#!/usr/bin/env python3
"""
Step 1: Serial Algorithm Runtime Scaling Benchmark on the UCI HIGGS Dataset
(https://archive.ics.uci.edu/dataset/280/higgs)

Measures:
1. Problem size scaling: Varying N in {25000, 50000, 100000, 200000, 400000, 597638} with fixed K=8, d=28, iters=30
2. Cluster count scaling: Varying K in {2, 4, 8, 16, 32} with fixed N=597638, d=28, iters=30
Saves CSV results to results/serial_scaling_N.csv and results/serial_scaling_K.csv.
"""
import subprocess
import re
import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_PATH = os.path.join(BASE_DIR, "bin", "kmeans_serial")
DATA_CSV = os.path.join(BASE_DIR, "data", "higgs_clean.csv")
LABEL_TXT = os.path.join(BASE_DIR, "data", "higgs_labels.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_benchmark(N, K, max_iters=30):
    cmd = [
        BIN_PATH,
        "-f", DATA_CSV,
        "-l", LABEL_TXT,
        "-n", str(N),
        "-k", str(K),
        "-m", str(max_iters),
        "-t", "0.0"  # Force exact max_iters for consistent per-iteration runtime comparison
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout

    total_time = float(re.search(r"Total Time\s*:\s*([\d.]+)", out).group(1))
    avg_iter_ms = float(re.search(r"Avg Time/Iter\s*:\s*([\d.]+)", out).group(1))
    throughput = float(re.search(r"Throughput\s*:\s*([\d.]+)", out).group(1))
    return total_time, avg_iter_ms, throughput

def main():
    print("==================================================================")
    print("  1-Node Serial Scaling Benchmark on UCI HIGGS Dataset (ID #280)  ")
    print("==================================================================")

    # 1. Scaling with Problem Size N (rows of UCI HIGGS dataset)
    n_values = [25000, 50000, 100000, 200000, 400000, 597638]
    k_fixed = 8
    iters = 30
    n_csv_path = os.path.join(RESULTS_DIR, "serial_scaling_N.csv")

    print(f"\n[A] Varying Problem Size N (Fixed K={k_fixed}, d=28, Iterations={iters}):")
    print(f"{'Points (N)':>12} | {'Total Floats':>14} | {'Total Time (s)':>14} | {'Avg Iter (ms)':>13} | {'Throughput (M pts/s)':>20}")
    print("-" * 83)

    with open(n_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "d", "K", "iterations", "total_floats", "total_time_s", "avg_iter_ms", "throughput_mpts_s"])
        for n in n_values:
            tot_s, iter_ms, thr = run_benchmark(n, k_fixed, iters)
            writer.writerow([n, 28, k_fixed, iters, n * 28, f"{tot_s:.4f}", f"{iter_ms:.2f}", f"{thr:.2f}"])
            print(f"{n:12,d} | {n*28:14,d} | {tot_s:14.4f} | {iter_ms:13.2f} | {thr:20.2f}")

    # 2. Scaling with Number of Clusters K (full UCI HIGGS dataset N=597,638)
    k_values = [2, 4, 8, 16, 32]
    n_fixed = 597638
    k_csv_path = os.path.join(RESULTS_DIR, "serial_scaling_K.csv")

    print(f"\n[B] Varying Cluster Count K (Fixed N={n_fixed:,}, d=28, Iterations={iters}):")
    print(f"{'Clusters (K)':>12} | {'Dist Evals/Iter':>15} | {'Total Time (s)':>14} | {'Avg Iter (ms)':>13} | {'Throughput (M pts/s)':>20}")
    print("-" * 84)

    with open(k_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "d", "K", "iterations", "dist_evals_per_iter", "total_time_s", "avg_iter_ms", "throughput_mpts_s"])
        for k in k_values:
            tot_s, iter_ms, thr = run_benchmark(n_fixed, k, iters)
            writer.writerow([n_fixed, 28, k, iters, n_fixed * k, f"{tot_s:.4f}", f"{iter_ms:.2f}", f"{thr:.2f}"])
            print(f"{k:12d} | {n_fixed*k:15,d} | {tot_s:14.4f} | {iter_ms:13.2f} | {thr:20.2f}")

    # 3. Run full 50-iteration reference run with K=8 to save reference centroids_serial.txt
    subprocess.run([
        BIN_PATH, "-f", DATA_CSV, "-l", LABEL_TXT, "-k", "8", "-m", "50", "-t", "1e-5",
        "-o", os.path.join(RESULTS_DIR, "centroids_serial.txt")
    ], capture_output=True, text=True, check=True)
    print("\nSaved benchmark tables to results/serial_scaling_N.csv and results/serial_scaling_K.csv.")

if __name__ == "__main__":
    main()
