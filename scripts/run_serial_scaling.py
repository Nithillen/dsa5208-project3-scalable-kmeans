#!/usr/bin/env python3
"""
Step 1: Serial Algorithm Runtime Scaling Benchmark on the Full UCI HIGGS Dataset
(https://archive.ics.uci.edu/dataset/280/higgs, N = 11,000,000 rows x 28 features)

Measures:
1. Reference 50-iteration baseline (creating data/higgs_clean.csv.bin cache & results/centroids_serial.txt)
2. Problem size scaling: Varying N up to full dataset (K=8, d=28, iters=30)
3. Cluster count scaling: Varying K in {2, 4, 8, 16, 32} on full dataset (d=28, iters=30)
Saves CSV results to results/serial_scaling_N.csv and results/serial_scaling_K.csv.
"""
import subprocess
import re
import csv
import os
import struct

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_PATH = os.path.join(BASE_DIR, "bin", "kmeans_serial")
DATA_CSV = os.path.join(BASE_DIR, "data", "higgs_clean.csv")
LABEL_TXT = os.path.join(BASE_DIR, "data", "higgs_labels.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_total_rows():
    bin_path = DATA_CSV + ".bin"
    if os.path.exists(bin_path) and os.path.exists(DATA_CSV):
        csv_size = os.path.getsize(DATA_CSV)
        with open(bin_path, "rb") as bf:
            hdr = bf.read(32)
            if len(hdr) == 32:
                magic, cached_csv_size, cached_N, cached_d = struct.unpack("<QQQQ", hdr)
                if magic == 0x4B4D45414E534231 and cached_csv_size == csv_size:
                    return int(cached_N)
    if os.path.exists(LABEL_TXT):
        with open(LABEL_TXT, "rb") as f:
            return sum(buf.count(b"\n") for buf in iter(lambda: f.read(8 * 1024 * 1024), b""))
    with open(DATA_CSV, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(8 * 1024 * 1024), b""))


def run_benchmark(N=None, K=8, max_iters=30, include_labels=False, out_file=None):
    cmd = [BIN_PATH, "-f", DATA_CSV]
    if include_labels and os.path.exists(LABEL_TXT):
        cmd.extend(["-l", LABEL_TXT])
    if N is not None:
        cmd.extend(["-n", str(N)])
    cmd.extend(["-k", str(K), "-m", str(max_iters), "-t", "0.0"])
    if out_file:
        cmd.extend(["-o", out_file])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout

    total_time = float(re.search(r"Total Time\s*:\s*([\d.]+)", out).group(1))
    avg_iter_ms = float(re.search(r"Avg Time/Iter\s*:\s*([\d.]+)", out).group(1))
    throughput = float(re.search(r"Throughput\s*:\s*([\d.]+)", out).group(1))
    sse_match = re.search(r"Final SSE \(WCSS\)\s*:\s*([\d.eE+-]+)", out)
    final_sse = float(sse_match.group(1)) if sse_match else 0.0
    n_match = re.search(r"Points \(N\)\s*:\s*(\d+)", out)
    actual_n = int(n_match.group(1)) if n_match else (N or 0)
    return total_time, avg_iter_ms, throughput, final_sse, actual_n, out


def main():
    # First run full-dataset 50-iteration reference (N=all rows, creates .bin cache for fast subsequent loads)
    ref_txt = os.path.join(RESULTS_DIR, "centroids_serial.txt")
    _, _, _, _, total_rows, ref_out = run_benchmark(
        N=None, K=8, max_iters=50, include_labels=True, out_file=ref_txt
    )

    print("==================================================================")
    print(f"  1-Node Serial Scaling Benchmark on UCI HIGGS (N={total_rows:,})")
    print("==================================================================")
    print(ref_out)

    # 1. Scaling with Problem Size N (rows of UCI HIGGS dataset)
    if total_rows >= 11000000:
        n_values = [100000, 500000, 1000000, 2500000, 5000000, total_rows]
    else:
        base_vals = [25000, 50000, 100000, 200000, 400000]
        n_values = [v for v in base_vals if v < total_rows] + [total_rows]

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
            tot_s, iter_ms, thr, _, _, _ = run_benchmark(N=n, K=k_fixed, max_iters=iters)
            writer.writerow([n, 28, k_fixed, iters, n * 28, f"{tot_s:.4f}", f"{iter_ms:.2f}", f"{thr:.2f}"])
            print(f"{n:12,d} | {n*28:14,d} | {tot_s:14.4f} | {iter_ms:13.2f} | {thr:20.2f}")

    # 2. Scaling with Number of Clusters K on full UCI HIGGS dataset
    k_values = [2, 4, 8, 16, 32]
    k_csv_path = os.path.join(RESULTS_DIR, "serial_scaling_K.csv")

    print(f"\n[B] Varying Cluster Count K (Fixed N={total_rows:,}, d=28, Iterations={iters}):")
    print(f"{'Clusters (K)':>12} | {'Dist Evals/Iter':>15} | {'Total Time (s)':>14} | {'Avg Iter (ms)':>13} | {'Throughput (M pts/s)':>20}")
    print("-" * 84)

    with open(k_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "d", "K", "iterations", "dist_evals_per_iter", "total_time_s", "avg_iter_ms", "throughput_mpts_s"])
        for k in k_values:
            tot_s, iter_ms, thr, _, _, _ = run_benchmark(N=None, K=k, max_iters=iters)
            writer.writerow([total_rows, 28, k, iters, total_rows * k, f"{tot_s:.4f}", f"{iter_ms:.2f}", f"{thr:.2f}"])
            print(f"{k:12d} | {total_rows*k:15,d} | {tot_s:14.4f} | {iter_ms:13.2f} | {thr:20.2f}")

    print("\nSaved benchmark tables to results/serial_scaling_N.csv, results/serial_scaling_K.csv, and results/centroids_serial.txt.")


if __name__ == "__main__":
    main()
