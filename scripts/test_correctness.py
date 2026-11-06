#!/usr/bin/env python3
"""
Automated Correctness & Regression Test Suite for DSA5208 Project 3 (Scalable K-Means)
Verifies:
1. Identical synthetic dataset & centroid initialization across Serial, MPI, Standalone OpenMP, Hybrid MPI+OpenMP, and GPU.
2. Exact agreement of Final SSE / WCSS objective (J = sum ||x_i - mu_{a_i}||^2) and Final Centroids on both
   Synthetic data and Real UCI HIGGS data.
3. Identical numerical results between Separate vs. Packed single-call MPI_Allreduce (-allreduce packed)
   and Centralized Scatterv vs. Parallel Chunked I/O (-io parallel).
4. GPU Shared-Memory Bounds Guard when K * d > 1024 (e.g., K=40, d=32 -> K*d = 1280).
"""

import math
import os
import re
import shutil
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_centroid_file(filepath):
    with open(filepath, "r") as f:
        header = f.readline().strip().split()
        K = int(header[0])
        d = int(header[1])
        sse = float(header[2]) if len(header) >= 3 else None
        vals = []
        for line in f:
            vals.extend(float(x) for x in line.strip().split())
    return K, d, sse, vals


def compare_outputs(ref_path, test_path, label, tol_centroid=1e-9, tol_rel_sse=1e-11):
    K1, d1, sse1, c1 = load_centroid_file(ref_path)
    K2, d2, sse2, c2 = load_centroid_file(test_path)

    assert (K1, d1) == (K2, d2), f"[{label}] Dimension mismatch: {(K1, d1)} vs {(K2, d2)}"
    max_err = max(abs(a - b) for a, b in zip(c1, c2))
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)) / len(c1))
    rel_sse = abs(sse1 - sse2) / sse1 if (sse1 and sse2) else 0.0

    passed = (max_err <= tol_centroid) and (rel_sse <= tol_rel_sse)
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] {label:<42} | MaxErr: {max_err:.3e} | RMSE: {rmse:.3e} | "
        f"SSE: {sse2:.6e} (RelDiff: {rel_sse:.3e})"
    )
    if not passed:
        raise AssertionError(
            f"Validation failed for {label}: max_err={max_err:.3e} (tol={tol_centroid}), "
            f"rel_sse={rel_sse:.3e} (tol={tol_rel_sse})"
        )


def run_cmd(cmd):
    res = subprocess.run(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}")
    return res.stdout


def main():
    print("==========================================================================================")
    print("  DSA5208 Automated Correctness & Objective (SSE/WCSS) Verification Suite")
    print("==========================================================================================")

    tmp_dir = os.path.join(BASE_DIR, "results", "test_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # ------------------------------------------------------------------------------------------
    # Suite 1: Synthetic Mode Initialization & Convergence Parity (N=20,000, d=28, K=8)
    # ------------------------------------------------------------------------------------------
    print("\n[Suite 1] Synthetic Data Initialization & Convergence Parity (N=20,000, d=28, K=8)")
    ref_syn = os.path.join(tmp_dir, "syn_serial.txt")
    run_cmd(["./bin/kmeans_serial", "-s", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5", "-o", ref_syn])

    syn_cases = [
        ("Pure MPI (P=1, separate Allreduce)", ["mpirun", "-np", "1", "./bin/kmeans_mpi", "-s", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5"], 1e-9, 1e-11),
        ("Pure MPI (P=4, separate Allreduce)", ["mpirun", "-np", "4", "./bin/kmeans_mpi", "-s", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5"], 1e-9, 1e-11),
        ("Pure MPI (P=4, packed 1-call Allreduce)", ["mpirun", "-np", "4", "./bin/kmeans_mpi", "-s", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5", "-p"], 1e-9, 1e-11),
        ("Standalone OpenMP (T=4, no MPI)", ["./bin/kmeans_omp_standalone", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5", "-threads", "4"], 1e-9, 1e-11),
        ("Hybrid MPI+OpenMP (P=2, T=2, private)", ["mpirun", "-np", "2", "./bin/kmeans_omp", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5", "-threads", "2", "-a", "private"], 1e-9, 1e-11),
        ("Hybrid MPI+OpenMP (P=2, T=2, packed)", ["mpirun", "-np", "2", "./bin/kmeans_omp", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5", "-threads", "2", "-p"], 1e-9, 1e-11),
        ("Hybrid MPI+OpenMP (P=2, T=2, atomic)", ["mpirun", "-np", "2", "./bin/kmeans_omp", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5", "-threads", "2", "-a", "atomic"], 1e-9, 1e-11),
    ]

    if os.path.exists(os.path.join(BASE_DIR, "bin", "kmeans_metal_gpu")):
        syn_cases.append(
            ("Hybrid MPI+Metal GPU (P=2, T=2)", ["mpirun", "-np", "2", "./bin/kmeans_metal_gpu", "-s", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5", "-threads", "2"], 1e-3, 1e-5)
        )
    if os.path.exists(os.path.join(BASE_DIR, "bin", "kmeans_cuda")):
        syn_cases.append(
            ("Hybrid MPI+CUDA GPU (P=1)", ["mpirun", "-np", "1", "./bin/kmeans_cuda", "-s", "-n", "20000", "-d", "28", "-k", "8", "-m", "30", "-t", "1e-5"], 1e-7, 1e-9)
        )

    for idx, (label, cmd, tol_c, tol_s) in enumerate(syn_cases):
        out_path = os.path.join(tmp_dir, f"syn_case_{idx}.txt")
        run_cmd(cmd + ["-r", ref_syn, "-o", out_path])
        compare_outputs(ref_syn, out_path, label, tol_centroid=tol_c, tol_rel_sse=tol_s)

    # ------------------------------------------------------------------------------------------
    # Suite 2: Real UCI HIGGS Dataset Parity (Scatterv vs Parallel I/O, Separate vs Packed)
    # ------------------------------------------------------------------------------------------
    higgs_csv = os.path.join(BASE_DIR, "data", "higgs_clean.csv")
    if os.path.exists(higgs_csv):
        print("\n[Suite 2] Real UCI HIGGS Dataset Parity (N=50,000, d=28, K=8)")
        ref_higgs = os.path.join(tmp_dir, "higgs_serial.txt")
        run_cmd(["./bin/kmeans_serial", "-f", higgs_csv, "-n", "50000", "-k", "8", "-m", "25", "-t", "1e-5", "-o", ref_higgs])

        higgs_cases = [
            ("Pure MPI (P=4, Scatterv, Separate)", ["mpirun", "-np", "4", "./bin/kmeans_mpi", "-f", higgs_csv, "-n", "50000", "-k", "8", "-m", "25", "-t", "1e-5"], 1e-9, 1e-11),
            ("Pure MPI (P=4, Scatterv, Packed)", ["mpirun", "-np", "4", "./bin/kmeans_mpi", "-f", higgs_csv, "-n", "50000", "-k", "8", "-m", "25", "-t", "1e-5", "-p"], 1e-9, 1e-11),
            ("Pure MPI (P=4, Parallel I/O, Packed)", ["mpirun", "-np", "4", "./bin/kmeans_mpi", "-f", higgs_csv, "-n", "50000", "-k", "8", "-m", "25", "-t", "1e-5", "-io", "parallel", "-p"], 1e-9, 1e-11),
            ("Standalone OpenMP (T=4, HIGGS)", ["./bin/kmeans_omp_standalone", "-f", higgs_csv, "-n", "50000", "-k", "8", "-m", "25", "-t", "1e-5", "-threads", "4"], 1e-9, 1e-11),
            ("Hybrid MPI+OpenMP (P=2, T=2, Parallel I/O)", ["mpirun", "-np", "2", "./bin/kmeans_omp", "-f", higgs_csv, "-n", "50000", "-k", "8", "-m", "25", "-t", "1e-5", "-threads", "2", "-io", "parallel", "-p"], 1e-9, 1e-11),
        ]

        if os.path.exists(os.path.join(BASE_DIR, "bin", "kmeans_metal_gpu")):
            higgs_cases.append(
                ("Hybrid MPI+Metal GPU (P=1, T=4, HIGGS)", ["mpirun", "-np", "1", "./bin/kmeans_metal_gpu", "-f", higgs_csv, "-n", "50000", "-k", "8", "-m", "25", "-t", "1e-5", "-threads", "4"], 1e-3, 1e-5)
            )

        for idx, (label, cmd, tol_c, tol_s) in enumerate(higgs_cases):
            out_path = os.path.join(tmp_dir, f"higgs_case_{idx}.txt")
            run_cmd(cmd + ["-r", ref_higgs, "-o", out_path])
            compare_outputs(ref_higgs, out_path, label, tol_centroid=tol_c, tol_rel_sse=tol_s)

    # ------------------------------------------------------------------------------------------
    # Suite 3: High-Capacity Centroid Bounds Test (K=40, d=32 -> K*d = 1280 > 1024)
    # ------------------------------------------------------------------------------------------
    print("\n[Suite 3] Shared-Memory Bounds Guard Test (K=40, d=32 -> K*d = 1280 > 1024)")
    ref_large = os.path.join(tmp_dir, "large_serial.txt")
    run_cmd(["./bin/kmeans_serial", "-s", "-n", "10000", "-d", "32", "-k", "40", "-m", "20", "-t", "1e-5", "-o", ref_large])

    out_mpi_large = os.path.join(tmp_dir, "large_mpi.txt")
    run_cmd(["mpirun", "-np", "4", "./bin/kmeans_mpi", "-s", "-n", "10000", "-d", "32", "-k", "40", "-m", "20", "-t", "1e-5", "-p", "-r", ref_large, "-o", out_mpi_large])
    compare_outputs(ref_large, out_mpi_large, "Pure MPI (K=40, d=32, K*d=1280)", tol_centroid=1e-9, tol_rel_sse=1e-11)

    if os.path.exists(os.path.join(BASE_DIR, "bin", "kmeans_metal_gpu")):
        out_gpu_large = os.path.join(tmp_dir, "large_metal.txt")
        run_cmd(["mpirun", "-np", "1", "./bin/kmeans_metal_gpu", "-s", "-n", "10000", "-d", "32", "-k", "40", "-m", "20", "-t", "1e-5", "-r", ref_large, "-o", out_gpu_large])
        compare_outputs(ref_large, out_gpu_large, "Metal GPU Fallback (K*d=1280 > 1024)", tol_centroid=1e-3, tol_rel_sse=1e-5)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n==========================================================================================")
    print("  ALL CORRECTNESS, INITIALIZATION, AND SSE/WCSS VALIDATION TESTS PASSED 100%!")
    print("==========================================================================================")


if __name__ == "__main__":
    main()
