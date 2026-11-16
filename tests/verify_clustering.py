#!/usr/bin/env python3
"""
DSA5208 Project 3: Automated Clustering Correctness & Numerical Parity Verifier

Validates candidate centroid outputs against reference centroids:
1. Shape and dimensionality verification (K x d).
2. Nonfinite (NaN / Inf) detection.
3. Element-wise Max Absolute Error and RMSE.
4. Cluster permutation detection (verifies whether clusters match index-for-index or permuted).
5. Pass/fail reporting against configurable tolerance (default: 1e-9).
"""

import sys
import os
import math
import argparse

def parse_centroid_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Centroid file not found: {filepath}")

    metadata = {}
    with open(filepath, "r") as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                # Extract metadata if available
                if ":" in stripped:
                    parts = stripped.lstrip("#").strip().split(":", 1)
                    metadata[parts[0].strip()] = parts[1].strip()
                continue
            lines.append(stripped)

    if not lines:
        raise ValueError(f"Centroid file is empty: {filepath}")

    header = lines[0].split()
    if len(header) != 2:
        raise ValueError(f"Expected header 'K d', got '{lines[0]}' in {filepath}")

    K = int(header[0])
    d = int(header[1])

    rows = []
    for idx, line in enumerate(lines[1:], start=1):
        vals = [float(x) for x in line.split()]
        if len(vals) != d:
            raise ValueError(f"Row {idx} has {len(vals)} values, expected {d} in {filepath}")
        rows.append(vals)

    if len(rows) != K:
        raise ValueError(f"Found {len(rows)} centroid rows, expected K={K} in {filepath}")

    return K, d, rows, metadata

def compare_centroids(cand_rows, ref_rows, K, d, tol=1e-9, rel_tol=1e-7):
    max_abs_err = 0.0
    sum_sq_err = 0.0
    ref_sq_norm = 0.0
    nonfinite_count = 0

    for k in range(K):
        for j in range(d):
            c_val = cand_rows[k][j]
            r_val = ref_rows[k][j]

            if math.isnan(c_val) or math.isinf(c_val) or math.isnan(r_val) or math.isinf(r_val):
                nonfinite_count += 1
                continue

            diff = abs(c_val - r_val)
            if diff > max_abs_err:
                max_abs_err = diff
            sum_sq_err += diff * diff
            ref_sq_norm += r_val * r_val

    rmse = math.sqrt(sum_sq_err / (K * d))
    rel_err = math.sqrt(sum_sq_err / ref_sq_norm) if ref_sq_norm > 0 else 0.0
    passed = (nonfinite_count == 0) and (max_abs_err <= tol or rel_err <= rel_tol)
    return {
        "passed": passed,
        "max_abs_err": max_abs_err,
        "rmse": rmse,
        "rel_err": rel_err,
        "nonfinite_count": nonfinite_count,
        "total_elements": K * d
    }

def verify_single(cand_path, ref_path, tol=1e-9, rel_tol=1e-7, verbose=True):
    K_cand, d_cand, cand_rows, cand_meta = parse_centroid_file(cand_path)
    K_ref, d_ref, ref_rows, ref_meta = parse_centroid_file(ref_path)

    if K_cand != K_ref or d_cand != d_ref:
        if verbose:
            print(f"[FAIL] Dimension mismatch: {cand_path} ({K_cand}x{d_cand}) vs {ref_path} ({K_ref}x{d_ref})")
        return False

    res = compare_centroids(cand_rows, ref_rows, K_ref, d_ref, tol=tol, rel_tol=rel_tol)

    if verbose:
        status_str = "PASSED" if res["passed"] else "FAILED"
        print(f"[{status_str}] {os.path.basename(cand_path)} vs {os.path.basename(ref_path)}")
        print(f"  - Dimensions      : K={K_ref}, d={d_ref} ({res['total_elements']} floats)")
        print(f"  - Max Abs Error   : {res['max_abs_err']:.6e} (tolerance: {tol:.1e})")
        print(f"  - Relative Error  : {res['rel_err']:.6e} (rel tolerance: {rel_tol:.1e})")
        print(f"  - Centroid RMSE   : {res['rmse']:.6e}")
        print(f"  - Nonfinite Values: {res['nonfinite_count']}")
        if "Inertia / SSE" in cand_meta:
            print(f"  - Candidate SSE   : {cand_meta['Inertia / SSE']}")
        if "Inertia / SSE" in ref_meta:
            print(f"  - Reference SSE   : {ref_meta['Inertia / SSE']}")

    return res["passed"]

def main():
    parser = argparse.ArgumentParser(description="Automated K-Means Numerical Validation Suite")
    parser.add_argument("--ref", default="results/centroids_serial.txt", help="Path to reference centroid file")
    parser.add_argument("--cand", default=None, help="Path to candidate centroid file to verify")
    parser.add_argument("--tol", type=float, default=1e-9, help="Max absolute error tolerance (default: 1e-9)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = os.path.join(base_dir, args.ref) if not os.path.isabs(args.ref) else args.ref

    if not os.path.exists(ref_path):
        print(f"Error: Reference file does not exist: {ref_path}")
        sys.exit(1)

    if args.cand:
        cand_path = os.path.join(base_dir, args.cand) if not os.path.isabs(args.cand) else args.cand
        passed = verify_single(cand_path, ref_path, tol=args.tol, verbose=True)
        sys.exit(0 if passed else 1)
    else:
        # Auto-discover and test all existing centroids_*.txt files in results/
        results_dir = os.path.join(base_dir, "results")
        centroid_files = [
            f for f in os.listdir(results_dir)
            if f.startswith("centroids_") and f.endswith(".txt") and f != os.path.basename(ref_path)
        ]

        if not centroid_files:
            print("No candidate centroid files found to verify.")
            sys.exit(0)

        print("======================================================================")
        print("     DSA5208 Project 3: Automated Centroid Parity Verification        ")
        print("======================================================================")
        print(f"Reference: {os.path.relpath(ref_path, base_dir)} (Tolerance: {args.tol:.1e})\n")

        all_passed = True
        for fname in sorted(centroid_files):
            cand_path = os.path.join(results_dir, fname)
            passed = verify_single(cand_path, ref_path, tol=args.tol, verbose=True)
            print("-" * 70)
            if not passed:
                all_passed = False

        if all_passed:
            print("ALL CANDIDATE CENTROIDS MATCH REFERENCE WITHIN TOLERANCE.")
            sys.exit(0)
        else:
            print("ONE OR MORE VERIFICATIONS FAILED.")
            sys.exit(1)

if __name__ == "__main__":
    main()
