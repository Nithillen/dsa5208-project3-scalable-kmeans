#!/usr/bin/env python3
"""
DSA5208 Project 3: Serial Reference Manifest Manager & Verifier

Manages and verifies the provenance, parameters, and baseline metrics of
the 1-core serial reference clustering run. Refuses to reuse stale or
mismatched reference centroids if dataset checksum, K, iterations, or
initialization methods differ.
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone

def parse_serial_log(log_path):
    """Extract total elapsed time and throughput from serial execution log."""
    total_time = None
    throughput = None
    if not os.path.exists(log_path):
        return None, None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Total Time" in line and ":" in line:
                parts = line.split(":", 1)[1].strip().split()
                if parts:
                    try:
                        total_time = float(parts[0])
                    except ValueError:
                        pass
            elif "Throughput" in line and ":" in line:
                parts = line.split(":", 1)[1].strip().split()
                if parts:
                    try:
                        throughput = float(parts[0])
                    except ValueError:
                        pass
    return total_time, throughput

def create_manifest(args):
    """Generate and write a fresh serial reference manifest."""
    if not os.path.exists(args.ref_centroids):
        sys.stderr.write(f"[ERROR] Reference centroids file not found: {args.ref_centroids}\n")
        sys.exit(1)

    total_time, throughput = parse_serial_log(args.log_file)
    if total_time is None:
        sys.stderr.write(f"[ERROR] Could not parse Total Time from serial log: {args.log_file}\n")
        sys.exit(1)

    manifest_data = {
        "manifest_version": "1.0",
        "dataset_path": args.dataset_path,
        "dataset_md5": args.dataset_md5,
        "ref_centroids_path": args.ref_centroids,
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty,
        "K": int(args.k),
        "iterations": int(args.iterations),
        "tolerance": float(args.tolerance),
        "initialization_method": args.init_method,
        "numeric_precision": args.precision,
        "node_model": args.node_model,
        "serial_total_time_s": total_time,
        "serial_throughput_mpts_s": throughput if throughput is not None else 0.0,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[MANIFEST CREATED] {args.manifest}")
    print(f"  - Dataset MD5   : {args.dataset_md5}")
    print(f"  - Serial Time   : {total_time:.4f} s")
    print(f"  - Throughput    : {throughput or 0.0:.2f} M pts/s")
    sys.exit(0)

def verify_manifest(args):
    """Verify whether an existing manifest matches current dataset and run parameters."""
    if not os.path.exists(args.manifest):
        sys.stderr.write(f"[MANIFEST NOT FOUND] {args.manifest}\n")
        sys.exit(1)

    if not os.path.exists(args.ref_centroids):
        sys.stderr.write(f"[REF CENTROIDS NOT FOUND] {args.ref_centroids}\n")
        sys.exit(1)

    try:
        with open(args.manifest, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Corrupted manifest JSON: {e}\n")
        sys.exit(1)

    mismatches = []
    if m.get("dataset_md5") != args.dataset_md5:
        mismatches.append(f"dataset_md5 mismatch: manifest={m.get('dataset_md5')}, current={args.dataset_md5}")
    if int(m.get("K", -1)) != int(args.k):
        mismatches.append(f"K mismatch: manifest={m.get('K')}, current={args.k}")
    if int(m.get("iterations", -1)) != int(args.iterations):
        mismatches.append(f"iterations mismatch: manifest={m.get('iterations')}, current={args.iterations}")
    if abs(float(m.get("tolerance", -1.0)) - float(args.tolerance)) > 1e-9:
        mismatches.append(f"tolerance mismatch: manifest={m.get('tolerance')}, current={args.tolerance}")
    if m.get("initialization_method") != args.init_method:
        mismatches.append(f"init_method mismatch: manifest={m.get('initialization_method')}, expected={args.init_method}")
    if m.get("numeric_precision") != args.precision:
        mismatches.append(f"precision mismatch: manifest={m.get('numeric_precision')}, expected={args.precision}")

    if mismatches:
        sys.stderr.write("[ERROR] Serial reference manifest verification FAILED with mismatches:\n")
        for err in mismatches:
            sys.stderr.write(f"  - {err}\n")
        sys.stderr.write("Refusing to reuse stale or mismatched reference centroids!\n")
        sys.exit(1)

    print(f"[MANIFEST VERIFIED] Matches current dataset and run configuration.")
    print(f"serial_total_time_s={m.get('serial_total_time_s')}")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Manage Serial Reference Manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: create
    p_create = subparsers.add_parser("create")
    p_create.add_argument("--manifest", required=True, help="Path to manifest JSON")
    p_create.add_argument("--dataset-path", required=True, help="Path to cleaned dataset CSV")
    p_create.add_argument("--dataset-md5", required=True, help="MD5 checksum of dataset CSV")
    p_create.add_argument("--ref-centroids", required=True, help="Path to serial reference centroids file")
    p_create.add_argument("--log-file", required=True, help="Path to serial execution log")
    p_create.add_argument("--git-commit", default="N/A", help="Git HEAD revision hash")
    p_create.add_argument("--git-dirty", default="CLEAN", help="Git tree dirty state")
    p_create.add_argument("-k", "--k", "--clusters", type=int, default=8, help="Number of clusters")
    p_create.add_argument("--iterations", type=int, default=30, help="Iteration count")
    p_create.add_argument("--tolerance", type=float, default=0.0, help="Convergence tolerance")
    p_create.add_argument("--init-method", default="deterministic_stride_k_times_N_div_K", help="Init method")
    p_create.add_argument("--precision", default="float64_ieee754", help="Numeric precision")
    p_create.add_argument("--node-model", default="N/A", help="Compute node CPU model")

    # Subcommand: verify
    p_verify = subparsers.add_parser("verify")
    p_verify.add_argument("--manifest", required=True, help="Path to manifest JSON")
    p_verify.add_argument("--dataset-md5", required=True, help="Current dataset MD5")
    p_verify.add_argument("--ref-centroids", required=True, help="Path to serial reference centroids file")
    p_verify.add_argument("-k", "--k", "--clusters", type=int, default=8, help="Number of clusters")
    p_verify.add_argument("--iterations", type=int, default=30, help="Iteration count")
    p_verify.add_argument("--tolerance", type=float, default=0.0, help="Convergence tolerance")
    p_verify.add_argument("--init-method", default="deterministic_stride_k_times_N_div_K", help="Init method")
    p_verify.add_argument("--precision", default="float64_ieee754", help="Numeric precision")

    args = parser.parse_args()
    if args.command == "create":
        create_manifest(args)
    elif args.command == "verify":
        verify_manifest(args)

if __name__ == "__main__":
    main()
