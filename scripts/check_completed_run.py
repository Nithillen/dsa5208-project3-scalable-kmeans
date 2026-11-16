#!/usr/bin/env python3
"""
DSA5208 Project 3: Structured CSV Check for Completed Benchmark Configurations

Uses Python's csv.DictReader to safely determine whether a given
(P, repetition) configuration has already been executed and validated
with validation_status == 'PASSED'.
Exits 0 if completed and valid (safe to skip).
Exits 1 if not found, invalid, or incomplete (must execute).
"""

import sys
import os
import csv
import argparse

def is_run_completed(csv_path, p_target, rep_target):
    if not os.path.exists(csv_path):
        return False

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p_val = int(row.get("P", -1))
                rep_val = int(row.get("repetition", -1))
                val_status = row.get("validation_status", "").strip().upper()
                if p_val == p_target and rep_val == rep_target and val_status == "PASSED":
                    return True
            except (ValueError, TypeError):
                continue
    return False

def main():
    parser = argparse.ArgumentParser(description="Check if benchmark run completed in raw CSV")
    parser.add_argument("--csv", required=True, help="Path to raw benchmark CSV")
    parser.add_argument("-p", type=int, required=True, help="Number of MPI ranks")
    parser.add_argument("--rep", type=int, required=True, help="Repetition number")
    args = parser.parse_args()

    if is_run_completed(args.csv, args.p, args.rep):
        sys.exit(0)  # Found and PASSED
    else:
        sys.exit(1)  # Not completed or not valid

if __name__ == "__main__":
    main()
