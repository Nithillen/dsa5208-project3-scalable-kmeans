#!/usr/bin/env python3
"""
DSA5208 Project 3: Scaling Metrics & Postprocessing Generator

Reads raw benchmark execution records from the pilot CSV, excludes invalid
or unverified runs, aggregates multiple repetitions using median statistics,
and computes:
  - MPI-relative Speedup & Parallel Efficiency (baseline: median T_mpi at P=1)
  - Serial-relative Speedup (baseline: serial_total_time_s from reference manifest)
  - Communication overhead percentages and throughput scaling

Outputs a structured scaling summary CSV and displays a Markdown report table.
"""

import sys
import os
import csv
import json
import statistics
import argparse

def load_serial_manifest(manifest_path):
    if not manifest_path or not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        sys.stderr.write(f"[WARN] Could not parse serial manifest {manifest_path}: {e}\n")
        return None

def compute_scaling(raw_csv_path, manifest_path, output_summary_path):
    if not os.path.exists(raw_csv_path):
        sys.stderr.write(f"[ERROR] Raw benchmark CSV not found: {raw_csv_path}\n")
        sys.exit(1)

    manifest = load_serial_manifest(manifest_path)
    t_serial_ref = None
    if manifest and "serial_total_time_s" in manifest:
        try:
            t_serial_ref = float(manifest["serial_total_time_s"])
        except (ValueError, TypeError):
            pass

    # Read and validate raw rows
    rows_by_p = {}
    invalid_rows = []

    with open(raw_csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r_idx, row in enumerate(reader, start=2):
            status = row.get("validation_status", "").strip().upper()
            p_val = row.get("P", "").strip()

            if status != "PASSED":
                invalid_rows.append((r_idx, p_val, row.get("repetition", "?"), status))
                continue

            try:
                p = int(p_val)
                rep = int(row["repetition"])
                tot_t = float(row["total_time_s"])
                comp_t = float(row["comp_time_s"])
                comm_t = float(row["comm_time_s"])
                comm_pct = float(row["comm_pct"]) if row.get("comm_pct") else (comm_t / tot_t * 100.0 if tot_t > 0 else 0.0)
                thru = float(row["throughput_mpts_s"])
                err_val = float(row["max_abs_err"]) if row.get("max_abs_err") and row["max_abs_err"] != "N/A" else 0.0
            except (ValueError, KeyError) as e:
                invalid_rows.append((r_idx, p_val, row.get("repetition", "?"), f"Parse error: {e}"))
                continue

            if p not in rows_by_p:
                rows_by_p[p] = {
                    "platform": row.get("platform", "Vanda"),
                    "node_model": row.get("node_model", "Unknown"),
                    "N": row.get("N", "11000000"),
                    "d": row.get("d", "28"),
                    "K": row.get("K", "8"),
                    "iterations": row.get("iterations", "30"),
                    "data_md5": row.get("data_md5", "Unknown"),
                    "git_commit": row.get("git_commit", "N/A"),
                    "git_dirty": row.get("git_dirty", "N/A"),
                    "records": []
                }

            rows_by_p[p]["records"].append({
                "repetition": rep,
                "total_time_s": tot_t,
                "comp_time_s": comp_t,
                "comm_time_s": comm_t,
                "comm_pct": comm_pct,
                "throughput_mpts_s": thru,
                "max_abs_err": err_val
            })

    if invalid_rows:
        print(f"[NOTE] Excluded {len(invalid_rows)} unverified or invalid run records:")
        for r_line, p_str, rep_str, reason in invalid_rows:
            print(f"  - Line {r_line} (P={p_str}, Rep={rep_str}): {reason}")

    if not rows_by_p:
        sys.stderr.write("[ERROR] No valid runs with validation_status == 'PASSED' found in CSV.\n")
        sys.exit(1)

    sorted_p = sorted(rows_by_p.keys())

    # Baseline for MPI strong scaling: median total_time at P=1
    t_mpi_p1 = None
    if 1 in rows_by_p and len(rows_by_p[1]["records"]) > 0:
        p1_times = [rec["total_time_s"] for rec in rows_by_p[1]["records"]]
        t_mpi_p1 = statistics.median(p1_times)

    summary_rows = []
    for p in sorted_p:
        p_data = rows_by_p[p]
        recs = p_data["records"]
        n_reps = len(recs)

        tot_list = [r["total_time_s"] for r in recs]
        comp_list = [r["comp_time_s"] for r in recs]
        comm_list = [r["comm_time_s"] for r in recs]
        comm_pct_list = [r["comm_pct"] for r in recs]
        thru_list = [r["throughput_mpts_s"] for r in recs]
        err_list = [r["max_abs_err"] for r in recs]

        med_tot = statistics.median(tot_list)
        min_tot = min(tot_list)
        max_tot = max(tot_list)
        std_tot = statistics.stdev(tot_list) if n_reps > 1 else 0.0

        med_comp = statistics.median(comp_list)
        med_comm = statistics.median(comm_list)
        med_comm_pct = statistics.median(comm_pct_list)
        med_thru = statistics.median(thru_list)
        max_err = max(err_list)

        # MPI-relative speedup and efficiency
        if t_mpi_p1 is not None and med_tot > 0:
            mpi_speedup = t_mpi_p1 / med_tot
            mpi_eff = (mpi_speedup / p) * 100.0
        else:
            mpi_speedup = None
            mpi_eff = None

        # Serial-relative speedup
        if t_serial_ref is not None and med_tot > 0:
            serial_speedup = t_serial_ref / med_tot
        else:
            serial_speedup = None

        summary_rows.append({
            "platform": p_data["platform"],
            "node_model": p_data["node_model"],
            "P": p,
            "reps_completed": n_reps,
            "N": p_data["N"],
            "d": p_data["d"],
            "K": p_data["K"],
            "iterations": p_data["iterations"],
            "median_total_time_s": round(med_tot, 4),
            "min_total_time_s": round(min_tot, 4),
            "max_total_time_s": round(max_tot, 4),
            "std_total_time_s": round(std_tot, 4),
            "median_comp_time_s": round(med_comp, 4),
            "median_comm_time_s": round(med_comm, 4),
            "median_comm_pct": round(med_comm_pct, 2),
            "median_throughput_mpts_s": round(med_thru, 2),
            "mpi_speedup": round(mpi_speedup, 4) if mpi_speedup is not None else "N/A",
            "serial_speedup": round(serial_speedup, 4) if serial_speedup is not None else "N/A",
            "mpi_efficiency_pct": round(mpi_eff, 2) if mpi_eff is not None else "N/A",
            "max_abs_err": f"{max_err:.4e}",
            "validation_status": "PASSED",
            "data_md5": p_data["data_md5"],
            "git_commit": p_data["git_commit"],
            "git_dirty": p_data["git_dirty"]
        })

    # Write summary CSV
    out_dir = os.path.dirname(os.path.abspath(output_summary_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "platform", "node_model", "P", "reps_completed", "N", "d", "K", "iterations",
        "median_total_time_s", "min_total_time_s", "max_total_time_s", "std_total_time_s",
        "median_comp_time_s", "median_comm_time_s", "median_comm_pct", "median_throughput_mpts_s",
        "mpi_speedup", "serial_speedup", "mpi_efficiency_pct", "max_abs_err", "validation_status",
        "data_md5", "git_commit", "git_dirty"
    ]

    with open(output_summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s_row in summary_rows:
            writer.writerow(s_row)

    print(f"\n[SUMMARY WRITTEN] {output_summary_path}")
    print("\n" + "=" * 90)
    print("           DSA5208 VANDA PILOT: MPI STRONG SCALING SUMMARY REPORT")
    print("=" * 90)
    if t_serial_ref is not None:
        print(f"Serial Reference Baseline Time : {t_serial_ref:.4f} s (from manifest)")
    if t_mpi_p1 is not None:
        print(f"MPI P=1 Median Baseline Time   : {t_mpi_p1:.4f} s (from P=1 repetitions)")
    print("-" * 90)
    print(f"{'P':>3} | {'Reps':>4} | {'Med Total(s)':>12} | {'Med Comp(s)':>11} | {'Med Comm(s)':>11} | {'Comm%':>6} | {'MPI Spd':>8} | {'Ser Spd':>8} | {'Eff(%)':>7}")
    print("-" * 90)
    for r in summary_rows:
        mpi_spd_str = f"{r['mpi_speedup']}x" if r['mpi_speedup'] != 'N/A' else "N/A"
        ser_spd_str = f"{r['serial_speedup']}x" if r['serial_speedup'] != 'N/A' else "N/A"
        eff_str = f"{r['mpi_efficiency_pct']}%" if r['mpi_efficiency_pct'] != 'N/A' else "N/A"
        print(f"{r['P']:>3} | {r['reps_completed']:>4} | {r['median_total_time_s']:>12.4f} | {r['median_comp_time_s']:>11.4f} | {r['median_comm_time_s']:>11.4f} | {r['median_comm_pct']:>5.1f}% | {mpi_spd_str:>8} | {ser_spd_str:>8} | {eff_str:>7}")
    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Compute median-based scaling metrics from raw benchmark CSV")
    parser.add_argument("--raw-csv", required=True, help="Path to raw benchmark CSV")
    parser.add_argument("--manifest", default=None, help="Path to serial reference manifest JSON")
    parser.add_argument("--output-summary", required=True, help="Path to output summary CSV")
    args = parser.parse_args()

    compute_scaling(args.raw_csv, args.manifest, args.output_summary)

if __name__ == "__main__":
    main()
