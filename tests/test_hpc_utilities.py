#!/usr/bin/env python3
"""
Unit and integration test suite for DSA5208 HPC benchmark utilities:
1. manage_reference_manifest.py (create, verify, reject on mismatch)
2. check_completed_run.py (PASSED vs FAILED vs missing)
3. verify_clustering.py (exact match, numerical tolerance, dimension error, nonfinite)
4. compute_scaling_metrics.py (median aggregation, speedup, efficiency, exclusion of invalid runs)
"""

import sys
import os
import shutil
import tempfile
import subprocess
import json
import csv

def run_cmd(cmd_list):
    res = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout, res.stderr

def test_suite():
    temp_dir = tempfile.mkdtemp(prefix="kmeans_test_")
    print(f"Running unit tests in temporary sandbox: {temp_dir}")
    try:
        # -------------------------------------------------------------
        # Test 1: verify_clustering.py
        # -------------------------------------------------------------
        print("\n--- Test 1: verify_clustering.py ---")
        ref_cent_path = os.path.join(temp_dir, "ref_cent.txt")
        cand_match_path = os.path.join(temp_dir, "cand_match.txt")
        cand_diff_path = os.path.join(temp_dir, "cand_diff.txt")
        cand_nan_path = os.path.join(temp_dir, "cand_nan.txt")

        header = "2 3\n"
        row1 = "1.0 2.0 3.0\n"
        row2 = "4.0 5.0 6.0\n"

        with open(ref_cent_path, "w") as f:
            f.write(header + row1 + row2)
        with open(cand_match_path, "w") as f:
            f.write(header + row1 + row2)
        with open(cand_diff_path, "w") as f:
            f.write(header + "1.0 2.0 3.001\n" + row2)
        with open(cand_nan_path, "w") as f:
            f.write(header + "1.0 nan 3.0\n" + row2)

        # 1.1: Exact match should exit 0
        code, out, err = run_cmd([sys.executable, "tests/verify_clustering.py", "--cand", cand_match_path, "--ref", ref_cent_path])
        assert code == 0, f"Expected 0, got {code}. Out: {out}, Err: {err}"
        assert "PASSED" in out
        print("  [PASS] Exact match verification succeeded.")

        # 1.2: Numerical difference > 1e-9 should exit 1
        code, out, err = run_cmd([sys.executable, "tests/verify_clustering.py", "--cand", cand_diff_path, "--ref", ref_cent_path, "--tol", "1e-9"])
        assert code == 1, f"Expected 1, got {code}."
        assert "FAILED" in out
        print("  [PASS] Difference > 1e-9 correctly rejected.")

        # 1.3: Non-finite value should exit 1
        code, out, err = run_cmd([sys.executable, "tests/verify_clustering.py", "--cand", cand_nan_path, "--ref", ref_cent_path])
        assert code == 1, f"Expected 1, got {code}."
        assert "FAILED" in out
        print("  [PASS] NaN correctly rejected.")

        # -------------------------------------------------------------
        # Test 2: manage_reference_manifest.py
        # -------------------------------------------------------------
        print("\n--- Test 2: manage_reference_manifest.py ---")
        mock_log = os.path.join(temp_dir, "serial_mock.log")
        with open(mock_log, "w") as f:
            f.write("Iteration 30 | Max Centroid Shift: 0.0000e+00 | Time: 12.34 ms\n")
            f.write("Total Time       : 125.5000 seconds\n")
            f.write("Throughput       : 2.63 M points/sec\n")

        manifest_path = os.path.join(temp_dir, "serial_manifest.json")

        # 2.1: Create manifest
        code, out, err = run_cmd([
            sys.executable, "scripts/manage_reference_manifest.py", "create",
            "--manifest", manifest_path,
            "--dataset-path", "data/test.csv",
            "--dataset-md5", "abc123456",
            "--ref-centroids", ref_cent_path,
            "--log-file", mock_log,
            "--git-commit", "870d6fd",
            "--git-dirty", "CLEAN",
            "-k", "2",
            "--iterations", "30",
            "--tolerance", "0.0",
            "--init-method", "deterministic_stride_k_times_N_div_K",
            "--precision", "float64_ieee754",
            "--node-model", "Test Xeon"
        ])
        assert code == 0, f"Create manifest failed: {err}"
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            m = json.load(f)
            assert m["serial_total_time_s"] == 125.5
            assert m["dataset_md5"] == "abc123456"
        print("  [PASS] Manifest creation succeeded.")

        # 2.2: Verify manifest with matching parameters -> exit 0
        code, out, err = run_cmd([
            sys.executable, "scripts/manage_reference_manifest.py", "verify",
            "--manifest", manifest_path,
            "--dataset-md5", "abc123456",
            "--ref-centroids", ref_cent_path,
            "-k", "2",
            "--iterations", "30",
            "--tolerance", "0.0"
        ])
        assert code == 0, f"Verify failed: {err}"
        assert "serial_total_time_s=125.5" in out
        print("  [PASS] Manifest verification with matching parameters succeeded.")

        # 2.3: Verify manifest with MD5 mismatch -> exit 1
        code, out, err = run_cmd([
            sys.executable, "scripts/manage_reference_manifest.py", "verify",
            "--manifest", manifest_path,
            "--dataset-md5", "DIFFERENT_MD5",
            "--ref-centroids", ref_cent_path,
            "-k", "2",
            "--iterations", "30",
            "--tolerance", "0.0"
        ])
        assert code == 1, "Expected exit 1 on MD5 mismatch"
        assert "dataset_md5 mismatch" in err
        print("  [PASS] MD5 mismatch correctly rejected.")

        # 2.4: Verify manifest with K mismatch -> exit 1
        code, out, err = run_cmd([
            sys.executable, "scripts/manage_reference_manifest.py", "verify",
            "--manifest", manifest_path,
            "--dataset-md5", "abc123456",
            "--ref-centroids", ref_cent_path,
            "-k", "8",
            "--iterations", "30",
            "--tolerance", "0.0"
        ])
        assert code == 1, "Expected exit 1 on K mismatch"
        assert "K mismatch" in err
        print("  [PASS] K mismatch correctly rejected.")

        # -------------------------------------------------------------
        # Test 3: check_completed_run.py
        # -------------------------------------------------------------
        print("\n--- Test 3: check_completed_run.py ---")
        mock_raw_csv = os.path.join(temp_dir, "raw_bench.csv")
        with open(mock_raw_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id","platform","node_model","P","repetition","N","d","K","iterations","total_time_s","comp_time_s","comm_time_s","comm_pct","throughput_mpts_s","max_abs_err","centroid_rmse","validation_status","git_commit","git_dirty","data_md5"])
            writer.writerow([1,"Vanda","Xeon",1,1,11000000,28,8,30,120.0,119.5,0.5,0.4,2.75,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            writer.writerow([2,"Vanda","Xeon",1,2,11000000,28,8,30,121.0,120.5,0.5,0.4,2.73,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            writer.writerow([3,"Vanda","Xeon",2,1,11000000,28,8,30,62.0,61.0,1.0,1.6,5.32,0.0,0.0,"FAILED_VALIDATION","870d6fd","CLEAN","abc123456"])

        # 3.1: Check completed (P=1, Rep=1) -> exit 0
        code, out, err = run_cmd([sys.executable, "scripts/check_completed_run.py", "--csv", mock_raw_csv, "-p", "1", "--rep", "1"])
        assert code == 0, f"Expected 0 for completed run, got {code}"
        print("  [PASS] Completed and PASSED run correctly detected.")

        # 3.2: Check failed run (P=2, Rep=1) -> exit 1
        code, out, err = run_cmd([sys.executable, "scripts/check_completed_run.py", "--csv", mock_raw_csv, "-p", "2", "--rep", "1"])
        assert code == 1, f"Expected 1 for failed run, got {code}"
        print("  [PASS] FAILED_VALIDATION run correctly rejected.")

        # 3.3: Check unstarted run (P=4, Rep=1) -> exit 1
        code, out, err = run_cmd([sys.executable, "scripts/check_completed_run.py", "--csv", mock_raw_csv, "-p", "4", "--rep", "1"])
        assert code == 1, f"Expected 1 for unstarted run, got {code}"
        print("  [PASS] Unstarted run correctly reported.")

        # -------------------------------------------------------------
        # Test 4: compute_scaling_metrics.py
        # -------------------------------------------------------------
        print("\n--- Test 4: compute_scaling_metrics.py ---")
        # Add more valid rows to mock_raw_csv: P=1 (3 reps: 120, 121, 119 -> median 120)
        # P=2 (3 reps: 62, 60, 61 -> median 61)
        # P=4 (3 reps: 32, 31, 30 -> median 31)
        with open(mock_raw_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id","platform","node_model","P","repetition","N","d","K","iterations","total_time_s","comp_time_s","comm_time_s","comm_pct","throughput_mpts_s","max_abs_err","centroid_rmse","validation_status","git_commit","git_dirty","data_md5"])
            # P=1
            writer.writerow([1,"Vanda","Xeon",1,1,11000000,28,8,30,120.0,119.5,0.5,0.4,2.75,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            writer.writerow([2,"Vanda","Xeon",1,2,11000000,28,8,30,121.0,120.5,0.5,0.4,2.73,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            writer.writerow([3,"Vanda","Xeon",1,3,11000000,28,8,30,119.0,118.5,0.5,0.4,2.77,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            # P=2
            writer.writerow([4,"Vanda","Xeon",2,1,11000000,28,8,30,62.0,60.8,1.2,1.9,5.32,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            writer.writerow([5,"Vanda","Xeon",2,2,11000000,28,8,30,60.0,58.9,1.1,1.8,5.50,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            writer.writerow([6,"Vanda","Xeon",2,3,11000000,28,8,30,61.0,59.8,1.2,2.0,5.41,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            # P=4 (one failed run, two passed)
            writer.writerow([7,"Vanda","Xeon",4,1,11000000,28,8,30,31.0,29.5,1.5,4.8,10.64,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            writer.writerow([8,"Vanda","Xeon",4,2,11000000,28,8,30,30.0,28.6,1.4,4.7,11.00,0.0,0.0,"PASSED","870d6fd","CLEAN","abc123456"])
            writer.writerow([9,"Vanda","Xeon",4,3,11000000,28,8,30,35.0,33.0,2.0,5.7,9.40,0.05,0.02,"FAILED_VALIDATION","870d6fd","CLEAN","abc123456"])

        summary_csv = os.path.join(temp_dir, "summary.csv")
        code, out, err = run_cmd([
            sys.executable, "scripts/compute_scaling_metrics.py",
            "--raw-csv", mock_raw_csv,
            "--manifest", manifest_path,
            "--output-summary", summary_csv
        ])
        assert code == 0, f"compute_scaling_metrics failed: {err}"
        assert os.path.exists(summary_csv)
        assert "Excluded 1 unverified or invalid run records" in out

        with open(summary_csv) as f:
            s_rows = list(csv.DictReader(f))
            assert len(s_rows) == 3 # P=1, P=2, P=4
            # Check P=1
            assert float(s_rows[0]["median_total_time_s"]) == 120.0
            assert float(s_rows[0]["mpi_speedup"]) == 1.0
            assert float(s_rows[0]["mpi_efficiency_pct"]) == 100.0
            assert round(float(s_rows[0]["serial_speedup"]), 4) == round(125.5 / 120.0, 4)

            # Check P=2: median is 61.0. Speedup = 120.0 / 61.0 = 1.9672
            assert float(s_rows[1]["median_total_time_s"]) == 61.0
            assert round(float(s_rows[1]["mpi_speedup"]), 4) == round(120.0 / 61.0, 4)
            assert round(float(s_rows[1]["mpi_efficiency_pct"]), 2) == round((120.0 / 61.0 / 2.0) * 100.0, 2)

            # Check P=4: 2 valid reps, median of (31.0, 30.0) is 30.5
            assert int(s_rows[2]["reps_completed"]) == 2
            assert float(s_rows[2]["median_total_time_s"]) == 30.5
            assert round(float(s_rows[2]["mpi_speedup"]), 4) == round(120.0 / 30.5, 4)
        print("  [PASS] Scaling metrics and median aggregation correctly computed.")

        print("\nALL 4 TEST SUITES PASSED PERFECTLY!\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_suite()
