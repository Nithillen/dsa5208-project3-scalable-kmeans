import urllib.request
import zlib
import os
import math
import time
import argparse

def prepare_higgs_dataset(target_rows=600000, full_precision=False, force=False,
                           custom_csv=None, custom_labels=None, archive_path=None):
    """
    Streams and cleans the UCI HIGGS Boson High-Energy Physics Dataset
    using fast chunked zlib decompression over HTTP or from local archive.

    Supports:
    - Default: 600,000 target rows (yielding ~597,638 clean events)
    - Full Dataset: target_rows=0 or target_rows=11000000 (all 11M rows)
    - Full precision mode (preserves original float strings without rounding)
    - Local archive reuse (reads existing HIGGS.csv.gz to avoid redownloading 2.6 GB)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_csv = custom_csv or os.path.join(data_dir, "higgs_clean.csv")
    output_labels = custom_labels or os.path.join(data_dir, "higgs_labels.txt")

    if not force and os.path.exists(output_csv) and os.path.exists(output_labels):
        print(f"HIGGS dataset already prepared at:\n  {output_csv}\n(Use --force to re-generate).")
        return

    # Check for existing local archive to avoid redundant network download
    local_archive = archive_path
    if not local_archive:
        potential_paths = [
            os.path.join(data_dir, "HIGGS.csv.gz"),
            os.path.join(base_dir, "HIGGS.csv.gz"),
            "/scratch/HIGGS.csv.gz"
        ]
        for p in potential_paths:
            if os.path.exists(p) and os.path.getsize(p) > 1024 * 1024 * 1024:
                local_archive = p
                break

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    print("======================================================================")
    print("  Streaming & Cleaning UCI HIGGS Boson Dataset (28 Physics Features)  ")
    print("======================================================================")
    if local_archive:
        print(f"Input Source     : Local archive on disk ({local_archive})")
    else:
        print(f"Input Source     : Upstream HTTP URL ({url})")
    target_desc = f"{target_rows:,} events" if target_rows > 0 else "ALL 11,000,000 events (Full Dataset)"
    print(f"Target Size      : {target_desc} x 28 features")
    print(f"Precision Mode   : {'Full Double Precision' if full_precision else '5-decimal truncation (Legacy baseline)'}")
    print(f"Output Feature CSV: {output_csv}")
    print(f"Output Labels TXT : {output_labels}")

    t0 = time.time()
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)

    valid_rows = 0
    skipped_rows = 0
    higgs_signal_count = 0
    background_count = 0
    remainder = b""
    last_log_time = t0

    if local_archive:
        source_stream = open(local_archive, 'rb')
    else:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        source_stream = urllib.request.urlopen(req, timeout=180)

    with source_stream as response, \
         open(output_csv, 'w', buffering=16*1024*1024) as feat_file, \
         open(output_labels, 'w', buffering=4*1024*1024) as label_file:

        while True:
            if target_rows > 0 and valid_rows >= target_rows:
                break

            chunk = response.read(1024 * 1024) # Read 1 MB compressed chunks
            if not chunk:
                break
            decompressed = remainder + decompressor.decompress(chunk)
            lines = decompressed.split(b'\n')
            remainder = lines.pop()

            for raw_line in lines:
                if not raw_line:
                    continue
                try:
                    line = raw_line.decode('ascii').strip()
                except UnicodeDecodeError:
                    skipped_rows += 1
                    continue

                tokens = line.split(',')
                if len(tokens) != 29:
                    skipped_rows += 1
                    continue

                try:
                    label_val = float(tokens[0])
                    label_int = 1 if label_val >= 0.5 else 0

                    features = []
                    valid = True
                    for tok in tokens[1:29]:
                        val = float(tok)
                        if math.isnan(val) or math.isinf(val):
                            valid = False
                            break
                        if full_precision:
                            features.append(tok.strip())
                        else:
                            features.append(f"{val:.5f}")

                    if not valid:
                        skipped_rows += 1
                        continue

                    feat_file.write(",".join(features) + "\n")
                    label_file.write(f"{label_int}\n")

                    if label_int == 1:
                        higgs_signal_count += 1
                    else:
                        background_count += 1

                    valid_rows += 1

                    if valid_rows % 500000 == 0:
                        cur_time = time.time()
                        rate = valid_rows / (cur_time - t0)
                        print(f"  Processed {valid_rows:,} clean events ({rate:,.0f} rows/s)...")

                    if target_rows > 0 and valid_rows >= target_rows:
                        break
                except ValueError:
                    skipped_rows += 1
                    continue

    total_time = time.time() - t0
    csv_size_mb = os.path.getsize(output_csv) / (1024.0 * 1024.0)
    print("----------------------------------------------------------------------")
    print(f"Completed: Saved {valid_rows:,} clean rows ({csv_size_mb:.2f} MB) in {total_time:.2f}s.")
    print(f"Skipped / Malformed Rows: {skipped_rows:,}")
    if valid_rows > 0:
        sig_pct = (100.0 * higgs_signal_count) / valid_rows
        print(f"Signal Events (y=1)      : {higgs_signal_count:,} ({sig_pct:.2f}%)")
        print(f"Background Events (y=0)  : {background_count:,} ({100.0 - sig_pct:.2f}%)")
    print("======================================================================")

def main():
    parser = argparse.ArgumentParser(description="Stream and Clean UCI HIGGS Dataset")
    parser.add_argument("-n", "--target-rows", type=int, default=600000,
                        help="Number of target rows to stream (default: 600,000; use 0 or 11000000 for full dataset)")
    parser.add_argument("--full-precision", action="store_true",
                        help="Preserve full floating-point precision (default: 5-decimal truncation)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force re-generation even if files already exist")
    parser.add_argument("--output-csv", default=None,
                        help="Custom output CSV path (default: data/higgs_clean.csv)")
    parser.add_argument("--output-labels", default=None,
                        help="Custom output labels path (default: data/higgs_labels.txt)")
    parser.add_argument("--archive-path", default=None,
                        help="Path to an existing HIGGS.csv.gz archive on disk (skips HTTP download)")
    args = parser.parse_args()

    prepare_higgs_dataset(
        target_rows=args.target_rows,
        full_precision=args.full_precision,
        force=args.force,
        custom_csv=args.output_csv,
        custom_labels=args.output_labels,
        archive_path=args.archive_path
    )

if __name__ == "__main__":
    main()
