import argparse
import gzip
import math
import os
import time
import urllib.request
import zlib


def prepare_higgs_dataset(target_rows=600000, input_gz=None, force=False, preserve_precision=True):
    """
    Streams and cleans the UCI HIGGS Boson High-Energy Physics Dataset (11,000,000 rows x 28 features).
    - Supports configurable row limits (--rows N, or --rows 0 / --all for all 11,000,000 rows).
    - Preserves full original floating-point precision from the UCI dataset without truncation.
    - Supports reading from a local HIGGS.csv.gz file (e.g., on Vanda scratch) or streaming via HTTP.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_csv = os.path.join(data_dir, "higgs_clean.csv")
    output_labels = os.path.join(data_dir, "higgs_labels.txt")

    if os.path.exists(output_csv) and os.path.exists(output_labels) and not force:
        with open(output_labels, "r") as lf:
            existing_rows = sum(1 for _ in lf)
        if target_rows > 0 and existing_rows >= target_rows:
            print(f"HIGGS dataset already prepared at:\n  {output_csv} ({existing_rows:,} rows)")
            print("  (Use --force to regenerate with different row count or precision settings.)")
            return

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    target_desc = "ALL (~11,000,000)" if target_rows == 0 else f"{target_rows:,}"

    print("======================================================================")
    print("  Streaming & Cleaning UCI HIGGS Boson Dataset (28 Physics Features)  ")
    print("======================================================================")
    print(f"Source       : {input_gz if input_gz else url}")
    print(f"Target Size  : {target_desc} collision events x 28 features")
    print(f"Precision    : {'Full original IEEE-754 precision preserved' if preserve_precision else 'Rounded (5 decimals)'}")

    t0 = time.time()
    valid_rows = 0
    skipped_rows = 0
    higgs_signal_count = 0
    background_count = 0

    def process_line(line_str, feat_file, label_file):
        nonlocal valid_rows, skipped_rows, higgs_signal_count, background_count
        tokens = line_str.split(",")
        if len(tokens) != 29:
            skipped_rows += 1
            return False

        try:
            label_val = float(tokens[0])
            label_int = 1 if label_val >= 0.5 else 0

            features = []
            for tok in tokens[1:29]:
                tok_clean = tok.strip()
                val = float(tok_clean)
                if math.isnan(val) or math.isinf(val):
                    skipped_rows += 1
                    return False
                if preserve_precision:
                    # Preserve full numeric precision while normalizing exponent notation if needed
                    features.append(f"{val:.17g}")
                else:
                    features.append(f"{val:.5f}")

            feat_file.write(",".join(features) + "\n")
            label_file.write(f"{label_int}\n")

            if label_int == 1:
                higgs_signal_count += 1
            else:
                background_count += 1

            valid_rows += 1
            return True
        except ValueError:
            skipped_rows += 1
            return False

    if input_gz and os.path.exists(input_gz):
        with gzip.open(input_gz, "rt", encoding="ascii", errors="ignore") as gz_file, \
             open(output_csv, "w", buffering=16 * 1024 * 1024) as feat_file, \
             open(output_labels, "w", buffering=4 * 1024 * 1024) as label_file:
            for line in gz_file:
                line = line.strip()
                if not line:
                    continue
                process_line(line, feat_file, label_file)
                if target_rows > 0 and valid_rows >= target_rows:
                    break
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        remainder = b""

        with urllib.request.urlopen(req, timeout=300) as response, \
             open(output_csv, "w", buffering=16 * 1024 * 1024) as feat_file, \
             open(output_labels, "w", buffering=4 * 1024 * 1024) as label_file:

            while target_rows == 0 or valid_rows < target_rows:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                decompressed = remainder + decompressor.decompress(chunk)
                lines = decompressed.split(b"\n")
                remainder = lines.pop()

                for raw_line in lines:
                    if not raw_line:
                        continue
                    try:
                        line = raw_line.decode("ascii").strip()
                    except UnicodeDecodeError:
                        skipped_rows += 1
                        continue

                    process_line(line, feat_file, label_file)
                    if target_rows > 0 and valid_rows >= target_rows:
                        break

    total_time = time.time() - t0
    csv_size_mb = os.path.getsize(output_csv) / (1024.0 * 1024.0)
    print(f"Saved {valid_rows:,} clean rows ({csv_size_mb:.2f} MB) in {total_time:.2f}s.")
    print(f"Signal (1): {higgs_signal_count:,} | Background (0): {background_count:,} | Skipped: {skipped_rows:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare UCI HIGGS Boson Dataset for Scalable K-Means")
    parser.add_argument(
        "--rows",
        type=int,
        default=600000,
        help="Number of rows to extract (set --rows 0 or --all for all 11,000,000 rows; default: 600000)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract the entire 11,000,000-row HIGGS dataset",
    )
    parser.add_argument(
        "--input-gz",
        type=str,
        default=None,
        help="Optional path to local HIGGS.csv.gz archive",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if data/higgs_clean.csv already exists",
    )
    args = parser.parse_args()
    rows = 0 if args.all else args.rows
    prepare_higgs_dataset(target_rows=rows, input_gz=args.input_gz, force=args.force, preserve_precision=True)
