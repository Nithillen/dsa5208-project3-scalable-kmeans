import urllib.request
import zlib
import os
import math
import time

def prepare_higgs_dataset(target_rows=600000):
    """
    Streams and cleans the UCI HIGGS Boson High-Energy Physics Dataset
    using fast chunked zlib decompression over HTTP.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_csv = os.path.join(data_dir, "higgs_clean.csv")
    output_labels = os.path.join(data_dir, "higgs_labels.txt")

    if os.path.exists(output_csv) and os.path.exists(output_labels):
        print(f"HIGGS dataset already prepared at:\n  {output_csv}")
        return

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    print("======================================================================")
    print("  Streaming & Cleaning UCI HIGGS Boson Dataset (28 Physics Features)  ")
    print("======================================================================")
    print(f"Source URL   : {url}")
    print(f"Target Size  : {target_rows:,} collision events x 28 features")

    t0 = time.time()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)

    valid_rows = 0
    skipped_rows = 0
    higgs_signal_count = 0
    background_count = 0
    remainder = b""

    with urllib.request.urlopen(req, timeout=180) as response, \
         open(output_csv, 'w', buffering=16*1024*1024) as feat_file, \
         open(output_labels, 'w', buffering=4*1024*1024) as label_file:

        while valid_rows < target_rows:
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
                    if valid_rows >= target_rows:
                        break
                except ValueError:
                    skipped_rows += 1
                    continue

    total_time = time.time() - t0
    csv_size_mb = os.path.getsize(output_csv) / (1024.0 * 1024.0)
    print(f"Saved {valid_rows:,} clean rows ({csv_size_mb:.2f} MB) in {total_time:.2f}s.")

if __name__ == "__main__":
    prepare_higgs_dataset()
