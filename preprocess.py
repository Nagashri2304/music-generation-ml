"""Step 1 - parse every .mid/.midi in data/, split by song (80/10/10), build chord vocabulary."""
import argparse
from musicgen.data import preprocess

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="outputs")
    ap.add_argument("--min-files", type=int, default=50, help="refuse to run on fewer files than this")
    ap.add_argument("--dataset-name", default="custom"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    preprocess(a.data_dir, a.out, a.min_files, a.seed, a.dataset_name)
