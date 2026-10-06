"""One command for the whole pipeline: preprocess -> train -> evaluate (-> plots, MIDI files).

  python run_all.py --dataset-name "piano-midi.de"      # real experiment, reads data/
  python run_all.py --smoke                             # 1-minute pipeline test on synthetic files ONLY
"""
import argparse, subprocess, sys
from pathlib import Path
from musicgen.data import preprocess
from train import train
from evaluate import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="outputs")
    ap.add_argument("--dataset-name", default="custom"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-files", type=int, default=50); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="pipeline test on synthetic data (results are NOT meaningful)")
    a = ap.parse_args()
    if a.smoke:
        a.data_dir, a.out, a.epochs, a.min_files, a.dataset_name = "data_sample", "outputs_smoke", 3, 10, "SMOKE-TEST (synthetic)"
        if not list(Path(a.data_dir).glob("*.mid")):
            subprocess.run([sys.executable, "make_sample_midi.py"], check=True)
    preprocess(a.data_dir, a.out, a.min_files, a.seed, a.dataset_name)
    train(a.out, a.epochs, seed=a.seed)
    evaluate(a.out, seed=a.seed)
    print(f"\nDone. See {a.out}/results/ (metrics, plots) and {a.out}/generated/ (MIDI).")
