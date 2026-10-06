"""Dataset creation: scan data/ -> parse -> song-level 80/10/10 split -> chord vocab -> windows."""
import json, pickle, random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from .config import *
from .midi_io import midi_to_events


def find_midi_files(data_dir):
    return sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in (".mid", ".midi"))


def preprocess(data_dir, out_dir, min_files=50, seed=42, dataset_name="custom"):
    out_dir = Path(out_dir); (out_dir / "results").mkdir(parents=True, exist_ok=True)
    files = find_midi_files(data_dir)
    if len(files) < min_files:
        raise SystemExit(
            f"\n[ERROR] Found only {len(files)} MIDI files in '{data_dir}' (need >= {min_files}).\n"
            "Put a real MIDI dataset (hundreds of .mid/.midi files) in data/ - see README section 'Dataset'.\n"
            "To only test that the code runs, use:  python run_all.py --smoke")
    songs, skipped = [], 0
    for f in tqdm(files, desc="Parsing MIDI"):
        try:
            ev = midi_to_events(f)
        except Exception:
            ev = None
        if ev is None:
            skipped += 1
        else:
            songs.append((f.name, ev))
    if len(songs) < 10:
        raise SystemExit("[ERROR] Fewer than 10 usable songs after parsing.")
    random.Random(seed).shuffle(songs)
    n = len(songs); n_tr, n_va = int(0.8 * n), int(0.1 * n)
    parts = {"train": songs[:n_tr], "val": songs[n_tr:n_tr + n_va], "test": songs[n_tr + n_va:]}
    counts = Counter(k for _, ev in parts["train"] for _, _, k in ev)      # vocab from TRAIN only
    chord_vocab = ["<other>"] + [k for k, _ in counts.most_common(MAX_CHORDS - 1)]
    k2i = {k: i for i, k in enumerate(chord_vocab)}
    splits = {s: [np.array([(p, d, k2i.get(k, 0)) for p, d, k in ev], dtype=np.int64) for _, ev in lst]
              for s, lst in parts.items()}
    names = {s: [nm for nm, _ in lst] for s, lst in parts.items()}
    covered = sum(counts[k] for k in chord_vocab[1:]) / max(1, sum(counts.values()))
    stats = {"dataset_name": dataset_name, "files_found": len(files), "files_skipped": skipped,
             "songs_used": n, "songs": {s: len(v) for s, v in splits.items()},
             "events": {s: int(sum(len(a) for a in v)) for s, v in splits.items()},
             "events_total": int(sum(len(a) for v in splits.values() for a in v)),
             "chord_vocab_size": len(chord_vocab), "chord_vocab_coverage": round(covered, 3),
             "seed": seed}
    starts = np.stack([a[0] for a in splits["train"]])
    with open(out_dir / "processed.pkl", "wb") as f:
        pickle.dump({"splits": splits, "names": names, "chord_vocab": chord_vocab,
                     "starts": starts, "stats": stats}, f)
    (out_dir / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def load_processed(out_dir):
    p = Path(out_dir) / "processed.pkl"
    if not p.exists():
        raise SystemExit(f"[ERROR] {p} not found. Run preprocess.py first.")
    with open(p, "rb") as f:
        return pickle.load(f)


def make_windows(songs, L=SEQ_LEN, stride=None):
    """Fixed-length (x, y) windows for next-event prediction; y = x shifted by one, pad target = -100."""
    stride = stride or L // 2
    X, Y = [], []
    for a in songs:
        n = len(a)
        if n < L + 1:                                     # short song -> one padded window
            x = np.zeros((L, 3), np.int64); y = np.full((L, 3), -100, np.int64)
            x[:n - 1] = a[:-1]; y[:n - 1] = a[1:]
            X.append(x); Y.append(y); continue
        starts = list(range(0, n - L, stride))
        if starts[-1] != n - L - 1:
            starts.append(n - L - 1)
        for s in starts:
            X.append(a[s:s + L]); Y.append(a[s + 1:s + L + 1])
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


def eval_songs(songs):
    """Same truncation for every model, so all models are scored on identical target events."""
    return [a[:MAX_EVAL_LEN + 1] for a in songs]


def flat_targets(songs):
    return np.concatenate([a[1:] for a in songs])
