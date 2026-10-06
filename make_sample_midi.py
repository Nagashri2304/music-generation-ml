"""Creates a TINY synthetic MIDI set in data_sample/ - ONLY to smoke-test that the pipeline runs.
It is NOT a dataset: real experiments must use real MIDI files placed in data/."""
import argparse, random
from pathlib import Path
import pretty_midi

PROG = [[0, 5, 3, 4], [0, 3, 4, 0], [0, 4, 5, 3], [0, 0, 3, 4]]          # scale degree roots
MAJ = [0, 2, 4, 5, 7, 9, 11]


def make(path, rng):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    mel, bass = pretty_midi.Instrument(0), pretty_midi.Instrument(0)
    key = rng.choice([48, 50, 53, 55]); prog = rng.choice(PROG) * 2; t = 0.0; deg = 7
    for ch in prog:
        root = key + MAJ[ch % 7] - 12
        for k in (0, 4, 7):
            bass.notes.append(pretty_midi.Note(60, root + k if root + k < 60 else root + k - 12, t, t + 2.0))
        end = t + 2.0
        while t < end - 1e-6:
            d = rng.choice([0.25, 0.5, 0.5, 1.0]); d = min(d, end - t)
            deg = max(0, min(13, deg + rng.choice([-2, -1, -1, 0, 1, 1, 2])))
            p = key + 12 + MAJ[deg % 7] + 12 * (deg // 7)
            mel.notes.append(pretty_midi.Note(90, max(60, p), t, t + d)); t += d
    pm.instruments += [mel, bass]; pm.write(str(path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dir", default="data_sample"); a = ap.parse_args()
    Path(a.dir).mkdir(exist_ok=True); rng = random.Random(0)
    for i in range(a.n):
        make(Path(a.dir) / f"synthetic_{i:03d}.mid", rng)
    print(f"wrote {a.n} synthetic files to {a.dir}/ (smoke test only)")
