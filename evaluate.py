"""Step 3 - test-set evaluation, LSTM vs Naive-Bayes-like vs Random, generation + plots + MIDI files."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from musicgen.config import *
from musicgen.data import load_processed, eval_songs, flat_targets
from musicgen.generation import load_lstm, sample_events
from musicgen.metrics import (next_event_metrics, sequence_metrics, ngram_set, interval_js)
from musicgen.midi_io import events_to_midi, check_midi
from musicgen.models import NaiveBayesLike, RandomBaseline
from musicgen.plotting import plot_next_event, plot_generation, plot_piano_rolls
from train import get_device


def evaluate(out="outputs", n_gen=50, gen_len=128, n_midi=5, temperature=1.0, seed=42, dataset_name=None):
    out = Path(out); res = out / "results"; (out / "generated").mkdir(exist_ok=True)
    D = load_processed(out); S = D["splits"]; C = len(D["chord_vocab"]); device = get_device()
    test, val = eval_songs(S["test"]), eval_songs(S["val"]); T = flat_targets(test)
    lstm, _, starts = load_lstm(out, device)
    nb_c = NaiveBayesLike(C, "concurrent").fit(S["train"]); nb_p = NaiveBayesLike(C, "prev").fit(S["train"])
    a1, a2 = nb_c.tune_alpha(val), nb_p.tune_alpha(val); print(f"NB smoothing alpha: concurrent={a1}, prev={a2}")
    rnd = RandomBaseline(C)

    # ---- (1) next-event prediction on the held-out test songs
    rows = []
    for name, m, pred in [("Random (flat)", rnd, rnd.predict(test)),
                          ("Naive Bayes-like (prev chord)", nb_p, nb_p.predict(test)),
                          ("Naive Bayes-like (given chord)", nb_c, nb_c.predict(test)),
                          ("LSTM", lstm, lstm.predict(test, device))]:
        rows.append({"model": name, **next_event_metrics(pred, T)})
    df = pd.DataFrame(rows); df.to_csv(res / "next_event_metrics.csv", index=False)
    print(df.round(3).to_string(index=False))

    # ---- (2) generation: n_gen sequences of gen_len events per model, objective metrics
    rng = np.random.default_rng(seed); gens = {}
    gens["Random (flat)"] = [rnd.sample(gen_len, rng) for _ in range(n_gen)]
    gens["Naive Bayes-like"] = [nb_c.sample(gen_len, rng, temperature) for _ in range(n_gen)]
    gens["LSTM"] = [sample_events(lstm, gen_len, rng, starts, temperature, device) for _ in range(n_gen)]
    real = [a[:gen_len] for a in S["test"]]
    tri = ngram_set(S["train"]); gm, summary = {}, {}
    for name, seqs in [("Real (test)", real)] + list(gens.items()):
        M = sequence_metrics(seqs, D["chord_vocab"], tri)
        gm[name] = {f"{k}_{s}": float(f(v[~np.isnan(v)])) for k, v in M.items() for s, f in (("mean", np.mean), ("std", np.std))}
        gm[name]["interval_js_vs_real"] = 0.0 if name == "Real (test)" else float(interval_js(seqs, real))
    gdf = pd.DataFrame(gm).T; gdf.to_csv(res / "generation_metrics.csv"); print(gdf.round(3).T.to_string())

    # ---- (3) write playable MIDI files and verify they re-load
    ok = {}
    for name, seqs in gens.items():
        tag = {"Random (flat)": "random", "Naive Bayes-like": "nb", "LSTM": "lstm"}[name]; good = 0
        for i, s in enumerate(seqs[:n_midi], 1):
            p = out / "generated" / f"{tag}_{i:02d}.mid"
            n = events_to_midi(s, D["chord_vocab"], p); good += int(check_midi(p, n))
        ok[name] = f"{good}/{n_midi}"
    print("MIDI files verified playable (reload check):", ok)

    plot_next_event(rows, res / "next_event_comparison.png"); plot_generation(gm, res / "generation_comparison.png")
    plot_piano_rolls({k: v[0] for k, v in gens.items()}, res / "piano_rolls.png")
    hist = json.loads((res / "history.json").read_text())
    (res / "metrics.json").write_text(json.dumps({
        "dataset": D["stats"], "next_event": rows, "generation": gm, "midi_check": ok,
        "training": hist, "n_generated": n_gen, "gen_len": gen_len, "temperature": temperature,
        "nb_alpha": {"concurrent": a1, "prev": a2}}, indent=2))
    return rows, gm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs"); ap.add_argument("--n-gen", type=int, default=50)
    ap.add_argument("--gen-len", type=int, default=128); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    evaluate(a.out, a.n_gen, a.gen_len, 5, a.temperature, a.seed)
