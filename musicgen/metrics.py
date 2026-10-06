"""Evaluation: (1) next-event prediction on the test set, (2) objective metrics of generated music."""
import numpy as np
from .config import *

MAJOR = np.array([0, 2, 4, 5, 7, 9, 11]); MINOR = np.array([0, 2, 3, 5, 7, 8, 10])


def next_event_metrics(pred, T, seed=0):
    """pred: dict of log-probs for every target event, T: [N,3] true next events."""
    rng = np.random.default_rng(seed); N = len(T); ar = np.arange(N)
    pl, dl = pred["pitch_logp"], pred["dur_logp"]
    ce_p, ce_d = -pl[ar, T[:, 0]].mean(), -dl[ar, T[:, 1]].mean()
    noisy = pl + rng.random(pl.shape, dtype=np.float32) * 1e-6          # random tie-breaking (flat model)
    top5 = np.argpartition(-noisy, 5, axis=1)[:, :5]
    return {"pitch_ce": float(ce_p), "pitch_ppl": float(np.exp(ce_p)),
            "pitch_top1": float((noisy.argmax(1) == T[:, 0]).mean()),
            "pitch_top5": float((top5 == T[:, [0]]).any(1).mean()),
            "dur_ce": float(ce_d),
            "dur_acc": float(((dl + rng.random(dl.shape, dtype=np.float32) * 1e-6).argmax(1) == T[:, 1]).mean()),
            "n_events": int(N)}


def chord_sets(chord_vocab):
    return [set() if k in ("<other>", "none") else {int(x) for x in k.split("-")} for k in chord_vocab]


def _scale_fit(pcs):
    h = np.bincount(pcs, minlength=12)
    best = max(h[(sc + r) % 12].sum() for r in range(12) for sc in (MAJOR, MINOR))
    return best / max(1, h.sum())


def _entropy(pcs):
    h = np.bincount(pcs, minlength=12) / len(pcs); h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def _js(p, q):
    p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ngram_set(songs, n=5):
    s = set()
    for a in songs:
        p = a[:, 0].tolist()
        s.update(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
    return s


def sequence_metrics(seqs, chord_vocab, train_ngrams, n=5):
    """Per-sequence metrics -> dict name -> array (one value per sequence)."""
    cs = chord_sets(chord_vocab); M = {k: [] for k in
        ["scale_consistency", "chord_tone_rate", "repetition_rate", "mean_abs_interval",
         "large_leap_rate", "pc_entropy", "novelty_5gram"]}
    for a in seqs:
        p = a[:, 0] + PITCH_MIN; iv = np.diff(p)
        M["scale_consistency"].append(_scale_fit(p % 12))
        valid = [(pp % 12) in cs[c] for pp, c in zip(p, a[:, 2]) if cs[c]]
        M["chord_tone_rate"].append(np.mean(valid) if valid else np.nan)
        M["repetition_rate"].append(np.mean(iv == 0))
        M["mean_abs_interval"].append(np.mean(np.abs(iv)))
        M["large_leap_rate"].append(np.mean(np.abs(iv) > 12))
        M["pc_entropy"].append(_entropy(p % 12))
        g = [tuple(a[i:i + n, 0].tolist()) for i in range(len(a) - n + 1)]
        M["novelty_5gram"].append(np.mean([x not in train_ngrams for x in g]) if g else np.nan)
    return {k: np.array(v, dtype=float) for k, v in M.items()}


def interval_hist(seqs, clip=24):
    h = np.zeros(2 * clip + 1)
    for a in seqs:
        np.add.at(h, np.clip(np.diff(a[:, 0]), -clip, clip) + clip, 1)
    return h + 1e-9


def interval_js(seqs, ref_seqs):
    return _js(interval_hist(seqs), interval_hist(ref_seqs))
