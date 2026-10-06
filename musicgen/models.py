"""The three models: Random (flat) baseline, Naive-Bayes-like chord->note model, LSTM."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *


def _norm(p, temperature=1.0):
    p = np.asarray(p, dtype=np.float64) ** (1.0 / max(temperature, 1e-3))
    return p / p.sum()


# ----------------------------------------------------------------------------- Random baseline
class RandomBaseline:
    """Flat distribution over the 88 keys / durations / chords (the paper's 'flat' reference)."""
    name = "Random (flat)"

    def __init__(self, n_chords):
        self.n_chords = n_chords

    def predict(self, songs):
        N = sum(len(a) - 1 for a in songs)
        return {"pitch_logp": np.full((N, N_PITCH), -np.log(N_PITCH), np.float32),
                "dur_logp": np.full((N, N_DUR), -np.log(N_DUR), np.float32)}

    def sample(self, n, rng, temperature=1.0, **_):
        return np.stack([rng.integers(0, N_PITCH, n), rng.integers(0, N_DUR, n),
                         rng.integers(0, self.n_chords, n)], 1)


# ----------------------------------------------------------------------------- Naive-Bayes-like
class NaiveBayesLike:
    """P(note | chord) with Laplace smoothing; every note is treated as independent given the chord.

    mode='concurrent': chord of the *same* event is given (paper protocol; uses extra information)
    mode='prev'      : chord of the *previous* event is given (fair next-event prediction)
    Generation (concurrent model only): chord sequence ~ first-order Markov chain, then
    pitch ~ P(pitch|chord), duration ~ P(duration|chord).
    """
    def __init__(self, n_chords, mode="concurrent", alpha=0.1):
        self.C, self.mode, self.alpha = n_chords, mode, alpha
        self.name = "Naive Bayes-like" + (" (prev chord)" if mode == "prev" else " (given chord)")

    def fit(self, songs):
        C = self.C
        self.pc = np.zeros((C, N_PITCH)); self.dc = np.zeros((C, N_DUR))
        self.trans = np.zeros((C, C)); self.start = np.zeros(C)
        for a in songs:
            if self.mode == "prev":
                np.add.at(self.pc, (a[:-1, 2], a[1:, 0]), 1); np.add.at(self.dc, (a[:-1, 2], a[1:, 1]), 1)
            else:
                np.add.at(self.pc, (a[:, 2], a[:, 0]), 1); np.add.at(self.dc, (a[:, 2], a[:, 1]), 1)
            np.add.at(self.trans, (a[:-1, 2], a[1:, 2]), 1); self.start[a[0, 2]] += 1
        return self

    def _probs(self, counts, alpha=None):
        a = self.alpha if alpha is None else alpha
        return (counts + a) / (counts.sum(1, keepdims=True) + a * counts.shape[1])

    def predict(self, songs):
        ctx = np.concatenate([a[:-1, 2] if self.mode == "prev" else a[1:, 2] for a in songs])
        return {"pitch_logp": np.log(self._probs(self.pc)[ctx]).astype(np.float32),
                "dur_logp": np.log(self._probs(self.dc)[ctx]).astype(np.float32)}

    def tune_alpha(self, val_songs, grid=(0.01, 0.05, 0.1, 0.5, 1.0)):
        from .metrics import next_event_metrics
        best = min(grid, key=lambda a: self._score(a, val_songs))
        self.alpha = best
        return best

    def _score(self, a, val):
        old, self.alpha = self.alpha, a
        T = np.concatenate([x[1:] for x in val]); pr = self.predict(val)
        s = -pr["pitch_logp"][np.arange(len(T)), T[:, 0]].mean() - pr["dur_logp"][np.arange(len(T)), T[:, 1]].mean()
        self.alpha = old
        return s

    def sample(self, n, rng, temperature=1.0, **_):
        assert self.mode == "concurrent", "generation needs the concurrent model"
        P, D, T = self._probs(self.pc), self._probs(self.dc), self._probs(self.trans)
        c = rng.choice(self.C, p=_norm(self.start + 1e-3))
        out = []
        for _i in range(n):
            out.append((rng.choice(N_PITCH, p=_norm(P[c], temperature)),
                        rng.choice(N_DUR, p=_norm(D[c], temperature)), c))
            c = rng.choice(self.C, p=_norm(T[c], temperature))
        return np.array(out)


# ----------------------------------------------------------------------------- LSTM
class MusicLSTM(nn.Module):
    """Single LSTM layer (256 units, as in the paper). Input = embeddings of the previous event's
    (pitch, duration, chord); three softmax heads predict the next event's (pitch, duration, chord)."""
    def __init__(self, n_chords, emb_p=32, emb_d=8, emb_c=16, hidden=256, layers=1, dropout=0.3):
        super().__init__()
        self.cfg = dict(n_chords=n_chords, emb_p=emb_p, emb_d=emb_d, emb_c=emb_c,
                        hidden=hidden, layers=layers, dropout=dropout)
        self.ep, self.ed, self.ec = nn.Embedding(N_PITCH, emb_p), nn.Embedding(N_DUR, emb_d), nn.Embedding(n_chords, emb_c)
        self.lstm = nn.LSTM(emb_p + emb_d + emb_c, hidden, layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.hp, self.hd, self.hc = nn.Linear(hidden, N_PITCH), nn.Linear(hidden, N_DUR), nn.Linear(hidden, n_chords)

    def forward(self, x, state=None):
        e = torch.cat([self.ep(x[..., 0]), self.ed(x[..., 1]), self.ec(x[..., 2])], -1)
        h, state = self.lstm(e, state)
        h = self.drop(h)
        return self.hp(h), self.hd(h), self.hc(h), state

    def loss(self, x, y):
        lp, ld, lc, _ = self(x)
        Lp = F.cross_entropy(lp.reshape(-1, N_PITCH), y[..., 0].reshape(-1), ignore_index=-100)
        Ld = F.cross_entropy(ld.reshape(-1, N_DUR), y[..., 1].reshape(-1), ignore_index=-100)
        Lc = F.cross_entropy(lc.reshape(-1, lc.shape[-1]), y[..., 2].reshape(-1), ignore_index=-100)
        return Lp + Ld + Lc, Lp.item(), Ld.item()

    @torch.no_grad()
    def sample(self, n, rng, temperature=1.0, start=None, device="cpu", **_):
        self.eval()
        ev = np.array(start if start is not None else [rng.integers(0, N_PITCH), 3, 0])
        out, state = [ev.copy()], None
        for _i in range(n - 1):
            x = torch.tensor(ev, dtype=torch.long, device=device).view(1, 1, 3)
            lp, ld, lc, state = self(x, state)
            ev = np.array([rng.choice(lg.shape[-1], p=_norm(torch.softmax(lg[0, 0], -1).cpu().numpy(), temperature))
                           for lg in (lp, ld, lc)])
            out.append(ev.copy())
        return np.stack(out)

    @torch.no_grad()
    def predict(self, songs, device="cpu", batch=32):
        self.eval()
        order = np.argsort([len(a) for a in songs]); res = {}
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]; L = max(len(songs[j]) for j in idx) - 1
            x = torch.zeros(len(idx), L, 3, dtype=torch.long)
            for r, j in enumerate(idx):
                x[r, :len(songs[j]) - 1] = torch.from_numpy(songs[j][:-1])
            lp, ld, _, _ = self(x.to(device))
            lp, ld = F.log_softmax(lp, -1).cpu().numpy(), F.log_softmax(ld, -1).cpu().numpy()
            for r, j in enumerate(idx):
                n = len(songs[j]) - 1; res[j] = (lp[r, :n], ld[r, :n])
        return {"pitch_logp": np.concatenate([res[j][0] for j in range(len(songs))]),
                "dur_logp": np.concatenate([res[j][1] for j in range(len(songs))])}
