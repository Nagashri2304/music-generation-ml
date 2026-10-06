"""Load trained models and sample new pieces."""
import numpy as np
import torch
from .data import load_processed
from .models import MusicLSTM, NaiveBayesLike, RandomBaseline


def load_lstm(out="outputs", device="cpu"):
    ck = torch.load(f"{out}/lstm.pt", map_location=device, weights_only=False)
    m = MusicLSTM(**ck["cfg"]).to(device); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["chord_vocab"], ck["starts"]


def build_model(name, out="outputs", device="cpu"):
    """Return (model, chord_vocab, starts) for 'lstm' | 'nb' | 'random'."""
    if name == "lstm":
        return load_lstm(out, device)
    D = load_processed(out); C = len(D["chord_vocab"])
    if name == "random":
        return RandomBaseline(C), D["chord_vocab"], D["starts"]
    nb = NaiveBayesLike(C, "concurrent").fit(D["splits"]["train"])
    from .data import eval_songs
    nb.tune_alpha(eval_songs(D["splits"]["val"]))
    return nb, D["chord_vocab"], D["starts"]


def sample_events(model, n, rng, starts, temperature=1.0, device="cpu"):
    start = starts[rng.integers(len(starts))]            # first event of a random training song
    return model.sample(n, rng, temperature=temperature, start=start, device=device)
