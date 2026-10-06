import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .config import *

COL = {"Random (flat)": "#9aa0a6", "Naive Bayes-like": "#e8a33d", "LSTM": "#2a6fdb", "Real (test)": "#3a9d5d"}


def plot_loss(hist, path):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(hist["train"], label="train", color="#2a6fdb"); ax.plot(hist["val"], label="validation", color="#e8743d")
    ax.axvline(hist["best_epoch"] - 1, ls=":", color="k", lw=0.8, label=f"best (epoch {hist['best_epoch']})")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (pitch+duration+chord)"); ax.set_title("LSTM training / validation loss")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_next_event(rows, path):
    names = [r["model"] for r in rows]; x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for i, (k, lab) in enumerate([("pitch_top1", "pitch top-1"), ("pitch_top5", "pitch top-5"), ("dur_acc", "duration acc.")]):
        v = [r[k] for r in rows]; b = ax.bar(x + (i - 1) * w, v, w, label=lab)
        for rect, val in zip(b, v): ax.text(rect.get_x() + w / 2, val + 0.005, f"{val:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
    ax.set_ylabel("test accuracy"); ax.set_title("Next-event prediction on held-out songs"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_generation(gen, path):
    keys = [("scale_consistency", "scale consistency"), ("chord_tone_rate", "chord-tone rate"),
            ("novelty_5gram", "5-gram novelty"), ("repetition_rate", "repetition rate")]
    fig, axs = plt.subplots(1, 4, figsize=(8, 2.5))
    for ax, (k, lab) in zip(axs, keys):
        names = list(gen); m = [gen[n][k + "_mean"] for n in names]; s = [gen[n][k + "_std"] for n in names]
        ax.bar(range(len(names)), m, yerr=s, color=[COL.get(n, "#777") for n in names], capsize=2)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n.split(" ")[0] for n in names], fontsize=7, rotation=20)
        ax.set_title(lab, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_piano_rolls(samples, path, n=48):
    fig, axs = plt.subplots(len(samples), 1, figsize=(7, 1.7 * len(samples)), sharex=True)
    for ax, (name, a) in zip(np.atleast_1d(axs), samples.items()):
        t = 0
        for p, d, _ in a[:n]:
            ax.barh(p + PITCH_MIN, DURATIONS[d], left=t, height=0.9, color=COL.get(name, "#555")); t += DURATIONS[d]
        ax.set_ylabel(name, fontsize=7); ax.tick_params(labelsize=6)
    np.atleast_1d(axs)[-1].set_xlabel("time (16th-note steps)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
