"""Review demo: generate (and optionally play) a new MIDI composition.

  python demo.py                       # LSTM, 64 events -> outputs/demo/demo_lstm.mid
  python demo.py --play                # ...and open it in a MIDI player
  python demo.py --model nb --seed 3   # Naive-Bayes-like, different seed
  python demo.py --count 3 --events 128 --temperature 0.9
"""
import argparse, os, platform, subprocess, sys
from pathlib import Path
import numpy as np
from musicgen.config import DURATIONS, STEPS_PER_BEAT
from musicgen.generation import build_model, sample_events
from musicgen.midi_io import events_to_midi, check_midi
from train import get_device


def play(path):
    try:                                    # 1) pygame (pip install pygame) plays inside the terminal
        import pygame
        pygame.mixer.init(); pygame.mixer.music.load(str(path)); pygame.mixer.music.play()
        print("Playing... (Ctrl+C to stop)")
        while pygame.mixer.music.get_busy():
            pygame.time.wait(200)
        return
    except Exception:
        pass
    try:                                    # 2) fall back to the OS default MIDI player
        if platform.system() == "Windows": os.startfile(str(path))
        elif platform.system() == "Darwin": subprocess.run(["open", str(path)])
        else: subprocess.run(["xdg-open", str(path)])
    except Exception:
        print("Could not auto-play; open the .mid file with any MIDI player (VLC, Windows Media Player, MuseScore).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "nb", "random"], default="lstm")
    ap.add_argument("--out", default="outputs", help="folder produced by run_all.py")
    ap.add_argument("--events", type=int, default=64); ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.9); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tempo", type=int, default=120); ap.add_argument("--play", action="store_true")
    a = ap.parse_args()
    device = get_device()
    model, vocab, starts = build_model(a.model, a.out, device)
    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 10**6)
    rng = np.random.default_rng(seed); d = Path(a.out) / "demo"; d.mkdir(parents=True, exist_ok=True)
    for i in range(a.count):
        ev = sample_events(model, a.events, rng, starts, a.temperature, device)
        p = d / (f"demo_{a.model}.mid" if a.count == 1 else f"demo_{a.model}_{i + 1}.mid")
        n = events_to_midi(ev, vocab, p, a.tempo)
        secs = sum(DURATIONS[x] for x in ev[:, 1]) * 60 / a.tempo / STEPS_PER_BEAT
        print(f"[{a.model}] seed={seed} -> {p}  ({n} melody notes, ~{secs:.0f}s, playable={check_midi(p, n)})")
        if a.play:
            play(p)
