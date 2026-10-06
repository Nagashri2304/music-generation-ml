"""MIDI <-> event conversion.

An *event* is one melody onset and is a triple (pitch_idx, dur_idx, chord):
  pitch_idx : 0..87   key of the highest note >= SPLIT_PITCH starting at that onset (right hand)
  dur_idx   : index into config.DURATIONS, the quantised inter-onset interval
  chord     : pitch-class set of the notes < SPLIT_PITCH sounding at that onset (left hand),
              a string like "0-4-7" (or "none") before encoding, an integer id afterwards.
This mirrors the paper's note-event vectors [right-hand note, duration, left-hand note(s), duration].
"""
import warnings
import numpy as np
import pretty_midi
from .config import *


def nearest_dur(steps):
    steps = max(1, min(DURATIONS[-1], steps))
    return int(np.argmin([abs(d - steps) for d in DURATIONS]))


def chord_key(pitches):
    pcs = sorted({int(p) % 12 for p in pitches})
    return "-".join(map(str, pcs)) if pcs else "none"


def midi_to_events(path, min_events=MIN_EVENTS):
    """Parse one MIDI file -> list of (pitch_idx, dur_idx, chord_key) or None if unusable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if len(notes) < min_events:
        return None
    tempi = pm.get_tempo_changes()[1]
    tempo = float(tempi[0]) if len(tempi) and 30 <= tempi[0] <= 300 else DEFAULT_TEMPO
    sps = 60.0 / tempo / STEPS_PER_BEAT                      # seconds per 16th step
    start = np.array([n.start for n in notes]); end = np.array([n.end for n in notes])
    pitch = np.array([n.pitch for n in notes])
    high = pitch >= SPLIT_PITCH
    if high.sum() < min_events:
        return None
    onset = np.round(start / sps).astype(int)
    mel = {}                                                  # onset step -> (highest pitch, its end time)
    for s, p, e in zip(onset[high], pitch[high], end[high]):
        if s not in mel or p > mel[s][0]:
            mel[s] = (p, e)
    steps = sorted(mel)
    lo_s, lo_e, lo_p = start[~high], end[~high], pitch[~high]
    events = []
    for i, s in enumerate(steps):
        p, e = mel[s]
        ioi = steps[i + 1] - s if i + 1 < len(steps) else int(round((e - s * sps) / sps))
        t = s * sps + 1e-3
        chord = chord_key(lo_p[(lo_s <= t) & (lo_e > t)])
        events.append((int(np.clip(p, PITCH_MIN, PITCH_MAX)) - PITCH_MIN, nearest_dur(ioi), chord))
    return events if len(events) >= min_events else None


def events_to_midi(events, chord_vocab, out_path, tempo=DEFAULT_TEMPO):
    """Encoded events [(pitch_idx, dur_idx, chord_id), ...] -> playable .mid (melody + chord track)."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    sps = 60.0 / tempo / STEPS_PER_BEAT
    mel = pretty_midi.Instrument(program=0, name="melody")
    harm = pretty_midi.Instrument(program=0, name="chords")
    t, cur, cur_t = 0.0, None, 0.0

    def flush(key, t0, t1):
        if key in (None, "<other>", "none") or t1 <= t0:
            return
        for pc in key.split("-"):
            harm.notes.append(pretty_midi.Note(velocity=55, pitch=48 + int(pc), start=t0, end=t1))

    for p, d, c in events:
        dur = DURATIONS[int(d)] * sps
        mel.notes.append(pretty_midi.Note(velocity=88, pitch=int(p) + PITCH_MIN, start=t, end=t + dur * 0.92))
        key = chord_vocab[int(c)]
        if key != cur:
            flush(cur, cur_t, t)
            cur, cur_t = key, t
        t += dur
    flush(cur, cur_t, t)
    pm.instruments.extend([mel, harm])
    pm.write(str(out_path))
    return len(mel.notes)


def check_midi(path, expected_melody_notes=None):
    """Re-open a written file with pretty_midi and sanity-check it. Returns True if playable."""
    pm = pretty_midi.PrettyMIDI(str(path))
    mel = [i for i in pm.instruments if i.name == "melody"]
    ok = bool(mel) and len(mel[0].notes) > 0 and pm.get_end_time() > 0
    if expected_melody_notes is not None and ok:
        ok = len(mel[0].notes) == expected_melody_notes
    return ok
