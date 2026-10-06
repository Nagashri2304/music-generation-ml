# Generating Music with Machine Learning (Mini-Project 22, UE24CS352A)

Team: Nagashri R Patil (PES1UG24CS289) and Nallamalli Kanaka Mani Sai Akhil (PES1UG24CS290)
https://github.com/Nagashri2304/music-generation-ml

A PyTorch re-implementation of the core of the Stanford project *"Music Composition with Machine Learning"*
(Kang, Kim, Ringdahl, 2018). It learns from a folder of MIDI files and generates new MIDI music with three models:

| Model | Role | Idea |
|---|---|---|
| **LSTM** | main model | 1 LSTM layer (256 units) predicts the next note event (pitch, duration, chord); Adam lr 0.005 as in the paper |
| **Naive-Bayes-like** | comparison | P(note \| chord), notes independent given the chord; Markov chain over chords for generation |
| **Random (flat)** | baseline | uniform over the 88 piano keys / durations / chords |

The paper's vanilla NN and encoder-decoder models are intentionally not included (see the report, Future work).

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # Python 3.9+; CPU is enough, CUDA is used automatically if present
```

## 2. Dataset (required for real experiments)
Put **real** `.mid` / `.midi` files (hundreds of them) into `data/` - see `data/README.md` for sources.
The code works on any MIDI folder (nothing is hard-coded). `preprocess.py` stops with a message if `data/` has fewer than 50 files.

## 3. Run everything (one command)
```bash
python run_all.py --dataset-name "piano-midi.de"      # use a name that describes YOUR dataset
```
This runs preprocessing -> training -> evaluation and writes everything to `outputs/`. Or run the steps separately:
```bash
python preprocess.py --data-dir data --out outputs --dataset-name "piano-midi.de"   # parse, split 80/10/10 by song, chord vocabulary
python train.py      --out outputs --epochs 40        # LSTM training/validation, best checkpoint -> outputs/lstm.pt
python evaluate.py   --out outputs                    # test metrics, model comparison, generated music, plots, MIDI files
```
CPU time on a laptop: roughly 10 s per epoch for ~100k events; early stopping usually ends training in 10-25 epochs.

## 4. Live demo for the review
```bash
python demo.py --play                                 # LSTM composes a new piece -> outputs/demo/demo_lstm.mid and plays it
python demo.py --model nb --seed 3                    # Naive-Bayes-like
python demo.py --model random                         # random baseline (for contrast)
python demo.py --count 3 --events 128 --temperature 0.9
```
`--play` uses `pygame` if installed (`pip install pygame`), otherwise opens the file with your OS default MIDI player.
Any MIDI player works (VLC, Windows Media Player, MuseScore). Pre-generated samples: `outputs/generated/{lstm,nb,random}_0X.mid`.

## 5. Outputs (`outputs/`)
| File | Content |
|---|---|
| `results/next_event_metrics.csv` | test pitch CE/perplexity, top-1/top-5, duration CE/accuracy for all models |
| `results/generation_metrics.csv` | objective metrics of generated music vs real test music |
| `results/loss_curve.png`, `next_event_comparison.png`, `generation_comparison.png`, `piano_rolls.png` | plots |
| `results/metrics.json`, `history.json`, `dataset_stats.json` | all numbers (used by the report/slide scripts) |
| `generated/*.mid`, `demo/*.mid` | playable generated MIDI (re-loaded and checked automatically) |

## 6. Report and slides
```bash
pip install reportlab python-pptx pypdf
python tools/make_report.py --out outputs --report report/Music_Generation_Report.pdf   # 2-page PDF (add --base 7.8 if it spills over)
python tools/make_slides.py --out outputs --pptx report/Music_Generation_Slides.pptx
```
Both read `outputs/results/metrics.json`, so re-running them after your own experiment updates every number.

## 7. How it works (for the viva)
* **Representation** (`musicgen/midi_io.py`): each melody onset becomes an event `[right-hand key (0-87), duration class (8), left-hand chord id (64)]`;
  right hand = highest note >= MIDI 60, left-hand chord = pitch-class set of lower sounding notes, durations quantised to a 16th-note grid.
* **Dataset** (`musicgen/data.py`): song-level 80/10/10 split (no leakage), chord vocabulary from training songs only, 64-event windows shifted by one for next-event targets.
* **Models** (`musicgen/models.py`): see the table above. LSTM loss = CE(pitch) + CE(duration) + CE(chord).
* **Evaluation** (`musicgen/metrics.py`): all models score the *same* held-out target events. The Naive-Bayes "given chord" row sees the true concurrent chord (paper protocol), so the "prev chord" row is the fair comparison.
  Generated music: scale consistency, chord-tone rate (harmony), repeated-note rate, interval statistics and Jensen-Shannon divergence to real intervals, 5-gram novelty vs training data (detects copying/overfitting).
* **Reconstruction**: `events_to_midi` writes a melody track and a chord track with `pretty_midi`; `check_midi` re-opens each file to verify it.

## 8. Smoke test (pipeline check only)
```bash
python run_all.py --smoke      # uses 30 synthetic files in data_sample/ - numbers are NOT meaningful and never touch data/ or outputs/
```

## Honest notes / differences from the paper
* The paper's Naive Bayes uses three consecutive left-hand notes as the chord; we use the pitch-class set sounding at the melody onset.
* The LSTM shares the paper's architecture/optimiser, but is trained on whichever dataset you provide (the paper: 24 Chopin etudes); results will differ from the paper's.
* No human "music Turing test" is included; objective proxies are used instead.
