<div align="left">

### 🎵 Learn the patterns. Predict the next event. Create new music.


Built with Python + PyTorch + MIDI

</div>

| Member | SRN |
|---|---|
| **Nagashri R Patil** | PES1UG24CS289 |
| **Nallamalli Kanaka Mani Sai Akhil** | PES1UG24CS290 |

**Course:** UE24CS352A — Machine Learning Mini-Project  
**Institution:** PES University  
**Repository:** <https://github.com/Nagashri2304/music-generation-ml>

## 1. Problem Statement

The objective of this project is to build a machine-learning system that learns musical structure from a folder of MIDI files and generates new MIDI music. Each melody onset is represented as an event containing a right-hand key, a duration class and a left-hand chord, and the models learn to predict the next musical event.

This project is a PyTorch re-implementation of the core of the Stanford project *"Music Composition with Machine Learning"* (Kang, Kim, Ringdahl, 2018). The paper's vanilla NN and encoder-decoder models are intentionally not included (see Future Work).

## 2. Project Overview

The project follows an end-to-end pipeline:

```text
MIDI Files (data/)
        ↓
Event Extraction (right hand, duration, left-hand chord)
        ↓
Song-level 80/10/10 Split
        ↓
Chord Vocabulary (training songs only)
        ↓
64-event Windows
        ↓
Model Training (LSTM / Naive-Bayes-like / Random)
        ↓
Next-Event Prediction
        ↓
MIDI Generation
        ↓
Objective Evaluation
        ↓
Report and Slides
```

The implementation uses `pretty_midi` for MIDI reading and writing and PyTorch for the LSTM model.

## 3. Models

| Model | Role | Idea |
|---|---|---|
| **LSTM** | Main model | 1 LSTM layer (256 units) predicts the next note event (pitch, duration, chord); Adam, lr 0.005 as in the paper |
| **Naive-Bayes-like** | Comparison | P(note \| chord), notes independent given the chord; Markov chain over chords for generation |
| **Random (flat)** | Baseline | Uniform over the 88 piano keys, durations and chords |

### LSTM architecture

```text
Input Window (64 events)
          ↓
[right-hand key | duration class | left-hand chord ID]
          ↓
LSTM (256 units)
          ↓
Output heads: Pitch · Duration · Chord
          ↓
Next Musical Event
```

Loss = CE(pitch) + CE(duration) + CE(chord)

### Training configuration

| Parameter | Value |
|---|---|
| LSTM layers | 1 |
| LSTM units | 256 |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Window length | 64 events |
| Data split | 80 / 10 / 10 by song |
| Maximum epochs | 40 (`--epochs 40`) |
| Early stopping | Yes (usually ends in 10–25 epochs) |
| Best checkpoint | `outputs/lstm.pt` |

## 4. Dataset

The code works on **any** folder of real `.mid` / `.midi` files — nothing is hard-coded. Place hundreds of MIDI files in `data/` (see `data/README.md` for sources).

```text
data/
├── README.md
├── song_001.mid
├── song_002.mid
└── ...
```

`preprocess.py` stops with a message if `data/` contains fewer than 50 files.

The MIDI dataset is **not** stored in the repository. Use a dataset name that describes your data when running the pipeline:

```cmd
python run_all.py --dataset-name "piano-midi.de"
```

## 5. Musical Representation and Preprocessing

Implemented in `musicgen/midi_io.py` and `musicgen/data.py`.

Each melody onset becomes an event:

```text
[right-hand key (0–87), duration class (8), left-hand chord ID (64)]
```

- **Right hand:** highest sounding note at or above MIDI note 60
- **Left-hand chord:** pitch-class set of the lower sounding notes
- **Durations:** quantised to a 16th-note grid

The preprocessing script:

1. Loads all MIDI files from `data/`.
2. Extracts melody and chord events.
3. Splits songs 80/10/10 at **song level** (no leakage between splits).
4. Builds the chord vocabulary from training songs only.
5. Creates 64-event windows shifted by one event for next-event targets.
6. Saves everything to `outputs/processed.pkl`.

## 6. Training and Validation

The LSTM is trained on the training songs and validated on the held-out validation songs after every epoch. The best checkpoint (lowest validation loss) is saved.

After training, the script saves:

```text
outputs/
├── lstm.pt
└── results/
    ├── history.json
    └── loss_curve.png
```

## 7. Music Generation

Generation is implemented in `musicgen/generation.py`. The trained model predicts one musical event at a time; a **temperature** value controls randomness (lower = more deterministic, higher = more varied).

```cmd
python demo.py --count 3 --events 128 --temperature 0.9
```

Generated events are written to MIDI with `events_to_midi` (a melody track and a chord track), and each file is re-opened automatically to verify it.

Pre-generated samples are available in:

```text
outputs/generated_music/{lstm,nb,random}_0X.mid
```

## 8. Evaluation

All models are scored on the **same held-out target events**.

**Next-event prediction** (`results/next_event_metrics.csv`)

- Pitch cross-entropy and perplexity
- Top-1 and top-5 accuracy
- Duration cross-entropy and accuracy

> The Naive-Bayes "given chord" row sees the true concurrent chord (paper protocol), so the "prev chord" row is the fair comparison.

**Generated music** (`results/generation_metrics.csv`)

- Scale consistency
- Chord-tone rate (harmony)
- Repeated-note rate
- Interval statistics and Jensen–Shannon divergence to real intervals
- 5-gram novelty vs training data (detects copying / overfitting)

These are objective proxies and **do not directly measure musical quality**.

## 9. Repository Structure

```text
music-generation-ml/
│
├── data/
│   └── README.md
│
├── musicgen/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── generation.py
│   ├── metrics.py
│   ├── midi_io.py
│   ├── models.py
│   └── plotting.py
│
├── outputs/
│   ├── demo/
│   │   ├── demo_lstm.mid
│   │   ├── demo_nb.mid
│   │   └── demo_random.mid
│   ├── generated_music/          # lstm_01..05, nb_01..05, random_01..05 (.mid)
│   ├── results/
│   │   ├── dataset_stats.json
│   │   ├── generation_comparison.png
│   │   ├── generation_metrics.csv
│   │   ├── history.json
│   │   ├── loss_curve.png
│   │   ├── metrics.json
│   │   ├── next_event_comparison.png
│   │   ├── next_event_metrics.csv
│   │   └── piano_rolls.png
│   ├── lstm.pt
│   └── processed.pkl
│
├── report/
│   ├── Music_Generation_Report.pdf
│   └── Music_Generation_Slides.pptx
│
├── tools/
│   ├── make_report.py
│   └── make_slides.py
│
├── demo.py
├── evaluate.py
├── make_sample_midi.py
├── preprocess.py
├── run_all.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 10. Installation

### Requirements

- Python **3.9+**
- Git
- A folder of real MIDI files (hundreds recommended)
- CPU is enough; CUDA is used automatically if present

Create an environment and install dependencies:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux / macOS use `source .venv/bin/activate`.

## 11. Running the Project

### Option A — Run everything (one command)

```cmd
python run_all.py --dataset-name "piano-midi.de"
```

This runs preprocessing → training → evaluation and writes everything to `outputs/`.

### Option B — Run step by step

**Step 1 — Preprocess**

```cmd
python preprocess.py --data-dir data --out outputs --dataset-name "piano-midi.de"
```

**Step 2 — Train**

```cmd
python train.py --out outputs --epochs 40
```

**Step 3 — Evaluate and generate**

```cmd
python evaluate.py --out outputs
```

CPU time on a laptop is roughly 10 s per epoch for ~100k events.

## 12. Live Demo

```cmd
python demo.py --play
python demo.py --model nb --seed 3
python demo.py --model random
python demo.py --count 3 --events 128 --temperature 0.9
```

- `--play` uses `pygame` if installed (`pip install pygame`), otherwise opens the file with your OS default MIDI player.
- Any MIDI player works (VLC, Windows Media Player, MuseScore).
- Output files: `outputs/demo/demo_lstm.mid`, `demo_nb.mid`, `demo_random.mid`.

## 13. Results and Output Files

All numbers are produced by your own experiment and stored in `outputs/results/`.

| File | Content |
|---|---|
| `metrics.json` | Main evaluation metrics (used by the report and slide scripts) |
| `history.json` | Training history |
| `dataset_stats.json` | Dataset statistics |
| `next_event_metrics.csv` | Next-event prediction metrics for all models |
| `generation_metrics.csv` | Objective metrics of generated music vs real test music |
| `loss_curve.png` | Training / validation loss |
| `next_event_comparison.png` | Prediction comparison |
| `generation_comparison.png` | Generation comparison |
| `piano_rolls.png` | Piano-roll visualisation |

The trained model is saved at `outputs/lstm.pt`.

> **Important:** Report the values from `results/metrics.json` and the CSV files after running your real experiment. Do not use smoke-test numbers as results.

## 14. Report and Presentation

The repository contains the final project documentation:

- 📄 Project report — `report/Music_Generation_Report.pdf`
- 🖥️ Presentation — `report/Music_Generation_Slides.pptx`

Both can be regenerated after running a new experiment:

```cmd
pip install reportlab python-pptx pypdf
python tools/make_report.py --out outputs --report report/Music_Generation_Report.pdf
python tools/make_slides.py --out outputs --pptx report/Music_Generation_Slides.pptx
```

Both read `outputs/results/metrics.json`, so re-running them updates every documented number. (Add `--base 7.8` to `make_report.py` if the report spills over two pages.)

## 15. Smoke Test

A pipeline-only smoke test is available:

```cmd
python run_all.py --smoke
```

This uses 30 synthetic files in `data_sample/` to verify that the pipeline works.

> ⚠️ Smoke-test numbers are **not meaningful** and must never be presented as real results. The smoke test never touches `data/` or `outputs/`.

## 16. Technologies

- Python 3.9+
- PyTorch
- LSTM / RNN
- pretty_midi
- NumPy
- Pandas
- Matplotlib
- ReportLab, python-pptx, pypdf (report and slides)
- pygame (optional, for playback)
- Git / GitHub

## 17. Limitations and Honest Notes

- The paper's Naive Bayes uses three consecutive left-hand notes as the chord; this project uses the pitch-class set sounding at the melody onset.
- The LSTM shares the paper's architecture and optimiser but is trained on whichever dataset is provided (the paper used 24 Chopin etudes), so results will differ.
- No human "music Turing test" is included; objective proxies are used instead.
- The representation does not model expressive performance details such as dynamics and articulation.
- Generated music is evaluated with objective metrics, which do not fully capture musical quality.

## 18. Future Work

- Add the paper's vanilla NN and encoder–decoder models.
- Train on a larger and more varied MIDI collection.
- Compare the LSTM with GRU and Transformer models.
- Represent pitch, duration, velocity and timing separately.
- Improve chord and rhythm modelling.
- Add a systematic human listening evaluation.
- Generate longer and more structurally coherent compositions.

## 19. References

- Kang, Kim, Ringdahl (2018). *Music Composition with Machine Learning*, Stanford University.
- [PyTorch](https://pytorch.org/)
- [pretty_midi](https://github.com/craffel/pretty-midi)
- [piano-midi.de](http://www.piano-midi.de/)

---
