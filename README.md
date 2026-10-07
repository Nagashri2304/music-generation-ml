# 🎹 Music Generation with Machine Learning

### Symbolic Music Generation using MIDI, LSTM, and Probabilistic Modeling


**UE24CS352A — Machine Learning Mini-Project · PES University**

</div>

---

## 👥 Team Members

| Name | SRN |
|------|-----|
| Nagashri R Patil | PES1UG24CS289 |
| Nallamalli Kanaka Mani Sai Akhil | PES1UG24CS290 |

## 1. Problem Statement

The objective of this project is to develop a machine-learning system that learns sequential patterns from symbolic music represented as MIDI files and generates new musical compositions.

Instead of generating raw audio, the system works with structured musical events containing:

- Pitch
- Duration
- Chord / harmony

A sequence model learns relationships between these events and predicts the next musical event. The project also compares the LSTM with probabilistic and random baselines.

---

## 2. Project Overview

The complete pipeline is:

```text
MIDI Dataset
     |
     v
MIDI Parsing
     |
     v
Musical Event Extraction
     |
     v
Pitch / Duration / Chord Representation
     |
     v
Song-Level Train / Validation / Test Split
     |
     v
64-Event Sequence Creation
     |
     +------------------+------------------+
     |                  |                  |
     v                  v                  v
   LSTM          Naive-Bayes-like       Random
     |                  |                  |
     +------------------+------------------+
                        |
                        v
                 Music Generation
                        |
                        v
                 MIDI Reconstruction
                        |
                        v
             Evaluation and Visualization
                        |
                        v
                Generated MIDI Files
```

The implementation uses **Python** and **PyTorch**, with **pretty_midi** for MIDI processing and reconstruction.

---

## 3. Models

Three approaches are implemented and compared.

| Model | Role | Approach |
|-------|------|----------|
| 🧠 **LSTM** | Main model | Learns temporal dependencies between musical events |
| 📊 **Naive-Bayes-like** | Comparison model | Models `P(note \| chord)` with a Markov process over chords |
| 🎲 **Random** | Baseline | Samples notes, durations, and chords without learning sequence structure |

### Why compare three models?

The comparison helps determine whether the learned LSTM produces more structured music than simpler probabilistic or random approaches. The LSTM is therefore evaluated alongside baselines rather than in isolation.

---

## 4. LSTM Architecture

The main model is a recurrent neural network that learns relationships between previous musical events and the next event.

```text
Input Sequence
64 Musical Events
       |
       v
     LSTM
   256 Units
       |
       +----------------+----------------+
       |                |                |
       v                v                v
   Pitch Head     Duration Head      Chord Head
       |                |                |
       +----------------+----------------+
                        |
                        v
                   Next Event
```

### Training configuration

| Parameter | Value |
|-----------|-------|
| Model | LSTM |
| LSTM layers | 1 |
| Hidden units | 256 |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Sequence length | 64 events |
| Loss | Pitch CE + Duration CE + Chord CE |
| Maximum epochs | 40 |
| Early stopping | Yes |
| Device | CPU / CUDA when available |

The trained checkpoint is stored at:

```text
outputs/lstm.pt
```

---

## 5. Musical Representation

Each melody onset is converted into a structured musical event.

| Component | Representation |
|-----------|----------------|
| Pitch | 88-key piano range, represented as `0–87` |
| Duration | Quantized duration class |
| Chord | Pitch-class set of lower sounding notes |

The representation allows the model to learn melody and harmony together while keeping the sequence compact.

- **Pitch** — the melody uses the 88-key piano range (`0–87`).
- **Duration** — note durations are quantized to a fixed musical grid and mapped to discrete duration classes.
- **Chord** — the accompanying harmony is represented using the pitch-class information of lower notes sounding at the melody onset.

---

## 6. Dataset

The project works with collections of symbolic MIDI files.

Supported formats:

```text
.mid
.midi
```

Place the MIDI files inside:

```text
data/
```

The project does not hard-code a particular MIDI dataset. For a meaningful real experiment, the preprocessing pipeline expects **at least 50 MIDI files**.

Additional dataset information is available in [`data/README.md`](data/README.md).

### Dataset split

Songs are divided at the **song level**:

| Split | Percentage |
|-------|-----------|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

Splitting at the song level helps reduce leakage between training and evaluation sequences.

---

## 7. Dataset Preprocessing

The preprocessing pipeline:

1. Reads MIDI files from `data/`.
2. Parses melody and accompaniment information.
3. Extracts pitch, duration, and chord events.
4. Builds the chord vocabulary from training songs.
5. Converts events into numerical representations.
6. Splits songs into training, validation, and test sets.
7. Creates 64-event sequence windows.
8. Saves the processed dataset for training and evaluation.

The main processed file is:

```text
outputs/processed.pkl
```

---

## 8. Installation

### Requirements

- Python 3.9 or later
- Git
- A MIDI dataset
- CPU is sufficient for running the project
- CUDA is used automatically when available

### Clone the repository

```bash
git clone https://github.com/Nagashri2304/music-generation-ml.git
cd music-generation-ml
```

### Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Optional MIDI playback:

```bash
pip install pygame
```

Optional report and presentation generation:

```bash
pip install reportlab python-pptx pypdf
```

---

## 9. Running the Project

### Step 1 — Add the MIDI dataset

Place your MIDI files inside `data/`, then verify that the directory contains enough files for the experiment.

### Step 2 — Preprocess the dataset

```bash
python preprocess.py --data-dir data --out outputs --dataset-name "piano-midi.de"
```

This prepares the musical event sequences for training and evaluation.

### Step 3 — Train the LSTM

```bash
python train.py --out outputs --epochs 40
```

The trained model is saved as `outputs/lstm.pt`.

### Step 4 — Evaluate the models

```bash
python evaluate.py --out outputs
```

This evaluates the models, generates music, calculates metrics, and creates visualizations.

---

## 10. Run the Complete Pipeline

The complete workflow can also be executed with a single command:

```bash
python run_all.py --dataset-name "piano-midi.de"
```

The pipeline performs:

```text
Preprocessing -> LSTM Training -> Model Evaluation -> Music Generation -> Metrics and Visualizations
```

Generated files are stored under `outputs/`.

---

## 11. Live Music Generation Demo

For a live demonstration, run:

```bash
python demo.py --play
```

The LSTM generates a new composition and saves it to `outputs/demo/demo_lstm.mid`. If `pygame` is installed, the generated MIDI can be played directly.

The file can also be opened with any MIDI-compatible application, such as:

- MuseScore
- VLC
- Windows Media Player

### Compare the models

```bash
# LSTM
python demo.py --play

# Naive-Bayes-like model
python demo.py --model nb --seed 3

# Random baseline
python demo.py --model random
```

### Generate multiple compositions

```bash
python demo.py --count 3 --events 128 --temperature 0.9
```

### Useful parameters

| Flag | Description |
|------|-------------|
| `--model` | Model to use (`lstm`, `nb`, `random`) |
| `--count` | Number of compositions to generate |
| `--events` | Number of musical events per composition |
| `--temperature` | Sampling randomness |
| `--seed` | Random seed for reproducibility |
| `--play` | Play the result with pygame |

A lower temperature generally produces more conservative predictions, while a higher temperature introduces more variation.

---

## 12. Generated Music

Generated MIDI files are stored in `outputs/generated_music/`.

| Model | Files |
|-------|-------|
| LSTM | `lstm_01.mid` … `lstm_05.mid` |
| Naive-Bayes-like | `nb_01.mid` … `nb_05.mid` |
| Random baseline | `random_01.mid` … `random_05.mid` |

Demo files are stored in `outputs/demo/`:

```text
demo_lstm.mid
demo_nb.mid
demo_random.mid
```

---

## 13. Evaluation

The project evaluates the models in two stages.

### 13.1 Next-Event Prediction

The models are evaluated on held-out musical events. Metrics include:

- Pitch Cross-Entropy
- Pitch Perplexity
- Top-1 Accuracy
- Top-5 Accuracy
- Duration Cross-Entropy
- Duration Accuracy

Results: `outputs/results/next_event_metrics.csv`

### 13.2 Generated Music Evaluation

Generated compositions are evaluated using objective musical statistics.

| Metric | Purpose |
|--------|---------|
| Scale consistency | Measures tonal consistency |
| Chord-tone rate | Measures harmonic alignment |
| Repeated-note rate | Measures repetition |
| Interval statistics | Measures melodic movement |
| Jensen-Shannon divergence | Compares musical distributions |
| 5-gram novelty | Measures new sequence patterns |

Results: `outputs/results/generation_metrics.csv`

> These metrics are objective proxies and do not completely measure human perception of musical quality.

---

## 14. Visualizations

| File | Description |
|------|-------------|
| `loss_curve.png` | Training and validation loss |
| `next_event_comparison.png` | Next-event prediction comparison |
| `generation_comparison.png` | Generated-music metric comparison |
| `piano_rolls.png` | Piano-roll representation of generated music |

All visualization files are stored in `outputs/results/`.

---

## 15. Repository Structure

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
│   │
│   ├── generated_music/
│   │   ├── lstm_01.mid ... lstm_05.mid
│   │   ├── nb_01.mid ... nb_05.mid
│   │   └── random_01.mid ... random_05.mid
│   │
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
│   │
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
├── .gitignore
├── README.md
├── demo.py
├── evaluate.py
├── make_sample_midi.py
├── preprocess.py
├── requirements.txt
├── run_all.py
└── train.py
```

---

## 16. Important Files

| File | Purpose |
|------|---------|
| `preprocess.py` | MIDI preprocessing and dataset preparation |
| `train.py` | LSTM model training |
| `evaluate.py` | Model evaluation and music generation |
| `demo.py` | Live music-generation demonstration |
| `run_all.py` | Complete project pipeline |
| `make_sample_midi.py` | Creates sample MIDI data |
| `musicgen/models.py` | LSTM and baseline model definitions |
| `musicgen/data.py` | Dataset and sequence handling |
| `musicgen/generation.py` | Music-generation logic |
| `musicgen/midi_io.py` | MIDI parsing and reconstruction |
| `musicgen/metrics.py` | Evaluation metrics |
| `musicgen/plotting.py` | Visualization utilities |
| `musicgen/config.py` | Project configuration |
| `tools/make_report.py` | Generates the PDF report |
| `tools/make_slides.py` | Generates the PowerPoint presentation |

---

## 17. Results and Output Files

The main experiment results are stored in `outputs/results/`.

| File | Description |
|------|-------------|
| `metrics.json` | Main evaluation metrics |
| `history.json` | Training history |
| `dataset_stats.json` | Dataset statistics |
| `next_event_metrics.csv` | Next-event prediction metrics |
| `generation_metrics.csv` | Generated music metrics |
| `loss_curve.png` | Training and validation loss |
| `next_event_comparison.png` | Prediction comparison |
| `generation_comparison.png` | Generation comparison |
| `piano_rolls.png` | Piano-roll visualization |

The trained model is `outputs/lstm.pt`.

---

## 18. Report and Presentation

The repository contains the final project documentation.

- 📄 **Project Report** — `report/Music_Generation_Report.pdf`
- 🖥️ **Presentation** — `report/Music_Generation_Slides.pptx`

Both can be regenerated after running a new experiment.

```bash
# Generate the report
python tools/make_report.py --out outputs --report report/Music_Generation_Report.pdf

# Generate the presentation
python tools/make_slides.py --out outputs --pptx report/Music_Generation_Slides.pptx
```

The report and slides use `outputs/results/metrics.json`, so the documented results update after a new experiment.

---

## 19. Smoke Test

A pipeline-only smoke test is available:

```bash
python run_all.py --smoke
```

This uses synthetic sample data to verify that the project pipeline works correctly.

> ⚠️ Smoke-test results are only for checking the implementation. They should not be presented as real experimental results.

The smoke test does not modify the actual `data/`, `outputs/`, or experiment directories.

---

## 20. How the System Works

### Musical Representation — `musicgen/midi_io.py`

Each melody onset is represented as:

```text
[right-hand key, duration class, left-hand chord ID]
```

- The right hand is the highest note at or above MIDI note 60.
- The left-hand chord is the pitch-class set of lower sounding notes.
- Durations are quantized to a 16th-note grid.

### Dataset Processing — `musicgen/data.py`

The system:

- Performs an 80/10/10 song-level split.
- Builds the chord vocabulary from training songs.
- Creates 64-event sequence windows.
- Generates next-event prediction targets.

### Model Training — `musicgen/models.py`

The LSTM predicts pitch, duration, and chord using the combined loss:

```text
Loss = CrossEntropy(Pitch) + CrossEntropy(Duration) + CrossEntropy(Chord)
```

### Music Generation — `musicgen/generation.py`

The model predicts one event at a time. Each generated event becomes part of the input for predicting the next one. The generated sequence is then converted back into MIDI.

### MIDI Reconstruction — `musicgen/midi_io.py`

```text
Melody Track + Chord Track = Playable MIDI
```

---

## 21. Reference and Implementation Notes

This project is inspired by:

> Kang, J., Kim, K., & Ringdahl, D. (2018). *Music Composition with Machine Learning.*

The implementation follows the core ideas of the reference work and is not an exact reproduction. Important differences:

- The project uses a configurable MIDI dataset rather than the original dataset.
- The Naive-Bayes-like representation uses the pitch-class set sounding at the melody onset.
- The LSTM implementation and training configuration are adapted for this project.
- Evaluation uses objective musical statistics.
- Results depend on the dataset and training configuration.

---

## 22. Limitations

- Generated music quality cannot be fully captured by numerical metrics.
- Quality depends on the dataset used for training.
- Some generated sequences may become repetitive depending on the random seed and sampling temperature.
- The representation simplifies expressive musical information.
- Dynamics, articulation, pedal information, and detailed performance timing are not fully modeled.
- The system generates symbolic MIDI rather than raw audio.
- Human evaluation is not part of the automated pipeline.

---

## 23. Future Work

- Transformer-based music generation
- Attention mechanisms
- Genre-conditioned generation
- Improved polyphonic modeling
- Longer-form composition
- Better chord progression modeling
- User-controlled tempo and style
- Real-time MIDI generation
- Human listening evaluation
- Larger and more diverse MIDI datasets
- Web-based music-generation interface

---

## 24. Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| PyTorch | LSTM and model training |
| pretty_midi | MIDI processing and reconstruction |
| NumPy | Numerical processing |
| pandas | Data handling |
| scikit-learn | Evaluation utilities |
| Matplotlib | Visualization |
| pygame | Optional MIDI playback |
| ReportLab | PDF report generation |
| python-pptx | PowerPoint generation |
| Git / GitHub | Version control |

---

## 25. Project Information

| Item | Details |
|------|---------|
| Course | UE24CS352A — Machine Learning Mini-Project |
| Institution | PES University |
| Project | Mini-Project 22 |
| Domain | Machine Learning / Deep Learning |
| Task | Symbolic Music Generation |
| Input | MIDI files |
| Output | Generated MIDI compositions |
| Main Model | LSTM |
| Comparison Models | Naive-Bayes-like, Random |

---

## 26. References

- MAESTRO Dataset
- PyTorch
- pretty_midi
- MIDI Association

---

<div align="center">

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
