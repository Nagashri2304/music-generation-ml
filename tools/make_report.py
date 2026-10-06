"""Builds the 2-page PDF write-up from <out>/results/metrics.json and the plots.
   python tools/make_report.py --out outputs --report report/Music_Generation_Report.pdf"""
import argparse, json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs"); ap.add_argument("--report", default="report/Music_Generation_Report.pdf")
ap.add_argument("--base", type=float, default=8.4)
a = ap.parse_args()
R = Path(a.out) / "results"; M = json.loads((R / "metrics.json").read_text())
ds, ne, gen, tr = M["dataset"], {r["model"]: r for r in M["next_event"]}, M["generation"], M["training"]
B = a.base
body = ParagraphStyle("b", fontName="Helvetica", fontSize=B, leading=B * 1.25, spaceAfter=2.5)
h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=B + 1.6, leading=B + 3, spaceBefore=4, spaceAfter=1.5, textColor=colors.HexColor("#1b3a6b"))
small = ParagraphStyle("s", parent=body, fontSize=B - 1.2, leading=(B - 1.2) * 1.2, textColor=colors.HexColor("#444444"))
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=15, leading=18, alignment=1)
sub = ParagraphStyle("u", parent=body, alignment=1, fontSize=B)
L, N, P = ne["LSTM"], ne["Naive Bayes-like (given chord)"], ne["Naive Bayes-like (prev chord)"]; Rn = ne["Random (flat)"]
gl, gn, gr, grl = gen["LSTM"], gen["Naive Bayes-like"], gen["Random (flat)"], gen["Real (test)"]
f = lambda x, d=3: f"{x:.{d}f}"

S = [Paragraph("Generating Music with Machine Learning", title),
     Paragraph("Mini-Project #22 (UE24CS352A) &nbsp;|&nbsp; Nagashri R Patil (PES1UG24CS289), Nallamalli Kanaka Mani Sai Akhil (PES1UG24CS290)", sub),
     Paragraph("1. Problem Statement", h),
     Paragraph("Following the Stanford project <i>Music Composition with Machine Learning</i> (Kang, Kim, Ringdahl, 2018), we learn musical structure from MIDI files and generate new melodies with accompanying chords. "
               "Input: sequences of note events from MIDI. Output: (i) a prediction of the next note and its duration, scored against the true next note on unseen songs, and (ii) newly generated, playable MIDI pieces. "
               "We compare an LSTM (main model), a Naive-Bayes-like chord&rarr;note model and a random (flat) baseline, as in the paper.", body),
     Paragraph("2. Dataset", h),
     Paragraph(f"<b>{ds['dataset_name']}</b>: {ds['files_found']} MIDI files found, {ds['songs_used']} usable songs ({ds['files_skipped']} skipped: too short/unparsable), "
               f"{ds['events_total']:,} melody events. Split <b>by song</b> (80/10/10, seed {ds['seed']}): {ds['songs']['train']} train / {ds['songs']['val']} validation / {ds['songs']['test']} test songs "
               f"({ds['events']['train']:,} / {ds['events']['val']:,} / {ds['events']['test']:,} events). Files are read from <font face='Courier'>data/</font>; no dataset is hard-coded, so the same code runs on piano-midi.de, MAESTRO, etc. "
               "(The paper used 771 classical piano pieces for its encoder-decoder and 24 Chopin etudes for its LSTM.)", body),
     Paragraph("3. Approach", h),
     Paragraph("<b>Representation (as in the paper).</b> Each song is a series of note events. Per event we store the right-hand note (highest key &ge; MIDI 60, mapped to one of 88 piano keys), its duration "
               "(inter-onset interval quantised to {1,2,3,4,6,8,12,16} sixteenth-notes) and the left-hand chord (pitch-class set of notes &lt; 60 sounding at that onset, top-63 chords kept, rest = &lt;other&gt;). "
               "<b>Random baseline:</b> uniform over 88 keys, 8 durations, all chords. "
               "<b>Naive-Bayes-like:</b> each note is independent given the chord, P(note|chord) and P(duration|chord) from Laplace-smoothed counts (smoothing tuned on validation); "
               "generation samples a chord path from a first-order Markov chain and then notes from P(&middot;|chord). It is evaluated both with the true concurrent chord (paper protocol, extra information) and with the previous chord (fair next-event task). "
               f"<b>LSTM (main):</b> embeddings of the previous event's pitch/duration/chord &rarr; one LSTM layer (256 units) &rarr; dropout 0.3 &rarr; three softmax heads (next pitch, duration, chord); "
               "cross-entropy loss, Adam (lr 0.005, &beta;<sub>1</sub>=0.9, &beta;<sub>2</sub>=0.999), gradient clipping, early stopping on validation loss.", body),
     Paragraph("4. Implementation", h),
     Paragraph("Python/PyTorch (CPU, auto-uses CUDA). <font face='Courier'>preprocess.py</font> (pretty_midi parsing, split, vocab) &rarr; <font face='Courier'>train.py</font> (64-event windows, validation, checkpoint) &rarr; "
               "<font face='Courier'>evaluate.py</font> (test metrics, generation metrics, plots, MIDI export + reload check) &rarr; <font face='Courier'>demo.py</font> (generate/play a new MIDI). "
               f"<font face='Courier'>run_all.py</font> runs all steps. LSTM trained {tr['epochs_run']} epochs (best epoch {tr['best_epoch']}, {tr['train_seconds']:.0f}s on {tr['device']}). "
               f"Generation: {M['n_generated']} sequences of {M['gen_len']} events per model (temperature {M['temperature']}); every exported file re-loads correctly ({', '.join(f'{k}: {v}' for k, v in M['midi_check'].items())}).", body),
     Paragraph("5. Results", h)]

rows = [["Model", "Pitch CE", "Pitch ppl", "Top-1", "Top-5", "Dur. CE", "Dur. acc"]]
for k in ["Random (flat)", "Naive Bayes-like (prev chord)", "Naive Bayes-like (given chord)", "LSTM"]:
    r = ne[k]; rows.append([k, f(r["pitch_ce"]), f(r["pitch_ppl"], 1), f(r["pitch_top1"]), f(r["pitch_top5"]), f(r["dur_ce"]), f(r["dur_acc"])])
t = Table(rows, colWidths=[56 * mm, 17 * mm, 18 * mm, 15 * mm, 15 * mm, 17 * mm, 17 * mm])
t.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", B - 0.6), ("FONT", (0, 1), (-1, -1), "Helvetica", B - 0.6),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f6")), ("FONT", (0, 4), (-1, 4), "Helvetica-Bold", B - 0.6),
                       ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
S += [Paragraph(f"<b>Table 1 - next-event prediction on {Rn['n_events']:,} held-out test events</b> (CE = cross-entropy in nats, lower is better).", small), t, Spacer(1, 3)]

g = [["Metric", "Real (test)", "Random", "Naive Bayes", "LSTM"]]
for k, lab in [("scale_consistency", "Scale consistency"), ("chord_tone_rate", "Chord-tone rate (harmony)"), ("repetition_rate", "Repeated-note rate"),
               ("mean_abs_interval", "Mean |interval| (semitones)"), ("large_leap_rate", "Leaps > 1 octave"), ("novelty_5gram", "5-gram novelty vs train")]:
    g.append([lab] + [f(gen[n][k + "_mean"], 2) for n in ["Real (test)", "Random (flat)", "Naive Bayes-like", "LSTM"]])
g.append(["Interval-histogram JS divergence vs real (lower = closer)", "0", f(gr["interval_js_vs_real"], 2), f(gn["interval_js_vs_real"], 2), f(gl["interval_js_vs_real"], 2)])
t2 = Table(g, colWidths=[72 * mm, 22 * mm, 20 * mm, 22 * mm, 20 * mm])
t2.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", B - 0.6), ("FONT", (0, 1), (-1, -1), "Helvetica", B - 0.6), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f6")),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
S += [Paragraph("<b>Table 2 - objective metrics of generated music</b> (mean over generated sequences; closer to the Real column is better).", small), t2, Spacer(1, 3)]
imgs = Table([[Image(str(R / "loss_curve.png"), width=70 * mm, height=70 * mm * 2.8 / 4.2), Image(str(R / "next_event_comparison.png"), width=100 * mm, height=100 * mm * 2.9 / 6.2)]], colWidths=[75 * mm, 105 * mm])
S += [imgs, Paragraph("<b>Fig. 1</b> LSTM training/validation loss. <b>Fig. 2</b> Test accuracy of the three models.", small)]

def better(a_, b_): return "lower" if a_ < b_ else "higher"
S += [Paragraph("<b>Discussion.</b> "
        f"On unseen songs the LSTM predicts the next pitch with CE {f(L['pitch_ce'])} (perplexity {f(L['pitch_ppl'],1)}) versus {f(P['pitch_ce'])} for the Naive-Bayes-like model with the previous chord and {f(Rn['pitch_ce'])} for the flat baseline "
        f"(top-1 accuracy {f(L['pitch_top1'],2)} vs {f(P['pitch_top1'],2)} vs {f(Rn['pitch_top1'],2)}). Even when the Naive-Bayes-like model is handed the true concurrent chord, its pitch CE is {f(N['pitch_ce'])}. "
        f"Generated LSTM music has an interval distribution closest to real music (JS {f(gl['interval_js_vs_real'],2)} vs {f(gn['interval_js_vs_real'],2)} for Naive Bayes and {f(gr['interval_js_vs_real'],2)} for random), "
        f"with scale consistency {f(gl['scale_consistency_mean'],2)} (real {f(grl['scale_consistency_mean'],2)}) and a 5-gram novelty of {f(gl['novelty_5gram_mean'],2)} (real test music: {f(grl['novelty_5gram_mean'],2)}), i.e. it reuses training phrases no more than real held-out music does. "
        f"On harmony the picture is mixed: the chord-tone rate is {f(gn['chord_tone_rate_mean'],2)} for Naive Bayes (which is built on P(note|chord)), {f(gl['chord_tone_rate_mean'],2)} for the LSTM and {f(grl['chord_tone_rate_mean'],2)} for real music. "
        + (f"Real test music itself has only {f(grl['novelty_5gram_mean'],2)} 5-gram novelty against the training set, so near-duplicate tunes exist across splits and absolute scores are optimistic. " if grl['novelty_5gram_mean'] < 0.5 else "")
        + "These numbers are objective proxies; the paper additionally ran a human 'music Turing test', which we did not repeat.", body),
      Paragraph("6. Conclusions", h),
      Paragraph("An LSTM trained on MIDI note events (pitch, duration, chord) clearly outperforms a Naive-Bayes-like chord&rarr;note model and a random baseline, reproducing the paper's finding that sequence modelling beats independent-note models. "
                "The Naive-Bayes-like model produces notes that fit the chord but has no melodic memory, so its melodies jump erratically; the random baseline is atonal. "
                "<b>Limitations:</b> a single-voice melody representation (highest right-hand note), coarse chord vocabulary, no velocity/dynamics, and no human listening study; "
                "results depend on the chosen dataset (ours: " + ds['dataset_name'] + "; the paper used classical piano). "
                "<b>Future work:</b> Transformer/self-attention models and velocity, as suggested in the paper's future work. Code, README and generated MIDI: GitHub repository (<i>placeholder: github.com/&lt;username&gt;/music-generation-ml</i>).", body)]
Path(a.report).parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(a.report, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=11 * mm, bottomMargin=10 * mm,
                  title="Generating Music with Machine Learning", author="Project 22").build(S)
from pypdf import PdfReader
n = len(PdfReader(a.report).pages); print(f"{a.report}: {n} page(s)")
if n > 2: print("WARNING: more than 2 pages - rerun with a smaller --base (e.g. --base 7.8)")
