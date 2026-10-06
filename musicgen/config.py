"""Global constants shared by every module."""
STEPS_PER_BEAT = 4            # 16th-note grid
SPLIT_PITCH = 60              # MIDI pitch >= 60 -> "right hand"/melody, < 60 -> "left hand"/chords
PITCH_MIN, PITCH_MAX = 21, 108  # 88 piano keys (A0..C8), as in the paper
N_PITCH = PITCH_MAX - PITCH_MIN + 1
DURATIONS = [1, 2, 3, 4, 6, 8, 12, 16]   # allowed event lengths, in 16th-note steps
N_DUR = len(DURATIONS)
MAX_CHORDS = 64               # chord vocabulary size (id 0 = "<other>")
MIN_EVENTS = 32               # songs with fewer melody events are discarded
SEQ_LEN = 64                  # LSTM training window (events)
MAX_EVAL_LEN = 256            # songs are truncated to this many events at test time
DEFAULT_TEMPO = 120
