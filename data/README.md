# Put your MIDI dataset here

Copy hundreds of real `.mid` / `.midi` files into this folder (sub-folders are fine), e.g.

* piano-midi.de (classical piano - used by the Stanford paper): https://www.piano-midi.de/
* Nottingham folk-tune MIDI: https://github.com/jukedeck/nottingham-dataset
* MAESTRO (piano performances): https://magenta.tensorflow.org/datasets/maestro
* Any other MIDI collection you are allowed to use.

The MIDI files themselves are git-ignored (check each dataset's licence before redistributing).
`python preprocess.py` refuses to run on fewer than 50 files so that the synthetic smoke-test files can never be mistaken for a dataset.
