# Reproduction: High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times

Paper: Kong, Li, Song, Wang, Wang (2020), arXiv:2010.01815v3. Official code: github.com/bytedance/piano_transcription
(cloned under `reference/`, used only for cross-checks).

`hpt/` is a compact, self-contained re-implementation of the paper written from the paper text, with the official
code used to settle details the paper leaves open. Everything in it was cross-checked against the official code on
real MAESTRO data (see "Verification"). The authors' released checkpoint loads into it strictly.

Hardware for all runs: one Apple M5 laptop (16 GB unified memory, PyTorch 2.15 on MPS). That is enough to reproduce
the paper's evaluation (inference ~8x realtime) but not its training (300k iterations at batch 12 took the authors a
week on a 32 GB V100; here a batch of 2 is the most that fits and one iteration costs ~7.5 s). Training is therefore
validated at small scale only; see "Training".

## Layout

| path | what |
|---|---|
| `hpt/model.py` | log-mel front end, conv blocks, `AcousticModel` (4 conv blocks + FC768 + 2-layer biGRU), `NoteModel` (frame / onset / offset / velocity with the paper's conditioning), `PedalModel`, `load_official` |
| `hpt/targets.py` | MIDI parsing, pedal extension, Eq. (5) regression targets (J = 5), segment targets with the reference's border masks |
| `hpt/postprocess.py` | peak picking + sub-frame refinement (Eq. 10/11), Algorithm 1 (notes) and Algorithm 2 (pedals) |
| `hpt/inference.py` | 10 s segments with 50 % overlap, stitched like the reference |
| `hpt/evaluate.py` | mir_eval metrics with the paper's tolerances (and the reference code's 200 ms pedal onset tolerance) |
| `hpt/data.py` | training segments and the losses of Eq. (9) and (15) |
| `hpt/midi.py` | MIDI writer |
| `scripts/download_maestro.py` | pulls single pieces out of the 110 GB MAESTRO v2.0.0 zip with HTTP range requests, stores 16 kHz mono wav |
| `scripts/verify_against_reference.py` | cross-check against the official code (model, targets, ground truth, stitching, post-processing) |
| `scripts/eval_pretrained.py`, `scripts/summarize.py` | test-split evaluation of the released checkpoint, per piece, resumable; table against the paper |
| `scripts/threshold_sweep.py` | Fig. 7 style threshold sweeps from the saved model outputs |
| `scripts/train.py` | small-scale training with the paper's recipe |
| `scripts/transcribe.py` | audio file -> MIDI |
| `tests/test_hpt.py` | self-checks of the numpy parts |

## Setup and usage

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch numpy librosa mir_eval mido soundfile remotezip   # + torchlibrosa pandas h5py sox audioread scikit-learn for verify_against_reference.py
curl -L -o "checkpoints/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth" "https://zenodo.org/records/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth?download=1"
# MAESTRO v2.0.0 metadata + MIDI zip go to data/maestro/{maestro-v2.0.0.json,midi/}; audio:
python scripts/download_maestro.py --split test                 # all 178 pieces, 20.5 h, ~15 min
python scripts/download_maestro.py --split train --limit 100    # seeded random subset for the training check
python scripts/download_maestro.py --split validation --limit 10
python tests/test_hpt.py
python scripts/verify_against_reference.py
python scripts/eval_pretrained.py && python scripts/summarize.py results/eval_pretrained_test.jsonl
python scripts/threshold_sweep.py
python scripts/train.py --model note --batch_size 2 --iters 1500
python scripts/transcribe.py reference/resources/cut_liszt.mp3 out.mid
```

## Verification against the official code (`results/verify.log`, piece 2013/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_14_R3_2013_wav--2)

* Architecture: the released checkpoint loads with `strict=True`; the note model has 20,218,778 parameters, the number the paper reports.
* Model outputs on the same 10 s segment: max abs difference vs the official model 1.5e-4 to 5.3e-4 over all seven outputs (torchlibrosa's conv-based STFT vs `torch.stft`). MPS vs CPU: 9e-7.
* Ground truth: identical note events (11,826), identical pedal events (1,049), identical whole-piece frame roll.
* Segment training targets: all binary rolls identical except 0.0005 % of cells at segment borders; regression rolls differ in 0.4 % of cells (see deviations 1 and 2).
* Segment stitching (`deframe`): identical.
* Post-processing on the real outputs of the full piece: identical note tuples for all 88 pitches, identical pedal events; 6x faster than the reference loops.
* Demo file: `scripts/transcribe.py` on the repo's `cut_liszt.mp3` recovers 97.7 % of the notes of the `cut_liszt.mid` shipped with the repo (same precision at a 10 ms tolerance) but emits 507 notes against 412. That MIDI also contains 19 zero-length notes and does not change with thresholds, so it was evidently produced by an older checkpoint or post-processor; the current reference code and `hpt` agree exactly on model outputs.

## Deliberate deviations from the reference code

1. Regression targets follow Eq. (5) literally: distance to the nearest onset/offset. The reference's `get_regression` uses the previous event's sub-frame offset for frames past the midpoint between two events.
2. No "fake" offset label at the last frame of a segment for notes/pedals that continue past the segment end (the reference labels one; it is masked for notes anyway).
3. Peak refinement guards against a zero denominator (plateau peaks) by using shift 0 instead of inf.
4. Estimated events with non-positive duration are dropped before scoring (mir_eval rejects them; only untrained models produce them).
5. Pedal thresholds, the 200 ms pedal onset tolerance and per-piece averaging follow `reference/pytorch/calculate_score_for_paper.py`, which is what produced the paper's tables; the paper itself does not state them.

## Results: released checkpoint on the MAESTRO v2.0.0 test split

RESULTS_PLACEHOLDER

## Training (pipeline check only)

TRAINING_PLACEHOLDER

## Not reproduced

* Training to convergence (300k iterations, batch 12) and therefore the paper's training-based ablations (J in Table II, misaligned labels in Table III, the onsets-and-frames baseline rows).
* MAPS evaluation (the paper reports none in v3 either).
