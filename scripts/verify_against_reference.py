"""Cross-check the re-implementation against the official code on one MAESTRO piece:
model outputs, training targets, ground-truth events, stitching, post-processing and metrics."""
import os, sys, time
import numpy as np, soundfile as sf, torch

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [P, f"{P}/reference/utils", f"{P}/reference/pytorch"]
from hpt.model import NotePedalModel, load_official, SR
from hpt.targets import load_piece, segment_targets
from hpt.postprocess import detect_all_notes, detect_all_pedals
from hpt.inference import run_model, deframe
from hpt.evaluate import note_scores, pedal_scores, frame_scores
import utilities as ref_utils, models as ref_models, inference as ref_inference  # reference code

CKPT = f"{P}/checkpoints/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
name = sys.argv[1] if len(sys.argv) > 1 else "2013/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_14_R3_2013_wav--2"
wav, midi = f"{P}/data/maestro/audio16k/{name}.wav", f"{P}/data/maestro/midi/{name}.midi"
audio = (sf.read(wav, dtype="int16")[0] / 32767.0).astype(np.float32)
dur = len(audio) / SR
print(f"piece {name}: {dur:.0f}s")

# 1. model outputs on one 10 s segment
mine = load_official(NotePedalModel(), CKPT).eval()
ref = ref_models.Note_pedal(frames_per_second=100, classes_num=88)
ref.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True)["model"])
ref.eval()
x = torch.from_numpy(audio[None, 30 * SR:40 * SR])
with torch.no_grad():
    o1, o2 = mine(x), ref(x)
for k in o1:
    print(f"[model] {k:24s} max|diff| = {(o1[k] - o2[k]).abs().max():.2e}")

# 2. segment training targets: reference TargetProcessor vs segment_targets (one call per segment start)
md = ref_utils.read_midi(midi)
notes, off_ext, pedals = load_piece(midi)
tp = ref_utils.TargetProcessor(10.0, 100, 21, 88)
cells, maxd = {}, {}
for start in np.arange(0.0, dur - 10, 23.0):
    rt = tp.process(start, md["midi_event_time"], md["midi_event"], extend_pedal=True)[0]
    mt = segment_targets(notes, off_ext, pedals, start, 10.0)
    for key in mt:
        d = np.abs(np.asarray(rt[key], np.float32) - mt[key])
        cells.setdefault(key, []).append(np.mean(d > 1e-3))
        maxd[key] = max(maxd.get(key, 0.0), float(d.max()))
for key in cells:
    print(f"[targets] {key:22s} differing cells {np.mean(cells[key]) * 100:.4f}%  max|diff| {maxd[key]:.3f}")

# 3. whole-piece ground truth (reference events vs mine)
rt_full, ref_notes, ref_pedals = ref_utils.TargetProcessor(dur, 100, 21, 88).process(0, md["midi_event_time"], md["midi_event"], extend_pedal=True)
ref_gt = np.array([[e["onset_time"], e["offset_time"], e["midi_note"], e["velocity"]] for e in ref_notes])
my_gt = np.column_stack([notes[:, 0], off_ext, notes[:, 2], notes[:, 3]])
ref_gt, my_gt = ref_gt[np.lexsort(ref_gt.T[::-1])], my_gt[np.lexsort(my_gt.T[::-1])]
print(f"[gt] notes ref {len(ref_gt)} mine {len(my_gt)}; max|diff| {np.abs(ref_gt - my_gt).max():.2e}" if ref_gt.shape == my_gt.shape else "[gt] NOTE COUNT MISMATCH")
ref_pd = np.array([[e["onset_time"], e["offset_time"]] for e in ref_pedals])
print(f"[gt] pedals ref {len(ref_pd)} mine {len(pedals)}; max|diff| {np.abs(ref_pd - pedals).max():.2e}" if ref_pd.shape == pedals.shape else "[gt] PEDAL COUNT MISMATCH")
gt_full = segment_targets(notes, off_ext, pedals, 0.0, dur)
print(f"[gt] whole-piece frame_roll differing cells {np.mean(rt_full['frame_roll'] != gt_full['frame_roll']) * 100:.5f}%")

# 4. stitching of overlapping segments
z = np.random.rand(7, 1001, 88).astype(np.float32)
print(f"[deframe] max|diff| vs reference {np.abs(ref_inference.PianoTranscription.deframe(None, z) - deframe(z)).max():.1e}")

# 5. full-piece inference + post-processing vs reference RegressionPostProcessor
device = "mps" if torch.backends.mps.is_available() else "cpu"
t = time.time()
out = run_model(mine.to(device), audio, device, batch_size=8)
print(f"[inference] {dur:.0f}s of audio in {time.time() - t:.1f}s on {device}")
pp = ref_utils.RegressionPostProcessor(100, classes_num=88, onset_threshold=0.3, offset_threshold=0.3, frame_threshold=0.3, pedal_offset_threshold=0.2)
t = time.time()
ref_est_notes, ref_est_pedals = pp.output_dict_to_note_pedal_arrays({k: v.copy() for k, v in out.items()})
t_ref = time.time() - t
t = time.time()
my_est_notes, my_est_pedals = detect_all_notes(out), detect_all_pedals(out)
print(f"[postprocess] reference {t_ref:.1f}s, mine {time.time() - t:.1f}s")
ref_est_notes = ref_est_notes.astype(np.float64)
ref_est_notes[:, 3] *= 128
for label, a, b in (("est notes", ref_est_notes, my_est_notes), ("est pedals", ref_est_pedals.astype(np.float64), my_est_pedals)):
    a, b = a[np.lexsort(a.T[::-1])], b[np.lexsort(b.T[::-1])]
    print(f"[postprocess] {label}: ref {len(a)} mine {len(b)}; " + (f"max|diff| {np.abs(a - b).max():.2e}" if a.shape == b.shape else "COUNT MISMATCH"))

# 6. metrics on this piece (paper protocol)
for desc, kw in (("note", dict(offset_ratio=None)), ("note w/ offset", dict(offset_ratio=0.2)), ("note w/ offset & velocity", dict(offset_ratio=0.2, velocity=True))):
    p, r, f = note_scores(my_gt, my_est_notes, **kw)
    print(f"[metrics] {desc:26s} P {p * 100:.2f} R {r * 100:.2f} F1 {f * 100:.2f}")
p, r, f = frame_scores(gt_full["frame_roll"], out["frame_output"], 0.3)
print(f"[metrics] {'frame':26s} P {p * 100:.2f} R {r * 100:.2f} F1 {f * 100:.2f}")
for desc, kw in (("pedal event", dict(offset_ratio=None)), ("pedal event w/ offset", dict(offset_ratio=0.2))):
    p, r, f = pedal_scores(pedals, my_est_pedals, **kw)
    print(f"[metrics] {desc:26s} P {p * 100:.2f} R {r * 100:.2f} F1 {f * 100:.2f}")
p, r, f = frame_scores(gt_full["pedal_frame_roll"], out["pedal_frame_output"][:, 0], 0.5)
print(f"[metrics] {'pedal frame':26s} P {p * 100:.2f} R {r * 100:.2f} F1 {f * 100:.2f}")
