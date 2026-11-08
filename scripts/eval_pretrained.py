"""Evaluate the authors' released checkpoint on the MAESTRO test split with the re-implemented inference,
post-processing and metrics (paper Table I / VI protocol). Per-piece results go to a resumable jsonl; raw model
outputs are kept as float16 npz for the threshold sweeps."""
import argparse, json, os, sys, time
import numpy as np, soundfile as sf, torch

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P)
from hpt.model import NotePedalModel, load_official, SR
from hpt.targets import load_piece, segment_targets
from hpt.postprocess import detect_all_notes, detect_all_pedals
from hpt.inference import run_model
from hpt.evaluate import note_scores, pedal_scores, frame_scores

ONSET_TOLS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]  # Table IV
OFFSET_TOLS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]  # Table V


def evaluate_piece(out, notes, off_ext, pedals, dur):
    gt = np.column_stack([notes[:, 0], off_ext, notes[:, 2], notes[:, 3]])
    rolls = segment_targets(notes, off_ext, pedals, 0.0, dur)
    est = detect_all_notes(out, 0.3, 0.3, 0.3)
    est_ped = detect_all_pedals(out, 0.2, 0.5)
    r = {"n_ref_notes": len(gt), "n_est_notes": len(est), "n_ref_pedals": len(pedals), "n_est_pedals": len(est_ped)}
    r["frame"] = frame_scores(rolls["frame_roll"], out["frame_output"], 0.3)
    r["note"] = note_scores(gt, est, offset_ratio=None)
    r["note_offset"] = note_scores(gt, est, offset_ratio=0.2)
    r["note_offset_velocity"] = note_scores(gt, est, offset_ratio=0.2, velocity=True)
    r["onset_tolerance_sweep"] = {str(t): note_scores(gt, est, offset_ratio=None, onset_tolerance=t) for t in ONSET_TOLS}
    r["offset_tolerance_sweep"] = {str(t): note_scores(gt, est, offset_ratio=0.0, offset_min_tolerance=t) for t in OFFSET_TOLS}
    if len(pedals):
        r["pedal"] = pedal_scores(pedals, est_ped, offset_ratio=None)
        r["pedal_offset"] = pedal_scores(pedals, est_ped, offset_ratio=0.2)
        r["pedal_50ms"] = pedal_scores(pedals, est_ped, offset_ratio=None, onset_tolerance=0.05)
        r["pedal_offset_50ms"] = pedal_scores(pedals, est_ped, offset_ratio=0.2, onset_tolerance=0.05)
        r["pedal_frame"] = frame_scores(rolls["pedal_frame_roll"], out["pedal_frame_output"][:, 0], 0.5)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=f"{P}/results/eval_pretrained_test.jsonl")
    ap.add_argument("--probs_dir", default=f"{P}/results/probs")
    a = ap.parse_args()

    rows = sorted((r for r in json.load(open(f"{P}/data/maestro/maestro-v2.0.0.json")) if r["split"] == a.split), key=lambda r: r["audio_filename"])
    rows = [r for r in rows if os.path.exists(f"{P}/data/maestro/audio16k/{r['audio_filename']}")][:a.limit]
    done = {json.loads(l)["name"] for l in open(a.out)} if os.path.exists(a.out) else set()
    print(f"{len(rows)} pieces with audio, {len(done)} already evaluated", flush=True)
    model = load_official(NotePedalModel(), f"{P}/checkpoints/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth").to(a.device)
    os.makedirs(a.probs_dir, exist_ok=True)
    for i, row in enumerate(rows):
        name = os.path.splitext(row["audio_filename"])[0]
        if name in done:
            continue
        t0 = time.time()
        audio = (sf.read(f"{P}/data/maestro/audio16k/{name}.wav", dtype="int16")[0] / 32767.0).astype(np.float32)
        out = run_model(model, audio, a.device, a.batch_size)
        t1 = time.time()
        np.savez(f"{a.probs_dir}/{os.path.basename(name)}.npz", **{k: v.astype(np.float16) for k, v in out.items()})
        notes, off_ext, pedals = load_piece(f"{P}/data/maestro/midi/{name}.midi")
        r = {"name": name, "duration": len(audio) / SR, **evaluate_piece(out, notes, off_ext, pedals, len(audio) / SR)}
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"[{i + 1}/{len(rows)}] {name} {r['duration']:.0f}s infer {t1 - t0:.0f}s eval {time.time() - t1:.0f}s "
              f"note F1 {r['note'][2] * 100:.2f} w/off {r['note_offset'][2] * 100:.2f} frame {r['frame'][2] * 100:.2f}", flush=True)
