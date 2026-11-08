"""Threshold sweeps (paper Fig. 7) from the saved model outputs: note F1 vs onset threshold (frame/offset thresholds 0.3)
and frame F1 / note-with-offset F1 vs frame threshold (onset/offset thresholds 0.3). Mean over pieces."""
import json, os, sys
from multiprocessing import Pool
import numpy as np, soundfile as sf

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P)
from hpt.targets import load_piece, segment_targets
from hpt.postprocess import detect_all_notes
from hpt.evaluate import note_scores, frame_scores

THRS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def one(name):
    out = {k: v.astype(np.float32) for k, v in np.load(f"{P}/results/probs/{os.path.basename(name)}.npz").items()}
    notes, off_ext, pedals = load_piece(f"{P}/data/maestro/midi/{name}.midi")
    gt = np.column_stack([notes[:, 0], off_ext, notes[:, 2], notes[:, 3]])
    dur = sf.info(f"{P}/data/maestro/audio16k/{name}.wav").duration
    frame_roll = segment_targets(notes, off_ext, pedals, 0.0, dur)["frame_roll"]
    r = {"name": name, "onset_thr": {}, "frame_thr": {}}
    for t in THRS:
        est = detect_all_notes(out, t, 0.3, 0.3)
        r["onset_thr"][str(t)] = [note_scores(gt, est, offset_ratio=None)[2], note_scores(gt, est)[2]]
        est = detect_all_notes(out, 0.3, 0.3, t)
        r["frame_thr"][str(t)] = [frame_scores(frame_roll, out["frame_output"], t)[2], note_scores(gt, est)[2]]
    return r


if __name__ == "__main__":
    names = [json.loads(l)["name"] for l in open(f"{P}/results/eval_pretrained_test.jsonl")]
    path = f"{P}/results/threshold_sweep_test.jsonl"
    done = {json.loads(l)["name"] for l in open(path)} if os.path.exists(path) else set()
    todo = [n for n in names if n not in done]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 5) as pool, open(path, "a") as f:
        for i, r in enumerate(pool.imap_unordered(one, todo)):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{i + 1}/{len(todo)}] {r['name']}", flush=True)
    rows = [json.loads(l) for l in open(path)]
    print(f"\n{len(rows)} pieces")
    print("onset threshold:  note F1 / note w/ offset F1   (frame, offset thresholds = 0.3)")
    for t in THRS:
        f1 = np.mean([r["onset_thr"][str(t)] for r in rows], axis=0) * 100
        print(f"  {t:4.2f}  {f1[0]:6.2f} / {f1[1]:6.2f}")
    print("frame threshold:  frame F1 / note w/ offset F1   (onset, offset thresholds = 0.3)")
    for t in THRS:
        f1 = np.mean([r["frame_thr"][str(t)] for r in rows], axis=0) * 100
        print(f"  {t:4.2f}  {f1[0]:6.2f} / {f1[1]:6.2f}")
