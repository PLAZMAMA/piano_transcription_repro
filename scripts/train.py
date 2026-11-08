"""Small-scale training with the paper's recipe (10 s segments, BCE losses of Eq. 9 / 15, Adam 5e-4 with amsgrad,
lr x0.9 every 10k iterations). Validates the training pipeline on a MAESTRO subset; the paper trains 300k iterations
at batch 12 on a V100, which is out of reach here."""
import argparse, json, os, sys, time
import numpy as np, soundfile as sf, torch

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P)
from hpt.model import NoteModel, PedalModel, SR
from hpt.data import SegmentDataset, note_loss, pedal_loss
from hpt.targets import load_piece, segment_targets
from hpt.inference import run_model
from hpt.postprocess import detect_all_notes, detect_all_pedals
from hpt.evaluate import note_scores, pedal_scores, frame_scores


def pieces(split, limit):
    rows = sorted((r for r in json.load(open(f"{P}/data/maestro/maestro-v2.0.0.json")) if r["split"] == split), key=lambda r: r["audio_filename"])
    out = [(f"{P}/data/maestro/audio16k/{r['audio_filename']}", f"{P}/data/maestro/midi/{r['midi_filename']}") for r in rows]
    return [(w, m) for w, m in out if os.path.exists(w)][:limit]


def validate(model, kind, val, device, seconds=120):
    """Transcribe the first `seconds` of each validation piece; F1 of events, events with offset, and frames."""
    scores = []
    for wav, midi in val:
        audio = (sf.read(wav, frames=seconds * SR, dtype="int16")[0] / 32767.0).astype(np.float32)
        out = run_model(model, audio, device)
        notes, off_ext, pedals = load_piece(midi)
        dur = len(audio) / SR
        rolls = segment_targets(notes, off_ext, pedals, 0.0, dur)
        if kind == "note":
            keep = notes[:, 0] < dur
            gt = np.column_stack([notes[keep, 0], np.minimum(off_ext[keep], dur), notes[keep, 2], notes[keep, 3]])
            est = detect_all_notes(out)
            scores.append([note_scores(gt, est, offset_ratio=None)[2], note_scores(gt, est)[2],
                           frame_scores(rolls["frame_roll"], out["frame_output"], 0.3)[2]])
        else:
            keep = pedals[:, 0] < dur
            gt = np.column_stack([pedals[keep, 0], np.minimum(pedals[keep, 1], dur)])
            est = detect_all_pedals(out)
            scores.append([pedal_scores(gt, est, offset_ratio=None)[2], pedal_scores(gt, est)[2],
                           frame_scores(rolls["pedal_frame_roll"], out["pedal_frame_output"][:, 0], 0.5)[2]])
    model.train()
    return dict(zip(["event_f1", "event_offset_f1", "frame_f1"], np.mean(scores, axis=0).tolist()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="note", choices=["note", "pedal"])
    ap.add_argument("--train_pieces", type=int, default=100)
    ap.add_argument("--val_pieces", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or f"{P}/results/train_{a.model}"
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)

    train, val = pieces("train", a.train_pieces), pieces("validation", a.val_pieces)
    ds = SegmentDataset(train)
    loader = torch.utils.data.DataLoader(ds, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    model = (NoteModel if a.model == "note" else PedalModel)().to(a.device)
    loss_fn = note_loss if a.model == "note" else pedal_loss
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
    print(f"{len(train)} train pieces -> {len(ds)} segments; {len(val)} validation pieces; device {a.device}", flush=True)

    log = open(f"{out_dir}/log.jsonl", "a")
    it, losses, t0 = 0, [], time.time()
    while it < a.iters:
        for batch in loader:
            if it % a.eval_every == 0:
                stats = {"iteration": it, **validate(model, a.model, val, a.device), "elapsed": time.time() - t0}
                log.write(json.dumps(stats) + "\n"); log.flush()
                print("validation", stats, flush=True)
            batch = {k: v.to(a.device) for k, v in batch.items()}
            model.train()
            loss = loss_fn(model(batch["waveform"]), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            it += 1
            if it % 10 == 0:
                rec = {"iteration": it, "loss": float(np.mean(losses[-10:])), "elapsed": time.time() - t0,
                       "mps_mem_gb": torch.mps.driver_allocated_memory() / 1e9 if a.device == "mps" else None}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
            if it % 10000 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 0.9
            if it >= a.iters:
                break
    stats = {"iteration": it, **validate(model, a.model, val, a.device), "elapsed": time.time() - t0}
    log.write(json.dumps(stats) + "\n"); log.flush()
    print("validation", stats, flush=True)
    torch.save(model.state_dict(), f"{out_dir}/model_{it}.pt")
