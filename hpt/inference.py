"""Whole-recording inference: 10 s segments with 50 % overlap, stitched like the reference (centre half of each inner segment)."""
import numpy as np, torch
from .model import SR

SEGMENT = 10 * SR


def run_model(model, audio, device, batch_size=8):
    """audio: float32 (samples,) at 16 kHz -> dict of (frames, K) arrays; frames = 100 per second of padded audio."""
    n = int(np.ceil(len(audio) / SEGMENT)) * SEGMENT
    audio = np.pad(audio, (0, n - len(audio)))
    segments = np.stack([audio[s:s + SEGMENT] for s in range(0, n - SEGMENT + 1, SEGMENT // 2)]).astype(np.float32)
    outs = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(segments), batch_size):
            for k, v in model(torch.from_numpy(segments[i:i + batch_size]).to(device)).items():
                outs.setdefault(k, []).append(v.cpu().numpy())
    return {k: deframe(np.concatenate(v)) for k, v in outs.items()}


def deframe(x):
    """(N, 1001, K) overlapping segment outputs -> (500 * N + 500, K)."""
    if len(x) == 1:
        return x[0]
    x = x[:, :-1]  # drop the extra frame produced by center=True
    q = x.shape[1] // 4
    return np.concatenate([x[0, :3 * q], *x[1:-1, q:3 * q], x[-1, q:]])
