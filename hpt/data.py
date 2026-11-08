"""Training data (random 10 s segments on a 1 s grid, like the reference sampler) and the paper's losses (Eq. 9, 15)."""
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from .model import SR
from .targets import load_piece, segment_targets

SEG_SECONDS = 10.0


class SegmentDataset(torch.utils.data.Dataset):
    def __init__(self, pieces):  # pieces: [(wav_path, midi_path), ...]
        self.pieces = [(w, *load_piece(m)) for w, m in pieces]
        self.index = [(i, s) for i, (w, *_) in enumerate(self.pieces) for s in range(int(sf.info(w).duration - SEG_SECONDS))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, start = self.index[j]
        wav, notes, off_ext, pedals = self.pieces[i]
        audio, _ = sf.read(wav, start=start * SR, frames=int(SEG_SECONDS * SR), dtype="int16")
        d = segment_targets(notes, off_ext, pedals, float(start), SEG_SECONDS)
        d["waveform"] = (audio / 32767.0).astype(np.float32)
        return d


def masked_bce(output, target, mask):
    output = output.clamp(1e-7, 1 - 1e-7)
    bce = -target * torch.log(output) - (1 - target) * torch.log(1 - output)
    return torch.sum(bce * mask) / torch.sum(mask).clamp(min=1)


def note_loss(out, t):
    """Eq. (9): BCE on onset/offset regression and frame targets (masked), velocity BCE at onsets (target vel/128)."""
    return (masked_bce(out["reg_onset_output"], t["reg_onset_roll"], t["mask_roll"])
            + masked_bce(out["reg_offset_output"], t["reg_offset_roll"], t["mask_roll"])
            + masked_bce(out["frame_output"], t["frame_roll"], t["mask_roll"])
            + masked_bce(out["velocity_output"], t["velocity_roll"] / 128, t["onset_roll"]))


def pedal_loss(out, t):
    """Eq. (15): BCE on pedal onset/offset regression and pedal frame targets."""
    pairs = (("reg_pedal_onset_output", "reg_pedal_onset_roll"), ("reg_pedal_offset_output", "reg_pedal_offset_roll"),
             ("pedal_frame_output", "pedal_frame_roll"))
    return sum(F.binary_cross_entropy(out[o], t[r][:, :, None]) for o, r in pairs)
