"""mir_eval metrics with the paper's / reference code's tolerances."""
import numpy as np
from mir_eval import transcription, transcription_velocity, util


def note_scores(ref, est, offset_ratio=0.2, velocity=False, onset_tolerance=0.05, offset_min_tolerance=0.05):
    """ref/est: (n, 4) [onset, offset, midi, velocity]. offset_ratio=None ignores offsets; velocity=True also
    requires velocity within 0.1 after mir_eval's normalisation. Returns (precision, recall, f1)."""
    est = est[est[:, 1] > est[:, 0]]  # mir_eval rejects zero-length intervals (untrained models can emit them)
    r_hz, e_hz = util.midi_to_hz(ref[:, 2]), util.midi_to_hz(est[:, 2])
    kw = dict(onset_tolerance=onset_tolerance, offset_ratio=offset_ratio, offset_min_tolerance=offset_min_tolerance)
    if velocity:
        p, r, f, _ = transcription_velocity.precision_recall_f1_overlap(ref[:, :2], r_hz, ref[:, 3], est[:, :2], e_hz, est[:, 3], **kw)
    else:
        p, r, f, _ = transcription.precision_recall_f1_overlap(ref[:, :2], r_hz, est[:, :2], e_hz, **kw)
    return p, r, f


def pedal_scores(ref, est, offset_ratio=0.2, onset_tolerance=0.2, offset_min_tolerance=0.05):
    """ref/est: (m, 2) intervals. The reference code scores pedal events with a 200 ms onset tolerance."""
    est = est[est[:, 1] > est[:, 0]]
    p, r, f, _ = transcription.precision_recall_f1_overlap(ref, np.ones(len(ref)), est, np.ones(len(est)),
        onset_tolerance=onset_tolerance, offset_ratio=offset_ratio, offset_min_tolerance=offset_min_tolerance)
    return p, r, f


def frame_scores(ref_roll, output, threshold):
    """Frame-wise P/R/F1 of output > threshold vs the binary roll, over the common length."""
    n = min(len(ref_roll), len(output))
    y, p = ref_roll[:n] > 0.5, output[:n] > threshold
    tp, fp, fn = np.sum(y & p), np.sum(~y & p), np.sum(y & ~p)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-12)
