"""Model outputs -> note / pedal events (paper Sec. III-D, Algorithms 1-2, Eq. 10-11); same decisions as the reference."""
import numpy as np
from .model import FPS, N_KEYS, BEGIN_NOTE


def peaks(x, threshold, neighbour):
    """Local maxima of a regression output x (T, K): x > threshold, non-decreasing over `neighbour` frames into the
    peak and non-increasing out of it. Returns binary peaks and the sub-frame shift of Eq. (10)/(11), in frames."""
    T = len(x)
    d = np.diff(x, axis=0)
    inc, dec = d >= 0, d <= 0  # inc[m]: x[m+1] >= x[m]; dec[m]: x[m+1] <= x[m]
    ok = x > threshold
    ok[:neighbour] = False
    ok[T - neighbour:] = False
    for i in range(neighbour):
        ok[neighbour:T - neighbour] &= inc[neighbour - i - 1:T - neighbour - i - 1]
        ok[neighbour:T - neighbour] &= dec[neighbour + i:T - neighbour + i]
    a, b, c = x[:-2], x[1:-1], x[2:]  # frames n-1, n, n+1
    shift = np.zeros_like(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift[1:-1] = np.where(a > c, (c - a) / (b - c) / 2, (c - a) / (b - a) / 2)
    return ok.astype(np.float32), np.nan_to_num(shift, nan=0.0, posinf=0.0, neginf=0.0) * ok


def detect_notes(frame, onset, onset_shift, offset, offset_shift, velocity, frame_threshold):
    """Reference note_detection_with_onset_offset_regress for one pitch, event-driven instead of frame-by-frame.
    A note opened at onset peak b closes at the earliest of: the next onset (ends at nxt-1, no offset shift); the
    first frame where frame <= threshold (ends at an earlier offset peak if that peak is closer to it than to b);
    600 frames after b; the last frame. Rows: [bgn, fin, onset_shift, offset_shift, velocity]."""
    T = len(frame)
    onsets = np.flatnonzero(onset)
    low = np.flatnonzero(frame <= frame_threshold)
    offs = np.flatnonzero(offset)
    first_after = lambda arr, i: arr[np.searchsorted(arr, i + 1)] if np.searchsorted(arr, i + 1) < len(arr) else T
    rows = []
    for j, b in enumerate(onsets):
        nxt = onsets[j + 1] if j + 1 < len(onsets) else T
        fd, oc = first_after(low, b), first_after(offs, b)
        cands = [(nxt, 0, "next"), (fd, 1, "fd"), (b + 600, 2, "cap")]
        if nxt == T and T - 1 > b:
            cands.append((T - 1, 2, "end"))
        i, _, kind = min(cands)
        if kind == "next":
            fin, s = max(nxt - 1, 0), 0.0
        elif kind == "fd":
            fin = oc if oc <= fd and oc - b > fd - oc else fd
            s = offset_shift[fin]
        else:
            fin, s = i, offset_shift[i]
        rows.append([b, fin, onset_shift[b], s, velocity[b]])
    return rows


def detect_all_notes(out, onset_threshold=0.3, offset_threshold=0.3, frame_threshold=0.3):
    """-> (n, 4) [onset s, offset s, MIDI pitch, velocity in [0, 1]] for all 88 keys."""
    on, on_shift = peaks(out["reg_onset_output"], onset_threshold, 2)
    off, off_shift = peaks(out["reg_offset_output"], offset_threshold, 4)
    rows = []
    for k in range(N_KEYS):
        for b, f, s1, s2, v in detect_notes(out["frame_output"][:, k], on[:, k], on_shift[:, k], off[:, k],
                                            off_shift[:, k], out["velocity_output"][:, k], frame_threshold):
            rows.append([(b + s1) / FPS, (f + s2) / FPS, k + BEGIN_NOTE, v * 128])  # Alg. 1: velocity * 128
    return np.array(rows, np.float64).reshape(-1, 4)


def detect_pedals(frame, offset, offset_shift, frame_threshold):
    """Reference pedal_detection_with_onset_offset_regress (paper Alg. 2): onset where the frame output rises
    through the threshold; offset at the first offset peak, else 10 frames after the frame output drops."""
    rows, bgn, fd, oc = [], None, None, None
    for i in range(1, len(frame)):
        if bgn is None and frame[i] >= frame_threshold and frame[i] > frame[i - 1]:
            bgn = i
        if bgn is not None and i > bgn:
            if fd is None and frame[i] <= frame_threshold:
                fd = i
            if oc is None and offset[i] == 1:
                oc = i
            if oc is not None:
                rows.append([bgn, oc, 0.0, offset_shift[oc]])
                bgn = fd = oc = None
            elif fd is not None and i - fd >= 10:
                rows.append([bgn, fd, 0.0, offset_shift[fd]])
                bgn = fd = oc = None
    return rows


def detect_all_pedals(out, pedal_offset_threshold=0.2, pedal_frame_threshold=0.5):
    """-> (m, 2) [onset s, offset s]. Thresholds are the reference evaluation values."""
    off, off_shift = peaks(out["reg_pedal_offset_output"], pedal_offset_threshold, 4)
    rows = detect_pedals(out["pedal_frame_output"][:, 0], off[:, 0], off_shift[:, 0], pedal_frame_threshold)
    return np.array([[(b + s1) / FPS, (f + s2) / FPS] for b, f, s1, s2 in rows], np.float64).reshape(-1, 2)
