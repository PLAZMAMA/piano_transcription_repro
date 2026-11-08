"""Self-checks of the numpy parts; run with `python tests/test_hpt.py` (or pytest)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpt.targets import triangle, extend_pedal, segment_targets
from hpt.postprocess import peaks, detect_notes
from hpt.inference import deframe


def test_triangle_matches_eq5():
    g = triangle([1.003], 200)  # onset 3 ms after the centre of frame 100
    assert np.isclose(g[100], 0.94) and np.isclose(g[105], 0.06) and g[106] == 0 and np.isclose(g[96], 1 - 0.043 * 20)


def test_peak_shift_recovers_subframe_offset():
    for delta in (-0.4, -0.1, 0.0, 0.25, 0.45):  # true onset at frame 20 + delta
        x = np.clip(1 - np.abs(np.arange(60) - (20 + delta)) / 5, 0, 1)[:, None].astype(np.float32)
        ok, shift = peaks(x, 0.3, 2)
        assert ok[:, 0].sum() == 1 and ok[20, 0] == 1 and abs(shift[20, 0] - delta) < 1e-5


def test_note_closing_rules():
    T = 1000
    frame, onset, offset = np.ones(T), np.zeros(T), np.zeros(T)
    onset[[10, 15, 100, 700]] = 1
    frame[300:] = 0.0  # frame output drops at 300
    offset[290] = 1  # offset peak 10 frames before the drop, 190 after the onset -> closer to the drop
    frame[700:] = 1.0  # sounding again from the last onset
    rows = detect_notes(frame, onset, np.zeros(T), offset, np.zeros(T), np.ones(T), 0.3)
    assert [r[:2] for r in rows] == [[10, 14], [15, 99], [100, 290], [700, 999]]  # next onset, next onset, offset peak, last frame


def test_extend_pedal_and_segment_masks():
    notes = np.array([[1.0, 1.2, 60, 80], [1.1, 1.3, 62, 70], [2.5, 9.0, 64, 60]], dtype=float)  # note-off order
    pedals = np.array([[0.9, 2.0]])
    off = extend_pedal(notes, pedals)
    assert np.allclose(off, [2.0, 2.0, 9.0])
    t = segment_targets(notes, off, pedals, 0.0, 4.0)
    assert t["frame_roll"][150, 60 - 21] == 1 and t["frame_roll"][201, 60 - 21] == 0 and t["onset_roll"][100, 60 - 21] == 1
    assert t["mask_roll"][:, 64 - 21][250:].max() == 0 and t["mask_roll"][:, 64 - 21][:250].min() == 1  # note-off after the segment
    assert t["pedal_frame_roll"][90:201].all() and np.isclose(t["reg_pedal_offset_roll"][200], 1.0)


def test_deframe_shape():
    assert deframe(np.zeros((5, 1001, 88))).shape == (3000, 88) and deframe(np.zeros((1, 1001, 88))).shape == (1001, 88)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
