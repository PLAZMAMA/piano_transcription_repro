"""MIDI -> frame-level training targets (paper Sec. III-B/C; Eq. 5 with J = 5 frames = 50 ms)."""
import mido, numpy as np
from .model import FPS, N_KEYS, BEGIN_NOTE

J = 5  # regression target half-width in frames


def read_midi(path):
    """MAESTRO MIDI -> notes (n, 4) [onset, offset, pitch, velocity] in note-off order (as the reference parses
    them), pedals (m, 2) [onset, offset]. A re-struck pitch replaces its unpaired onset; sustain CC64 >= 64 is on."""
    mid = mido.MidiFile(path)
    tempo = next(m.tempo for m in mid.tracks[0] if m.type == "set_tempo")
    ticks_per_second = mid.ticks_per_beat * 1e6 / tempo  # MAESTRO files have a single tempo
    notes, pedals, active, pedal_on, ticks, t = [], [], {}, None, 0, 0.0
    for m in mid.tracks[1]:
        ticks += m.time
        t = ticks / ticks_per_second
        if m.type == "note_on" and m.velocity > 0:
            active[m.note] = (t, m.velocity)
        elif m.type in ("note_on", "note_off") and m.note in active:
            onset, vel = active.pop(m.note)
            notes.append([onset, t, m.note, vel])
        elif m.type == "control_change" and m.control == 64:
            if m.value >= 64:
                pedal_on = t if pedal_on is None else pedal_on
            elif pedal_on is not None:
                pedals.append([pedal_on, t])
                pedal_on = None
    notes += [[on, t, p, v] for p, (on, v) in active.items()]  # unpaired at the end of the piece
    if pedal_on is not None:
        pedals.append([pedal_on, t])
    return np.array(notes, dtype=np.float64).reshape(-1, 4), np.array(pedals, dtype=np.float64).reshape(-1, 2)


def extend_pedal(notes, pedals):
    """Port of the reference extend_pedal: a note ending inside a pedal interval is held until the pedal release;
    a re-struck pitch inside the same pedal truncates the earlier note at the new onset. Notes must be in the
    reference (note-off) order. Returns extended offsets (n,)."""
    off = notes[:, 1].copy()
    pos = 0
    for ped_on, ped_off in pedals:
        held = {}
        while pos < len(notes):
            k = pos
            pos += 1
            if ped_on < off[k] < ped_off:
                pitch = notes[k, 2]
                if pitch in held:
                    off[held.pop(pitch)] = notes[k, 0]
                off[k] = ped_off
                held[pitch] = k
            if off[k] > ped_off:
                break
    return off


def load_piece(midi_path):
    """-> notes (n, 4) sorted by onset, pedal-extended offsets (n,), pedals (m, 2)."""
    notes, pedals = read_midi(midi_path)
    off_ext = extend_pedal(notes, pedals)
    order = np.argsort(notes[:, 0], kind="stable")
    return notes[order], off_ext[order], pedals


def triangle(times, T):
    """Eq. (5): g = 1 - |t_frame - t_event| / (J / FPS) to the nearest event, clipped to [0, 1]."""
    out = np.zeros(T, np.float32)
    if len(times):
        ev = np.sort(times)
        t = np.arange(T) / FPS
        i = np.searchsorted(ev, t)
        d = np.minimum(np.abs(t - ev[np.maximum(i - 1, 0)]), np.abs(ev[np.minimum(i, len(ev) - 1)] - t))
        out = np.clip(1 - d * FPS / J, 0, 1).astype(np.float32)
    return out


def segment_targets(notes, off_ext, pedals, start, seconds):
    """Targets for [start, start + seconds]; rolls have round(seconds * FPS) + 1 frames. mask_roll zeroes notes that
    cross the segment borders (onset before start, or note-off after the end), as the reference does."""
    T = int(round(seconds * FPS)) + 1
    end = start + seconds
    frame, onset, offset, velocity, reg_onset, reg_offset = (np.zeros((T, N_KEYS), np.float32) for _ in range(6))
    mask = np.ones((T, N_KEYS), np.float32)
    on_times, off_times = [[] for _ in range(N_KEYS)], [[] for _ in range(N_KEYS)]
    for (on, off_raw, pitch, vel), off in zip(notes, off_ext):
        k = int(pitch) - BEGIN_NOTE
        b, f = int(round((on - start) * FPS)), int(round((off - start) * FPS))
        if not 0 <= k < N_KEYS or f < 0 or b > T - 1:
            continue
        fc = min(f, T - 1)
        frame[max(b, 0):fc + 1, k] = 1
        velocity[max(b, 0):fc + 1, k] = vel
        if f <= T - 1:
            offset[f, k] = 1
            off_times[k].append(off - start)
        if b >= 0:
            onset[b, k] = 1
            on_times[k].append(on - start)
            if off_raw > end:
                mask[b:, k] = 0
        else:
            mask[:fc + 1, k] = 0
    for k in range(N_KEYS):
        reg_onset[:, k] = triangle(on_times[k], T)
        reg_offset[:, k] = triangle(off_times[k], T)

    pedal_frame, pedal_onset, pedal_offset = (np.zeros(T, np.float32) for _ in range(3))
    pon, poff = [], []
    for on, off in pedals:
        b, f = int(round((on - start) * FPS)), int(round((off - start) * FPS))
        if f < 0 or b > T - 1:
            continue
        pedal_frame[max(b, 0):min(f, T - 1) + 1] = 1
        if f <= T - 1:
            pedal_offset[f] = 1
            poff.append(off - start)
        if b >= 0:
            pedal_onset[b] = 1
            pon.append(on - start)
    return {"frame_roll": frame, "onset_roll": onset, "offset_roll": offset, "velocity_roll": velocity,
            "reg_onset_roll": reg_onset, "reg_offset_roll": reg_offset, "mask_roll": mask,
            "pedal_frame_roll": pedal_frame, "pedal_onset_roll": pedal_onset, "pedal_offset_roll": pedal_offset,
            "reg_pedal_onset_roll": triangle(pon, T), "reg_pedal_offset_roll": triangle(poff, T)}
