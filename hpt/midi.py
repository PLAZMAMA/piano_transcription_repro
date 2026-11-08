"""Write transcribed note / pedal events to a MIDI file (same tempo/resolution as MAESTRO files)."""
import mido


def write_midi(path, notes, pedals=()):
    """notes: (n, 4) [onset s, offset s, midi pitch, velocity 0-127]; pedals: (m, 2) [onset s, offset s]."""
    ticks_per_second = 384 * 2  # 384 ticks per beat at 120 bpm
    msgs = []
    for on, off, pitch, vel in notes:
        msgs.append((on, 1, mido.Message("note_on", note=int(pitch), velocity=int(min(max(vel, 1), 127)))))
        msgs.append((off, 0, mido.Message("note_off", note=int(pitch), velocity=0)))
    for on, off in pedals:
        msgs.append((on, 1, mido.Message("control_change", control=64, value=127)))
        msgs.append((off, 0, mido.Message("control_change", control=64, value=0)))
    msgs.sort(key=lambda m: (m[0], m[1]))  # offsets before onsets at equal times
    mid = mido.MidiFile(ticks_per_beat=384)
    meta = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000), mido.MetaMessage("time_signature", numerator=4, denominator=4)])
    track, prev = mido.MidiTrack(), 0
    for t, _, m in msgs:
        tick = max(int(round(t * ticks_per_second)), 0)
        m.time, prev = tick - prev, tick
        track.append(m)
    mid.tracks += [meta, track]
    mid.save(path)
