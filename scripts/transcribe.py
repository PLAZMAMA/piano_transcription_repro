"""Transcribe a piano recording to MIDI with the re-implementation and the authors' released weights."""
import argparse, os, sys, time
import librosa, torch

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P)
from hpt.model import NotePedalModel, load_official, SR
from hpt.inference import run_model
from hpt.postprocess import detect_all_notes, detect_all_pedals
from hpt.midi import write_midi

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("midi_out")
    ap.add_argument("--checkpoint", default=f"{P}/checkpoints/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    audio, _ = librosa.load(a.audio, sr=SR, mono=True)
    model = load_official(NotePedalModel(), a.checkpoint).to(a.device)
    t = time.time()
    out = run_model(model, audio, a.device)
    notes, pedals = detect_all_notes(out, 0.3, 0.3, 0.1), detect_all_pedals(out, 0.2, 0.5)  # reference inference.py thresholds
    write_midi(a.midi_out, notes, pedals)
    print(f"{len(audio) / SR:.1f}s of audio -> {len(notes)} notes, {len(pedals)} pedal events in {time.time() - t:.1f}s -> {a.midi_out}")
