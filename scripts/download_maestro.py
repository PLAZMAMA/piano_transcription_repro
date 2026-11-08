"""Pull single MAESTRO v2.0.0 pieces out of the 110 GB zip with HTTP range requests
and store them as 16 kHz mono int16 wav (the paper's input format)."""
import argparse, io, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor
import librosa, soundfile as sf
from remotezip import RemoteZip

URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "maestro")
_local = threading.local()


def fetch(audio_filename, attempts=4):
    out = os.path.join(ROOT, "audio16k", audio_filename)
    if os.path.exists(out):
        return "cached"
    t = time.time()
    for attempt in range(attempts):  # the connection occasionally drops mid-member; reopen and retry
        try:
            if not hasattr(_local, "zip"):
                _local.zip = RemoteZip(URL)
            data = _local.zip.read("maestro-v2.0.0/" + audio_filename)
            break
        except Exception as e:
            if hasattr(_local, "zip"):
                del _local.zip
            if attempt == attempts - 1:
                return f"FAILED after {attempts} attempts: {e!r}"
            time.sleep(5 * (attempt + 1))
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=16000)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out + ".part.wav", y, 16000, subtype="PCM_16")
    os.replace(out + ".part.wav", out)
    return f"{len(y) / 16000:.0f}s in {time.time() - t:.0f}s"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--limit", type=int, default=None, help="random subset size (seeded)")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    rows = [r for r in json.load(open(os.path.join(ROOT, "maestro-v2.0.0.json"))) if r["split"] == a.split]
    if a.limit:
        rows = random.Random(0).sample(rows, a.limit)
    names = [r["audio_filename"] for r in rows]
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (n, msg) in enumerate(zip(names, ex.map(fetch, names))):
            print(f"[{i + 1}/{len(names)}] {msg} {n}", flush=True)
