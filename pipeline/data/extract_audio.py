"""Extract 16 kHz mono 16-bit WAV audio from MELD .mp4 clips.

Why 16 kHz mono: wav2vec2 and Whisper were both trained on 16 kHz single-channel
audio, so converting once here avoids resampling on every run.

Resumable: existing non-empty outputs are skipped. Every clip, including failures,
is written to a manifest CSV so nothing is dropped silently.

Usage:
  python pipeline/data/extract_audio.py --split train --videos-dir data/raw/extracted/train_splits \
      --out-dir data/audio/train --manifest data/audio/train_manifest.csv
"""

import argparse
import csv
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

# The test archive has 132 clips twice: "diaX_uttY.mp4" and "final_videos_testdiaX_uttY.mp4".
# The files differ, and for all 131 labeled pairs only the prefixed one matches the label's
# StartTime/EndTime (median |duration error| 0.00 s vs 1.49 s), so the prefixed file wins.
NAME_RE = re.compile(r"^(final_videos_test)?dia(\d+)_utt(\d+)\.mp4$")


def convert(src: Path, dst: Path) -> tuple[str, str]:
    if dst.exists() and dst.stat().st_size > 44:  # 44 bytes = empty WAV header
        return "skipped", ""
    tmp = dst.with_suffix(".tmp.wav")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 44:
        tmp.unlink(missing_ok=True)
        return "failed", (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
    tmp.rename(dst)
    return "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--videos-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    by_key, ignored, n_prefixed_wins = {}, [], 0
    for p in sorted(args.videos_dir.iterdir()):
        m = NAME_RE.match(p.name)
        if not m:
            ignored.append(p)
            continue
        key, prefixed = (int(m.group(2)), int(m.group(3))), m.group(1) is not None
        if key in by_key:
            n_prefixed_wins += 1
            if not prefixed:
                continue
        by_key[key] = p
    clips = sorted(by_key.items())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.split}: {len(clips)} clips, {n_prefixed_wins} duplicates resolved to the "
          f"'final_videos_test' copy, {len(ignored)} other files ignored")
    if ignored:
        print("  ignored examples:", [p.name for p in ignored[:5]])

    def job(item):
        (dia, utt), p = item
        dst = args.out_dir / f"dia{dia}_utt{utt}.wav"
        status, err = convert(p, dst)
        dur = sf.info(str(dst)).duration if dst.exists() else 0.0
        return {"split": args.split, "dialogue_id": dia, "utterance_id": utt, "source": p.name,
                "wav": str(dst), "duration_s": round(dur, 3), "status": status, "error": err}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(tqdm(ex.map(job, clips), total=len(clips), desc=args.split))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_fail = sum(r["status"] == "failed" for r in rows)
    hours = sum(r["duration_s"] for r in rows) / 3600
    print(f"{args.split}: {len(rows) - n_fail} ok, {n_fail} failed, {hours:.2f} h of audio -> {args.manifest}")


if __name__ == "__main__":
    main()
