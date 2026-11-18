"""Transcribe every utterance with the live pipeline's Whisper and cache text embeddings of the transcripts.

Why: the fusion head was trained on gold MELD transcripts, but in the live pipeline it
sees Whisper's transcripts (WER ~0.32 on this audio). Training on the same kind of text
it gets at inference time removes that mismatch.

Uses the exact live code path (Stages.text_branch: same Whisper model, greedy decoding,
mask cache, same distilroberta pooling), on the ground-truth clips.

Outputs (git-ignored):
  data/asr/{split}.csv           utt_key, transcript, tokens   (row order == data/index/{split}.csv)
  data/emb/{split}_text_asr.npy  [N, 7, 768] fp16

Usage: python pipeline/embed/cache_asr.py [--asr-device mps]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

from pipeline.live.stages import StageOptions, Stages  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-device", default="mps")
    ap.add_argument("--splits", nargs="+", default=["test", "dev", "train"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    S = Stages(Path("models/fusion_text_audio_seed0.pt"), StageOptions(asr_device=args.asr_device, mask_cache=True))
    Path("data/asr").mkdir(parents=True, exist_ok=True)
    meta = {"asr_model": S.asr_model_name, "asr_device": args.asr_device, "mask_cache": True,
            "torch": torch.__version__, "splits": {}}

    for split in args.splits:
        out_csv, out_npy = Path(f"data/asr/{split}.csv"), Path(f"data/emb/{split}_text_asr.npy")
        if out_csv.exists() and out_npy.exists():
            print(f"{split}: already cached, skipping")
            continue
        idx = pd.read_csv(f"data/index/{split}.csv", keep_default_na=False)
        rows, embs = [], []
        t0 = time.perf_counter()
        for k, w in tqdm(zip(idx.utt_key, idx.wav), total=len(idx), desc=split, mininterval=10):
            x, _ = sf.read(w, dtype="float32")
            transcript, emb, t = S.text_branch(x)
            rows.append({"utt_key": k, "transcript": transcript, "tokens": t["asr_tokens"]})
            embs.append(emb.numpy().astype(np.float16))
        secs = time.perf_counter() - t0
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        np.save(out_npy, np.stack(embs))
        meta["splits"][split] = {"n": len(rows), "seconds": round(secs, 1),
                                 "empty_transcripts": int(sum(r["transcript"] == "" for r in rows))}
        print(f"{split}: {len(rows)} utterances in {secs:.0f} s, "
              f"{meta['splits'][split]['empty_transcripts']} empty transcripts", flush=True)
    Path("data/asr/meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
