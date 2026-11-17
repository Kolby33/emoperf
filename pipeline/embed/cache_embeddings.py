"""Cache frozen text and audio embeddings for every utterance in data/index/{split}.csv.

Text:  distilbert/distilroberta-base, all 7 hidden states (embeddings + 6 layers),
       each mean-pooled over real (non-pad) tokens -> [N, 7, 768] fp16.
Audio: facebook/wav2vec2-base, all 13 hidden states (CNN projection + 12 layers),
       each mean-pooled over time -> [N, 13, 768] fp16.

Keeping every layer lets the fusion head learn which layers matter (a learned
softmax-weighted layer mix) instead of us guessing; it costs ~350 MB total.

Audio runs one clip at a time: wav2vec2-base takes no attention mask and its
group-norm feature encoder would treat padding zeros as signal, so batching
padded clips would silently change the embeddings.

Row order of every .npy file == row order of data/index/{split}.csv.

Usage:
  python pipeline/embed/cache_embeddings.py --device mps --check-cpu 32
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

TEXT_MODEL = "distilbert/distilroberta-base"
AUDIO_MODEL = "facebook/wav2vec2-base"
SPLITS = ["train", "dev", "test"]


@torch.inference_mode()
def embed_text(texts, tok, model, device, batch_size=64, max_len=128):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="text", leave=False):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states, dim=1)  # [B, L, T, H]
        m = enc["attention_mask"][:, None, :, None].to(hs.dtype)
        out.append(((hs * m).sum(2) / m.sum(2)).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


@torch.inference_mode()
def embed_one_audio(wav_path, fe, model, device, max_seconds):
    x, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000, f"{wav_path}: expected 16 kHz, got {sr}"
    x = x[: int(max_seconds * sr)]
    if len(x) < 400:  # shorter than one 25 ms receptive field: pad so the CNN emits a frame
        x = np.pad(x, (0, 400 - len(x)))
    inp = fe(x, sampling_rate=sr, return_tensors="pt").input_values.to(device)
    hs = torch.stack(model(inp, output_hidden_states=True).hidden_states, dim=1)  # [1, L, T, H]
    return hs.mean(2)[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=Path, default=Path("data/index"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/emb"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--max-audio-seconds", type=float, default=20.0)
    ap.add_argument("--check-cpu", type=int, default=0,
                    help="re-embed this many audio clips on CPU and report the max difference")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    tmodel = AutoModel.from_pretrained(TEXT_MODEL).eval().to(device)
    fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
    amodel = AutoModel.from_pretrained(AUDIO_MODEL).eval().to(device)

    meta = {"text_model": TEXT_MODEL, "audio_model": AUDIO_MODEL, "device": args.device,
            "max_audio_seconds": args.max_audio_seconds, "torch": torch.__version__,
            "fe_do_normalize": bool(getattr(fe, "do_normalize", None)), "splits": {}}

    for split in SPLITS:
        idx = pd.read_csv(args.index_dir / f"{split}.csv", keep_default_na=False)
        t0 = time.perf_counter()
        text = embed_text(idx.text.astype(str).tolist(), tok, tmodel, device)
        t_text = time.perf_counter() - t0

        t0 = time.perf_counter()
        audio = np.stack([embed_one_audio(p, fe, amodel, device, args.max_audio_seconds).astype(np.float16)
                          for p in tqdm(idx.wav, desc=f"audio {split}", leave=False)])
        t_audio = time.perf_counter() - t0

        np.save(args.out_dir / f"{split}_text.npy", text)
        np.save(args.out_dir / f"{split}_audio.npy", audio)
        n_trunc = int((idx.duration_s > args.max_audio_seconds).sum())
        meta["splits"][split] = {"n": len(idx), "text_shape": list(text.shape), "audio_shape": list(audio.shape),
                                 "text_seconds": round(t_text, 1), "audio_seconds": round(t_audio, 1),
                                 "audio_clips_truncated": n_trunc}
        print(f"{split}: n={len(idx)} text {text.shape} in {t_text:.0f}s, audio {audio.shape} in {t_audio:.0f}s, "
              f"{n_trunc} clips truncated to {args.max_audio_seconds}s")

    if args.check_cpu and args.device != "cpu":
        idx = pd.read_csv(args.index_dir / "dev.csv", keep_default_na=False).head(args.check_cpu)
        cpu_model = AutoModel.from_pretrained(AUDIO_MODEL).eval()
        cached = np.load(args.out_dir / "dev_audio.npy")[: args.check_cpu].astype(np.float32)
        ref = np.stack([embed_one_audio(p, fe, cpu_model, torch.device("cpu"), args.max_audio_seconds)
                        for p in idx.wav])
        rel = np.abs(cached - ref).max() / np.abs(ref).max()
        cos = (cached * ref).sum(-1) / (np.linalg.norm(cached, axis=-1) * np.linalg.norm(ref, axis=-1))
        meta["mps_vs_cpu_check"] = {"n_clips": args.check_cpu, "max_abs_diff_over_max_abs": float(rel),
                                    "min_cosine": float(cos.min())}
        print(f"MPS vs CPU on {args.check_cpu} dev clips: max |diff|/max|ref| = {rel:.2e}, "
              f"min cosine = {cos.min():.6f}")

    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
