"""Evaluate a trained fusion head on the full MELD test set and the fixed 500-utterance subset.

The subset (bench/subsets/test_fixed_500.txt) is a seeded random sample of test utt_keys.
It is the input set every later latency benchmark uses, so its accuracy is reported here
as the reference that optimizations are compared against.

Usage (one command, after embeddings are cached and the head is trained):
  python pipeline/fusion/evaluate.py
  python pipeline/fusion/evaluate.py --model models/fusion_text_seed0.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.fusion.model import EMOTIONS, FusionHead  # noqa: E402

SUBSET_FILE = Path("bench/subsets/test_fixed_500.txt")


def make_subset(idx: pd.DataFrame, n=500, seed=0):
    keys = idx.utt_key.sample(n=n, random_state=seed).sort_values().tolist()
    SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_FILE.write_text("\n".join(keys) + "\n")
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/fusion_text_audio_seed0.pt"))
    ap.add_argument("--emb-dir", type=Path, default=Path("data/emb"))
    ap.add_argument("--index-dir", type=Path, default=Path("data/index"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.model, weights_only=True)
    model = FusionHead(ckpt["mode"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    idx = pd.read_csv(args.index_dir / "test.csv", keep_default_na=False)
    text = torch.from_numpy(np.load(args.emb_dir / "test_text.npy").astype(np.float32))
    audio = torch.from_numpy(np.load(args.emb_dir / "test_audio.npy").astype(np.float32))
    y = idx.label.values
    with torch.inference_mode():
        pred = model(text, audio).argmax(-1).numpy()

    keys = SUBSET_FILE.read_text().split() if SUBSET_FILE.exists() else make_subset(idx)
    sub = idx.utt_key.isin(set(keys)).values
    assert sub.sum() == len(keys), "subset keys missing from test index"

    result = {"model": str(args.model), "mode": ckpt["mode"]}
    for name, mask in (("full_test", np.ones(len(y), bool)), ("fixed_subset_500", sub)):
        result[name] = {"n": int(mask.sum()),
                        "weighted_f1": float(f1_score(y[mask], pred[mask], average="weighted")),
                        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro")),
                        "accuracy": float((y[mask] == pred[mask]).mean())}
        r = result[name]
        print(f"{name:>17} n={r['n']:4d}  weighted F1 {r['weighted_f1']:.4f}  macro F1 {r['macro_f1']:.4f}  "
              f"accuracy {r['accuracy']:.4f}")

    print("\nPer-class report (full test):")
    print(classification_report(y, pred, labels=range(7), target_names=EMOTIONS, digits=3, zero_division=0))
    cm = confusion_matrix(y, pred, labels=range(7))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(" " * 10 + " ".join(f"{e[:5]:>6}" for e in EMOTIONS))
    for e, row in zip(EMOTIONS, cm):
        print(f"{e:>9} " + " ".join(f"{v:6d}" for v in row))
    result["confusion_matrix"] = cm.tolist()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
