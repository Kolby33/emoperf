"""Train the fusion head on cached embeddings; report weighted F1, macro F1 and baselines.

Model selection uses dev weighted F1 only; test is evaluated once per trained model.
Runs 3 seeds per mode and reports mean and std, because MELD dev/test are small
enough that one seed can be misleading.

Imbalance: cross-entropy with class weights proportional to 1/sqrt(class frequency).
Full inverse-frequency weights push macro F1 up but hurt weighted F1 a lot; sqrt is
the middle ground. This is a choice, not a law, and is logged in experiments.md.

Usage:
  python pipeline/fusion/train.py                    # all modes, seeds 0 1 2
  python pipeline/fusion/train.py --modes text+audio --seeds 0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.fusion.model import EMOTIONS, FusionHead  # noqa: E402


def load(split, emb_dir, index_dir):
    idx = pd.read_csv(index_dir / f"{split}.csv", keep_default_na=False)
    t = torch.from_numpy(np.load(emb_dir / f"{split}_text.npy").astype(np.float32))
    a = torch.from_numpy(np.load(emb_dir / f"{split}_audio.npy").astype(np.float32))
    y = torch.tensor(idx.label.values)
    assert len(t) == len(a) == len(y), split
    return t, a, y


def metrics(y, p):
    return {"weighted_f1": float(f1_score(y, p, average="weighted")),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "accuracy": float(accuracy_score(y, p))}


@torch.inference_mode()
def predict(model, t, a):
    model.eval()
    return model(t, a).argmax(-1).numpy()


def train_one(mode, seed, data, epochs, patience, lr, batch_size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (tt, ta, ty), (dt, da, dy) = data["train"], data["dev"]
    counts = torch.bincount(ty, minlength=len(EMOTIONS)).float()
    w = (counts.sum() / counts).sqrt()
    loss_fn = nn.CrossEntropyLoss(weight=w / w.mean())

    model = FusionHead(mode)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, bad = -1.0, None, -1, 0
    for epoch in range(epochs):
        model.train()
        for b in torch.randperm(len(ty), generator=g).split(batch_size):
            opt.zero_grad()
            loss_fn(model(tt[b], ta[b]), ty[b]).backward()
            opt.step()
        dev_wf1 = f1_score(dy, predict(model, dt, da), average="weighted")
        if dev_wf1 > best:
            best, best_epoch, bad = dev_wf1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=Path, default=Path("data/emb"))
    ap.add_argument("--index-dir", type=Path, default=Path("data/index"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=Path("bench/results/m1_baseline.json"))
    ap.add_argument("--modes", nargs="+", default=["text+audio", "text", "audio"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    torch.set_num_threads(4)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    data = {s: load(s, args.emb_dir, args.index_dir) for s in ("train", "dev", "test")}
    _, _, test_y = data["test"]

    train_y = data["train"][2]
    majority = int(torch.bincount(train_y).argmax())
    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "torch": torch.__version__,
              "n": {s: len(v[2]) for s, v in data.items()},
              "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "class_weighting": "1/sqrt(freq), normalized to mean 1",
              "majority_baseline_test": {"class": EMOTIONS[majority],
                                         **metrics(test_y, np.full(len(test_y), majority))},
              "runs": {}}

    for mode in args.modes:
        runs = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            model, best_epoch = train_one(mode, seed, data, args.epochs, args.patience, args.lr, args.batch_size)
            dt, da, dy = data["dev"]
            tt, ta, _ = data["test"]
            run = {"seed": seed, "best_epoch": best_epoch, "train_seconds": round(time.perf_counter() - t0, 1),
                   "dev": metrics(dy, predict(model, dt, da)), "test": metrics(test_y, predict(model, tt, ta))}
            if model.text_mix is not None:
                run["text_layer_weights"] = torch.softmax(model.text_mix.logits, 0).tolist()
            if model.audio_mix is not None:
                run["audio_layer_weights"] = torch.softmax(model.audio_mix.logits, 0).tolist()
            torch.save({"mode": mode, "state_dict": model.state_dict()},
                       args.models_dir / f"fusion_{mode.replace('+', '_')}_seed{seed}.pt")
            runs.append(run)
            print(f"{mode:>10} seed {seed}: epoch {best_epoch:2d}  dev wF1 {run['dev']['weighted_f1']:.4f}  "
                  f"test wF1 {run['test']['weighted_f1']:.4f}  macro {run['test']['macro_f1']:.4f}  "
                  f"({run['train_seconds']}s)")
        summary = {}
        for m in ("weighted_f1", "macro_f1", "accuracy"):
            vals = [r["test"][m] for r in runs]
            summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        report["runs"][mode] = {"seeds": runs, "test_summary": summary}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    mb = report["majority_baseline_test"]
    print(f"\nTest set (n={len(test_y)}), mean ± std over seeds {args.seeds}")
    print(f"{'model':>16} {'weighted F1':>14} {'macro F1':>14} {'accuracy':>14}")
    print(f"{'majority (' + mb['class'] + ')':>16} {mb['weighted_f1']:>14.4f} {mb['macro_f1']:>14.4f} "
          f"{mb['accuracy']:>14.4f}")
    for mode, r in report["runs"].items():
        s = r["test_summary"]
        print(f"{mode:>16} " + " ".join(f"{s[m]['mean']:>7.4f} ± {s[m]['std']:.4f}"
                                        for m in ("weighted_f1", "macro_f1", "accuracy")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
