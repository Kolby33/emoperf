"""M4: run the two encoder stages with ONNX Runtime (CPU and Core ML providers) vs PyTorch.

Scope: wav2vec2 (branch A's encoder, incl. layer pooling) and the Whisper encoder
(full 3000-frame input, as in the baseline). The Whisper decode loop stays in PyTorch:
exporting HF generate is a different project. Each output is compared to PyTorch's.

Usage: python bench/m4_ort.py --n 100
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
warnings.filterwarnings("ignore")
import onnxruntime as ort  # noqa: E402
from transformers import AutoFeatureExtractor, AutoModel, WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

from pipeline.live import optim  # noqa: E402


class AudioPooled(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.stack(self.m(x, output_hidden_states=True).hidden_states, 1).mean(2)[0]


class WhisperEnc(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, f):
        return optim.whisper_encode(self.enc, f).last_hidden_state


def export(model, example, path, dynamic_axis):
    if not path.exists():
        torch.onnx.export(model, (example,), str(path), input_names=["x"], output_names=["y"],
                          dynamic_axes={"x": {dynamic_axis: "T"}} if dynamic_axis is not None else None,
                          opset_version=18, dynamo=False)
    return path


def bench(fn, inputs):
    for x in inputs[:3]:
        fn(x)
    out, ts = [], []
    for x in inputs:
        t = time.perf_counter()
        out.append(fn(x))
        ts.append(time.perf_counter() - t)
    return out, np.array(ts) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--onnx-dir", type=Path, default=Path("data/onnx"))
    ap.add_argument("--out", type=Path, default=Path("bench/results/m4_ort.json"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    args.onnx_dir.mkdir(parents=True, exist_ok=True)

    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    a_model = AudioPooled(AutoModel.from_pretrained("facebook/wav2vec2-base").eval()).eval()
    proc = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    w_enc = WhisperEnc(WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").model.encoder).eval()

    idx = pd.read_csv("data/index/test.csv", keep_default_na=False).set_index("utt_key")
    keys = Path("bench/subsets/test_fixed_500.txt").read_text().split()[: args.n]
    clips = [sf.read(idx.loc[k, "wav"], dtype="float32")[0][:320000] for k in keys]
    a_in = [fe(x, sampling_rate=16000, return_tensors="pt").input_values for x in clips]
    w_in = [proc.feature_extractor(x, sampling_rate=16000, return_tensors="pt").input_features for x in clips]

    with torch.inference_mode():
        a_path = export(a_model, a_in[0], args.onnx_dir / "wav2vec2_pooled.onnx", 1)
        w_path = export(w_enc, w_in[0], args.onnx_dir / "whisper_base_en_encoder.onnx", None)

    print("onnxruntime", ort.__version__, "providers:", ort.get_available_providers())
    results = {"onnxruntime": ort.__version__, "torch": torch.__version__, "n": len(keys), "stages": {}}
    for stage, path, inputs, tmodel in (("audio_enc", a_path, a_in, a_model), ("asr_enc", w_path, w_in, w_enc)):
        with torch.inference_mode():
            ref, t_ref = bench(lambda x: tmodel(x).numpy(), inputs)
        rows = {"torch_cpu": {"p50_ms": float(np.median(t_ref)), "p95_ms": float(np.percentile(t_ref, 95))}}
        for prov in ("CPUExecutionProvider", "CoreMLExecutionProvider"):
            if prov not in ort.get_available_providers():
                rows[prov] = "not available"
                continue
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            try:
                t0 = time.perf_counter()
                sess = ort.InferenceSession(str(path), so, providers=[prov, "CPUExecutionProvider"])
                load = time.perf_counter() - t0
                out, t = bench(lambda x: sess.run(None, {"x": x.numpy()})[0], inputs)
                rel = max(np.abs(o - r).max() / np.abs(r).max() for o, r in zip(out, ref))
                rows[prov] = {"p50_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
                              "session_load_s": load, "max_rel_diff_vs_torch": float(rel),
                              "speedup_p50_vs_torch": float(np.median(t_ref) / np.median(t))}
            except Exception as e:  # noqa: BLE001 - record failures as results
                rows[prov] = f"failed: {type(e).__name__}: {str(e)[:200]}"
        results["stages"][stage] = rows
        print(stage, json.dumps(rows, indent=1))
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
