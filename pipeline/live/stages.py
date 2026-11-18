"""Inference stages of the live pipeline, each timed separately.

  branch A (audio):  audio_feat -> audio_enc                                    -> [13, 768]
  branch B (text):   asr_feat -> asr_enc -> asr_dec (Whisper) -> text_enc        -> transcript, [7, 768]
  fusion:            fusion head -> emotion
  response:          deterministic template from emotion + transcript snippet

Preprocessing matches pipeline/embed/cache_embeddings.py exactly (same models, same
pooling), so the fusion head sees the same kind of input it was trained on.

StageOptions holds the M4 experiment switches; the defaults are the M2 baseline.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoFeatureExtractor, AutoModel, AutoTokenizer, WhisperForConditionalGeneration,
                          WhisperProcessor)

from pipeline.fusion.model import EMOTIONS, FusionHead
from pipeline.live import optim

TEXT_MODEL = "distilbert/distilroberta-base"
AUDIO_MODEL = "facebook/wav2vec2-base"
ASR_MODEL = "openai/whisper-base.en"
SR = 16000

RESPONSES = {
    "anger": "That sounds really frustrating.",
    "disgust": "That sounds pretty unpleasant.",
    "fear": "That sounds worrying. Are you okay?",
    "joy": "That's great to hear!",
    "neutral": "Got it.",
    "sadness": "I'm sorry, that sounds hard.",
    "surprise": "Wow, I didn't see that coming either.",
}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class StageOptions:
    asr_model: str = ASR_MODEL
    device: str = "cpu"                 # device for all three models ("cpu" or "mps")
    asr_device: str | None = None       # override for Whisper only
    audio_device: str | None = None     # override for wav2vec2 only
    dtype: str = "fp32"                 # encoder/ASR weights and activations: fp32, bf16, fp16
    int8_dynamic: bool = False          # torch.ao dynamic int8 on every nn.Linear of the 3 models
    compile: bool = False               # torch.compile the three encoders (Whisper decoder stays eager)
    mask_cache: bool = False            # M3 hotspot 2 fix
    asr_trim_pad_s: float | None = None  # M3 hotspot 1: keep audio + pad seconds of mel frames...
    asr_trim_bucket_s: float = 1.0      # ...rounded up to this bucket
    branch_threads: tuple | None = None  # (audio, text) intra-op threads, set inside each branch thread
    max_audio_seconds: float = 20.0
    max_new_tokens: int = 96


def make_response(emotion: str, transcript: str, max_words: int = 8) -> str:
    words = transcript.strip().split()
    snippet = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
    quote = f' You said: "{snippet}"' if snippet else ""
    return f"[{emotion}] {RESPONSES[emotion]}{quote}"


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


class Stages:
    def __init__(self, head_path: Path, opts: StageOptions | None = None, device: str | None = None):
        o = opts or StageOptions()
        if device is not None:
            o.device = device
        self.opts = o
        self.asr_model_name = o.asr_model
        self.max_samples = int(o.max_audio_seconds * SR)
        self.max_new_tokens = o.max_new_tokens
        self.device = torch.device(o.device)
        self.asr_dev = torch.device(o.asr_device or o.device)
        self.audio_dev = torch.device(o.audio_device or o.device)
        self.dtype = DTYPES[o.dtype]
        self._tls = threading.local()

        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.text_model = AutoModel.from_pretrained(TEXT_MODEL).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
        self.audio_model = AutoModel.from_pretrained(AUDIO_MODEL).eval()
        self.asr_proc = WhisperProcessor.from_pretrained(o.asr_model)
        self.asr = WhisperForConditionalGeneration.from_pretrained(o.asr_model).eval()

        if o.int8_dynamic:
            torch.backends.quantized.engine = "qnnpack"
            q = lambda m: torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)  # noqa
            self.text_model, self.audio_model, self.asr = q(self.text_model), q(self.audio_model), q(self.asr)
        if not o.int8_dynamic:  # quantized modules stay CPU fp32-activation
            self.text_model.to(self.device, self.dtype)
            self.audio_model.to(self.audio_dev, self.dtype)
            self.asr.to(self.asr_dev, self.dtype)

        self.text_fwd, self.audio_fwd = self.text_model, self.audio_model
        self.asr_enc = lambda f, n=None: optim.whisper_encode(self.asr.model.encoder, f, n)
        if o.compile:
            self.text_fwd = torch.compile(self.text_model, dynamic=True)
            self.audio_fwd = torch.compile(self.audio_model, dynamic=True)
            enc_c = torch.compile(lambda f: optim.whisper_encode(self.asr.model.encoder, f), dynamic=True)
            self.asr_enc = lambda f, n=None: enc_c(f if n is None else f[..., :n])
        if o.mask_cache:
            optim.install_cached_suppress_masks()

        ckpt = torch.load(head_path, weights_only=True)
        self.head = FusionHead(ckpt["mode"])
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval()  # tiny; always CPU fp32

    def _threads(self, which: int):
        """In concurrent mode, give each branch thread its own OpenMP team size (set once per thread)."""
        bt = self.opts.branch_threads
        if bt and getattr(self._tls, "set", None) != bt[which]:
            torch.set_num_threads(bt[which])
            self._tls.set = bt[which]

    # ---- branch A -------------------------------------------------------------
    @torch.inference_mode()
    def audio_branch(self, audio: np.ndarray):
        self._threads(0)
        t0 = time.perf_counter()
        x = audio[: self.max_samples]
        if len(x) < 400:
            x = np.pad(x, (0, 400 - len(x)))
        inp = self.fe(x, sampling_rate=SR, return_tensors="pt").input_values.to(self.audio_dev, self.dtype)
        t1 = time.perf_counter()
        hs = torch.stack(self.audio_fwd(inp, output_hidden_states=True).hidden_states, dim=1)
        emb = hs.mean(2)[0].float().cpu()
        _sync(self.audio_dev)
        t2 = time.perf_counter()
        return emb, {"audio_feat": t1 - t0, "audio_enc": t2 - t1}

    # ---- branch B -------------------------------------------------------------
    @torch.inference_mode()
    def text_branch(self, audio: np.ndarray):
        self._threads(1)
        o = self.opts
        t0 = time.perf_counter()
        feats = self.asr_proc.feature_extractor(audio[: 30 * SR], sampling_rate=SR,
                                                return_tensors="pt").input_features.to(self.asr_dev, self.dtype)
        n = (optim.trimmed_frames(len(audio), SR, o.asr_trim_pad_s, o.asr_trim_bucket_s)
             if o.asr_trim_pad_s is not None else None)
        t1 = time.perf_counter()
        enc = self.asr_enc(feats, n)
        _sync(self.asr_dev)
        t2 = time.perf_counter()
        ids = self.asr.generate(input_features=feats, encoder_outputs=enc, max_new_tokens=self.max_new_tokens)
        transcript = self.asr_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        t3 = time.perf_counter()
        emb = self.embed_text(transcript)
        t4 = time.perf_counter()
        return transcript, emb, {"asr_feat": t1 - t0, "asr_enc": t2 - t1, "asr_dec": t3 - t2, "asr": t3 - t1,
                                 "text_enc": t4 - t3, "asr_tokens": int(ids.shape[-1]),
                                 "asr_frames": int(n or feats.shape[-1])}

    @torch.inference_mode()
    def embed_text(self, text: str):
        enc = self.tok([text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        hs = torch.stack(self.text_fwd(**enc, output_hidden_states=True).hidden_states, dim=1)
        m = enc["attention_mask"][:, None, :, None].to(hs.dtype)
        out = ((hs * m).sum(2) / m.sum(2))[0].float().cpu()
        _sync(self.device)
        return out

    # ---- fusion + response ----------------------------------------------------
    @torch.inference_mode()
    def fuse(self, text_emb, audio_emb):
        t0 = time.perf_counter()
        logits = self.head(text_emb[None].float(), audio_emb[None].float())[0]
        label = int(logits.argmax())
        t1 = time.perf_counter()
        return label, torch.softmax(logits, 0).numpy(), {"fusion": t1 - t0}

    def respond(self, label: int, transcript: str):
        t0 = time.perf_counter()
        text = make_response(EMOTIONS[label], transcript)
        return text, {"response": time.perf_counter() - t0}
