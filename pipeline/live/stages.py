"""Inference stages of the live pipeline, each timed separately.

  branch A (audio):  audio_feat -> audio_enc                      -> [13, 768]
  branch B (text):   asr_feat -> asr (Whisper generate) -> text_enc -> transcript, [7, 768]
  fusion:            fusion head -> emotion
  response:          deterministic template from emotion + transcript snippet

Preprocessing matches pipeline/embed/cache_embeddings.py exactly (same models, same
pooling), so the fusion head sees the same kind of input it was trained on. The only
intended difference is that the text comes from ASR instead of the gold transcript.
"""

import time
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoFeatureExtractor, AutoModel, AutoTokenizer, WhisperForConditionalGeneration,
                          WhisperProcessor)

from pipeline.fusion.model import EMOTIONS, FusionHead

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


def make_response(emotion: str, transcript: str, max_words: int = 8) -> str:
    words = transcript.strip().split()
    snippet = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
    quote = f' You said: "{snippet}"' if snippet else ""
    return f"[{emotion}] {RESPONSES[emotion]}{quote}"


class Stages:
    def __init__(self, head_path: Path, device: str = "cpu", asr_model: str = ASR_MODEL,
                 max_audio_seconds: float = 20.0, max_new_tokens: int = 96):
        self.device = torch.device(device)
        self.max_samples = int(max_audio_seconds * SR)
        self.max_new_tokens = max_new_tokens
        self.asr_model_name = asr_model

        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.text_model = AutoModel.from_pretrained(TEXT_MODEL).eval().to(self.device)
        self.fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
        self.audio_model = AutoModel.from_pretrained(AUDIO_MODEL).eval().to(self.device)
        self.asr_proc = WhisperProcessor.from_pretrained(asr_model)
        self.asr = WhisperForConditionalGeneration.from_pretrained(asr_model).eval().to(self.device)

        ckpt = torch.load(head_path, weights_only=True)
        self.head = FusionHead(ckpt["mode"])
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval().to(self.device)

    # ---- branch A -------------------------------------------------------------
    @torch.inference_mode()
    def audio_branch(self, audio: np.ndarray):
        t0 = time.perf_counter()
        x = audio[: self.max_samples]
        if len(x) < 400:
            x = np.pad(x, (0, 400 - len(x)))
        inp = self.fe(x, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        t1 = time.perf_counter()
        hs = torch.stack(self.audio_model(inp, output_hidden_states=True).hidden_states, dim=1)
        emb = hs.mean(2)[0]
        t2 = time.perf_counter()
        return emb, {"audio_feat": t1 - t0, "audio_enc": t2 - t1}

    # ---- branch B -------------------------------------------------------------
    @torch.inference_mode()
    def text_branch(self, audio: np.ndarray):
        t0 = time.perf_counter()
        feats = self.asr_proc.feature_extractor(audio[: 30 * SR], sampling_rate=SR,
                                                return_tensors="pt").input_features.to(self.device)
        t1 = time.perf_counter()
        ids = self.asr.generate(feats, max_new_tokens=self.max_new_tokens)
        transcript = self.asr_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        t2 = time.perf_counter()
        emb = self.embed_text(transcript)
        t3 = time.perf_counter()
        return transcript, emb, {"asr_feat": t1 - t0, "asr": t2 - t1, "text_enc": t3 - t2,
                                 "asr_tokens": int(ids.shape[-1])}

    @torch.inference_mode()
    def embed_text(self, text: str):
        enc = self.tok([text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        hs = torch.stack(self.text_model(**enc, output_hidden_states=True).hidden_states, dim=1)
        m = enc["attention_mask"][:, None, :, None].to(hs.dtype)
        return ((hs * m).sum(2) / m.sum(2))[0]

    # ---- fusion + response ----------------------------------------------------
    @torch.inference_mode()
    def fuse(self, text_emb, audio_emb):
        t0 = time.perf_counter()
        logits = self.head(text_emb[None].float(), audio_emb[None].float())[0]
        label = int(logits.argmax())
        t1 = time.perf_counter()
        return label, torch.softmax(logits, 0).cpu().numpy(), {"fusion": t1 - t0}

    def respond(self, label: int, transcript: str):
        t0 = time.perf_counter()
        text = make_response(EMOTIONS[label], transcript)
        return text, {"response": time.perf_counter() - t0}
