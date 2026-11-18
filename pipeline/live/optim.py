"""Optimizations tested in M4. Each is opt-in through StageOptions; defaults reproduce M2.

1. Cached suppress-token masks (M3 hotspot 2).
   Hugging Face's Whisper logits processors rebuild a full-vocabulary boolean mask with
   torch.isin (which sorts the 51,864-entry vocabulary) on every decode step. The mask
   never changes, so compute it once per processor and reuse it. Same math, same tokens.

2. Trimmed Whisper encoder input (M3 hotspot 1).
   Whisper pads every clip to 30 s (3000 mel frames -> 1500 encoder positions). This runs
   the same encoder layers on only the first `n_frames` frames: the real audio plus a
   margin, rounded up to a bucket. Not mathematically identical (attention sees fewer
   padding positions, and Whisper was trained on 30 s windows), so it is checked on WER/F1.
"""

import math

import torch
from torch import nn
from transformers.generation import logits_process as lp
from transformers.modeling_outputs import BaseModelOutput

_ORIG = {}


def install_cached_suppress_masks():
    """Patch the two processors so the vocab mask is built once per (processor, vocab, device)."""
    if _ORIG:
        return

    def _mask(self, attr, scores):
        key = (scores.shape[-1], scores.device)
        cache = self.__dict__.setdefault("_mask_cache", {})
        if key not in cache:
            vocab = torch.arange(scores.shape[-1], device=scores.device)
            cache[key] = torch.isin(vocab, getattr(self, attr).to(scores.device))
        return cache[key]

    def suppress_call(self, input_ids, scores):
        return torch.where(_mask(self, "suppress_tokens", scores), -float("inf"), scores)

    def begin_call(self, input_ids, scores):
        if input_ids.shape[-1] != self.begin_index:  # original builds the mask, then ignores it here
            return scores
        return torch.where(_mask(self, "begin_suppress_tokens", scores), -float("inf"), scores)

    _ORIG["suppress"] = lp.SuppressTokensLogitsProcessor.__call__
    _ORIG["begin"] = lp.SuppressTokensAtBeginLogitsProcessor.__call__
    lp.SuppressTokensLogitsProcessor.__call__ = suppress_call
    lp.SuppressTokensAtBeginLogitsProcessor.__call__ = begin_call


def uninstall_cached_suppress_masks():
    if _ORIG:
        lp.SuppressTokensLogitsProcessor.__call__ = _ORIG.pop("suppress")
        lp.SuppressTokensAtBeginLogitsProcessor.__call__ = _ORIG.pop("begin")


def trimmed_frames(n_samples: int, sr: int, pad_s: float, bucket_s: float, max_frames: int = 3000) -> int:
    """Mel frames to keep: (audio + pad) rounded up to a bucket, even, at most 3000 (10 ms per frame)."""
    seconds = n_samples / sr + pad_s
    frames = math.ceil(seconds / bucket_s) * round(bucket_s * 100)
    frames = min(max_frames, frames)
    return frames + (frames % 2)


@torch.inference_mode()
def whisper_encode(encoder, feats: torch.Tensor, n_frames: int | None = None) -> BaseModelOutput:
    """Same computation as WhisperEncoder.forward (eval mode), without the 3000-frame check."""
    if n_frames is not None:
        feats = feats[..., :n_frames]
    x = nn.functional.gelu(encoder.conv1(feats))
    x = nn.functional.gelu(encoder.conv2(x)).permute(0, 2, 1)
    x = x + encoder.embed_positions.weight[: x.shape[1]]
    for layer in encoder.layers:
        x = layer(x, None)
    return BaseModelOutput(last_hidden_state=encoder.layer_norm(x))
