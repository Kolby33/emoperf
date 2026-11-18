"""Build the simulated live audio stream from MELD test clips.

Clips are concatenated in a fixed order with gaps of low-level white noise between
them (default 0.8 s at -75 dBFS, seeded), so the result is one long 16 kHz signal plus
a ground-truth table saying where each utterance starts and ends in it.

Why noise and not digital zeros: real input is never exactly silent, and log-energy
of exact zeros is a degenerate case. -75 dBFS is far below the -55 dBFS speech
threshold, so the gaps are an easy case. That limitation is documented.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000


@dataclass
class Stream:
    audio: np.ndarray
    gt: pd.DataFrame  # utt_key, label, emotion, text, start, end (absolute samples)

    @property
    def seconds(self):
        return len(self.audio) / SR


def build_stream(index_csv, subset_file, n=None, gap_s=0.8, gap_db=-75.0, seed=0) -> Stream:
    idx = pd.read_csv(index_csv, keep_default_na=False).set_index("utt_key")
    keys = subset_file.read_text().split()
    if n is not None:
        keys = keys[:n]
    rng = np.random.default_rng(seed)
    gap_len = int(gap_s * SR)
    noise = lambda: (rng.standard_normal(gap_len) * 10 ** (gap_db / 20)).astype(np.float32)

    parts, rows, pos = [noise()], [], gap_len
    for k in keys:
        x, sr = sf.read(idx.loc[k, "wav"], dtype="float32")
        assert sr == SR
        rows.append({"utt_key": k, "label": int(idx.loc[k, "label"]), "emotion": idx.loc[k, "emotion"],
                     "text": idx.loc[k, "text"], "start": pos, "end": pos + len(x)})
        parts += [x, noise()]
        pos += len(x) + gap_len
    return Stream(np.concatenate(parts), pd.DataFrame(rows))
