# Published MELD results (checked against the papers)

Every number below was read from the paper's own table (PDF text extracted on 2026-11-18), not from a secondary source. All are **7-class emotion, MELD test set, weighted F1 (%)**, using the **human (gold) transcripts**. None of these papers runs speech recognition in the loop.

## Sources

**1. Poria et al., "MELD", ACL 2019** ([arXiv:1810.02508](https://arxiv.org/abs/1810.02508)), Table 11 (test-set weighted F-score, emotion). Text: CNN over word embeddings; audio: openSMILE 6373-dim features + feature selection. bcLSTM and DialogueRNN use conversational context; text-CNN and cMKL do not.

| Model | Text | Audio | Text + audio |
|---|---|---|---|
| text-CNN (no context) | 55.02 | — | — |
| cMKL (no context) | — | — | 55.51 |
| bcLSTM (context) | 56.44 | 39.08 | 59.25 |
| DialogueRNN (context) | 57.03 | 41.79 | 60.25 |

**2. Kim & Vossen, "EmoBERTa", 2021** ([arXiv:2108.12009](https://arxiv.org/abs/2108.12009)), Table 2. Text only, starting from pretrained `roberta-large` with a linear classification layer added and trained on MELD. EmoBERTa rows are **means of 5 random seeds**.

| Model | MELD weighted F1 |
|---|---|
| EmoBERTa, no past or future utterances | 63.46 |
| EmoBERTa, only past utterances | 64.55 |
| EmoBERTa, past and future utterances | 65.61 |
| COSMIC (Ghosal et al., 2020), as reported in this table | 65.21 |
| DialogueGCN (Ghosal et al., 2019), as reported in this table | 58.1 |

**3. "AIMDiT", 2024** ([arXiv:2407.00743](https://arxiv.org/abs/2407.00743)), Table 2 (modality ablation). Text: RoBERTa; audio: mel-spectrogram features (librosa); visual: EfficientNet.

| Modalities | Acc-7 | Weighted F1 |
|---|---|---|
| Text | 62.83 | 60.15 |
| Audio | 48.46 | 32.39 |
| Text + audio | 63.12 | 60.38 |
| Text + audio + visual (full) | 64.83 | 62.33 |

## How this project compares (same test set, n = 2610, gold transcripts)

Ours: frozen `distilroberta-base` + frozen `wav2vec2-base`, mean-pooled, small fusion head, **no conversational context**, model selected on dev, mean ± std over 3 seeds (`docs/experiments.md`, E2):

| | Text | Audio | Text + audio |
|---|---|---|---|
| This project | 60.46 ± 0.21 | 46.38 ± 0.37 | 61.33 ± 0.82 |

What these comparisons support, with care:
- Our text + audio result (61.3) is in the same range as the 2019 MELD baselines with context (59.25–60.25) and AIMDiT's text + audio ablation (60.38).
- It's about 2 points below EmoBERTa without context (63.46), a fine-tuned model several times larger, and about 4 points below the context-using models in EmoBERTa's table (64.55–65.61, COSMIC 65.21).
- Our audio-only result (46.4) is higher than the paper audio baselines above (32.39–41.79). wav2vec2 features are a stronger audio input than openSMILE or mel spectrograms. But the models, training and selection all differ, so this is a weak comparison.

What they do **not** support:
- Any claim about the live pipeline. Our live weighted F1 (0.53, Whisper transcripts, energy segmentation) has no published counterpart above, because all of these use human transcripts.
- A strict ranking. Implementations, seeds and selection differ between papers. A GitHub issue on the MELD repo reports trouble reproducing the bcLSTM numbers ([declare-lab/meld#10](https://github.com/declare-lab/meld/issues/10)). Some papers report maxima, others means.

Not verified, so not quoted: higher numbers in recent papers (for example, a 2024 missing-modality paper's abstract reports 71.98 weighted F1). I haven't checked their setups.
