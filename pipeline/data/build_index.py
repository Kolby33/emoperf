"""Join MELD label CSVs with the audio manifests into one index per split.

Output: data/index/{split}.csv with one row per utterance that has both a label and audio:
  split, dialogue_id, utterance_id, utt_key, text, speaker, emotion, label, wav, duration_s

Prints class counts and the majority-class baseline, and reports every utterance
dropped (label without audio, or audio without label) so the dataset size is explainable.

Usage:
  python pipeline/data/build_index.py --csv-dir data/raw/extracted --audio-dir data/audio --out-dir data/index
"""

import argparse
from pathlib import Path

import pandas as pd

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
CSV_NAMES = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}


def srt_seconds(s: str) -> float:
    """'00:16:16,059' -> 976.059"""
    hms, ms = s.strip().split(",")
    h, m, sec = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True, type=Path)
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-duration", type=float, default=0.1,
                    help="drop clips shorter than this many seconds (too short to encode)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, name in CSV_NAMES.items():
        labels = pd.read_csv(args.csv_dir / name, encoding="utf-8")
        labels = labels.rename(columns={"Dialogue_ID": "dialogue_id", "Utterance_ID": "utterance_id",
                                        "Utterance": "text", "Speaker": "speaker", "Emotion": "emotion"})
        audio = pd.read_csv(args.audio_dir / f"{split}_manifest.csv")
        audio = audio[audio.status != "failed"]

        merged = labels.merge(audio[["dialogue_id", "utterance_id", "wav", "duration_s"]],
                              on=["dialogue_id", "utterance_id"], how="outer", indicator=True)
        no_audio = merged[merged._merge == "left_only"]
        no_label = merged[merged._merge == "right_only"]
        both = merged[merged._merge == "both"].copy()
        too_short = both[both.duration_s < args.min_duration]
        both = both[both.duration_s >= args.min_duration]

        # Data-quality check: does each audio file's length match the label's StartTime/EndTime?
        label_dur = both.EndTime.map(srt_seconds) - both.StartTime.map(srt_seconds)
        dur_err = (both.duration_s - label_dur).abs()

        both["emotion"] = both.emotion.str.lower().str.strip()
        unknown = sorted(set(both.emotion) - set(EMOTIONS))
        assert not unknown, f"unexpected emotion labels: {unknown}"
        both["label"] = both.emotion.map(EMOTIONS.index)
        both["split"] = split
        both["utt_key"] = "dia" + both.dialogue_id.astype(str) + "_utt" + both.utterance_id.astype(str)
        cols = ["split", "dialogue_id", "utterance_id", "utt_key", "text", "speaker", "emotion", "label",
                "wav", "duration_s"]
        both = both[cols].sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
        both.to_csv(args.out_dir / f"{split}.csv", index=False)

        counts = both.emotion.value_counts()
        print(f"\n== {split}: {len(both)} utterances kept | dropped: {len(no_audio)} label-without-audio, "
              f"{len(no_label)} audio-without-label, {len(too_short)} shorter than {args.min_duration}s")
        if len(no_audio):
            print("   label-without-audio examples:",
                  [f"dia{r.dialogue_id}_utt{r.utterance_id}" for r in no_audio.head(5).itertuples()])
        print(f"   audio: {both.duration_s.sum() / 3600:.2f} h, median {both.duration_s.median():.2f} s, "
              f"p95 {both.duration_s.quantile(0.95):.2f} s, max {both.duration_s.max():.2f} s")
        print(f"   |audio length - label length|: median {dur_err.median():.2f} s, "
              f"{(dur_err > 1.0).mean():.1%} of clips off by > 1 s")
        for emo in EMOTIONS:
            c = int(counts.get(emo, 0))
            print(f"   {emo:>9}: {c:5d}  ({c / len(both):5.1%})")
        print(f"   majority class '{counts.index[0]}' = accuracy {counts.iloc[0] / len(both):.1%}")


if __name__ == "__main__":
    main()
