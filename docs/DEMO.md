# Live demo

```bash
.venv/bin/python demo.py --web
```

This opens a local page at http://127.0.0.1:8765 (only reachable from this Mac). The emotion of your latest sentence is the large word at the top. Below it, a **voice strip** scrolls through the last 15 seconds (9 on a phone): each sentence's bars and underline take its emotion's color, and an amber gap marks the time from when you stopped talking to when the answer appeared, labeled in ms. Under that: all 7 probabilities, a latency chart that splits each sentence into *waiting to be sure you stopped* and *model time*, and a history. Fonts (Fraunces, IBM Plex Sans; SIL Open Font License, files in `pipeline/live/web/fonts/`) are bundled so the page works offline. Without `--web` the same information prints in the terminal. In web mode the page stays up after the session ends, until Ctrl+C.

1. The models load (about 10 s), then the demo asks you to **stay quiet for 1 second** while it measures the room noise. It prints the noise level and the resulting speech threshold.
2. Speak a sentence, then pause. About 0.35 s after you stop, it prints what it heard, the emotion (with the runner-up), a response, and the latency.
3. Ctrl+C prints a summary (number of utterances, latency p50 / p95). Add `--save` to keep the session in `data/runs/` (git-ignored).

The first time, macOS asks whether the app running Python (Terminal, or the Claude app's terminal) may use the microphone. If you said no earlier, turn it on in System Settings → Privacy & Security → Microphone.

**Without a microphone:** `--web --source meld --n 14` replays 8 MELD test clips in real time and prints the true label next to each prediction. `--source file --file x.wav` plays a 16 kHz mono WAV.

**Useful flags:** `--margin-db 15` if background noise triggers false utterances (default 12), `--hangover 0.3` if it cuts you off mid-sentence (default 0.2 s), `--device N` for another microphone (list them with `python -c "import sounddevice; print(sounddevice.query_devices())"`).

**Privacy:** everything runs locally. The demo sets `HF_HUB_OFFLINE=1`, so it doesn't contact the network, and audio is never written to disk unless you pass `--save`, which saves only the text and timings.

## What it runs

The M4 best configuration: 20 ms microphone blocks → ring buffer → energy segmenter with an **adaptive noise floor** (10th percentile of the last 3 s of frame energies; speech = 12 dB above it) → Whisper base.en on the GPU (MPS) → text encoder, **in parallel with** wav2vec2 on the CPU → fusion head trained on gold + Whisper text (`fusion_text_audio_trainboth_seed0.pt`, E7) → response template.

## Measured (MELD replay, 8 clips, 2026-11-18)

Latency after speech ended: p50 339 ms, p95 356 ms (200 ms waiting for silence + about 140 ms of model time). Emotion matched the MELD label for 4 of 8 clips. Microphone input hasn't been measured by me; it needs you speaking. Expect Whisper to do better on clean close-up speech than on TV audio, but that hasn't been measured either.

**Known limits:** English only (Whisper base.en). Short replies ("yeah", "ok") are often tagged neutral. Sarcasm usually reads as its literal meaning. Accuracy on MELD is about 53–56% live (see `docs/experiments.md`, E7, and `docs/related_results.md`).

## Troubleshooting: "Room noise -120 dBFS" / no utterances detected

-120 dBFS is digital silence: every sample is exactly 0. On macOS that means the app the terminal runs in has **no microphone permission**. macOS doesn't raise an error; it delivers zeros. The demo now detects this and says so (terminal message and a red banner on the page).

This happened on 2026-11-18 when running from the Claude desktop app's terminal pane: macOS attributes the request to `Claude.app`, which didn't have microphone access, while the same Python code run elsewhere got real audio (-37 dBFS). Fixes (you change these, not the code):
- Run the demo from the **Terminal** app (Applications → Utilities → Terminal). macOS asks for microphone access the first time; choose Allow.
- Or enable the app in **System Settings → Privacy & Security → Microphone**, then restart the demo.
