"""Live demo: speak into the Mac's microphone, get an emotion tag and a response per utterance.

    python demo.py --web                # microphone + live web page at http://127.0.0.1:8765
    python demo.py                      # microphone, terminal output only
    python demo.py --web --source meld --n 12   # no microphone: replay MELD test clips in real time
    python demo.py --source file --file some.wav

Everything runs locally and offline: no audio or text leaves the machine. Stop with Ctrl+C.
See docs/DEMO.md.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are cached; never call the network
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import webbrowser  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

from pipeline.live.session import LiveSession, SessionConfig  # noqa: E402


def print_event(ev):
    t = ev["type"]
    if t == "status":
        print(ev["message"], flush=True)
    elif t == "calibrated":
        print(f"Room noise {ev['floor_db']:.0f} dBFS -> speech threshold {ev['threshold_db']:.0f} dBFS", flush=True)
    elif t == "utterance":
        second = sorted(ev["probs"].items(), key=lambda kv: -kv[1])[1]
        mark = f"   true: {ev['true']} {'(match)' if ev['true'] == ev['emotion'] else '(miss)'}" if "true" in ev else ""
        print(f"\n[{ev['start_s']:6.1f}s, {ev['duration_s']:.1f}s] \"{ev['transcript']}\"\n"
              f"    -> {ev['emotion'].upper()} {ev['prob']:.0%} (next: {second[0]} {second[1]:.0%}){mark}\n"
              f"    {ev['response']}\n"
              f"    latency {ev['e2e_ms']:.0f} ms after you stopped (waiting for silence {ev['endpoint_ms']:.0f} ms, "
              f"models {ev['inference_ms']:.0f} ms)", flush=True)
    elif t == "summary" and ev["n"]:
        print(f"\n--- {ev['n']} utterances | latency p50 {ev['e2e_p50_ms']:.0f} ms, p95 {ev['e2e_p95_ms']:.0f} ms "
              f"| models p50 {ev['inference_p50_ms']:.0f} ms")
        if "scored" in ev:
            print(f"--- emotion matched the MELD label in {ev['matched']}/{ev['scored']} utterances")
        if ev["mic_overflows"]:
            print(f"--- warning: the microphone buffer overflowed {ev['mic_overflows']} times (audio was lost)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["mic", "meld", "file"], default="mic")
    ap.add_argument("--file", type=Path, help="16 kHz mono WAV for --source file")
    ap.add_argument("--n", type=int, default=8, help="MELD test clips for --source meld")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--head", type=Path, default=Path("models/fusion_text_audio_trainboth_seed0.pt"))
    ap.add_argument("--hangover", type=float, default=0.2)
    ap.add_argument("--margin-db", type=float, default=12.0, help="speech must be this far above the room noise")
    ap.add_argument("--calibrate-s", type=float, default=1.0)
    ap.add_argument("--web", action="store_true", help="also show a live page at http://127.0.0.1:PORT")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="with --web: don't open the browser automatically")
    ap.add_argument("--save", action="store_true", help="save the session (incl. transcripts) to data/runs/")
    args = ap.parse_args()

    session = LiveSession(SessionConfig(source=args.source, file=args.file, n=args.n, device=args.device,
                                        head=args.head, hangover=args.hangover, margin_db=args.margin_db,
                                        calibrate_s=args.calibrate_s))
    session.on(print_event)
    if args.web:
        from pipeline.live.web_server import EventHub, serve
        hub = EventHub()
        session.on(hub.publish)
        serve(hub, args.port)
        url = f"http://127.0.0.1:{args.port}"
        print(f"Web page: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)

    session.load()
    signal.signal(signal.SIGINT, lambda *_: session.stop())
    session.start()
    session.wait()

    if args.save and session.results:
        out = Path("data/runs") / f"demo_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                   "results": session.results}, indent=1))
        print(f"--- saved to {out} (git-ignored)")
    if args.web:  # keep the finished session viewable (and reloadable) until Ctrl+C
        import threading
        done = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: done.set())
        print(f"Session finished. The page stays up at http://127.0.0.1:{args.port} until Ctrl+C.", flush=True)
        while not done.is_set():
            time.sleep(0.2)


if __name__ == "__main__":
    main()
