"""Tiny local web server for the live demo (standard library only, no extra packages).

  GET /         the page (pipeline/live/web/index.html)
  GET /events   Server-Sent Events: one JSON event per message (see session.py)

Binds to 127.0.0.1 only, so nothing on the network can reach it. A browser that connects
late (or reloads) first receives the history so far, then live events.
"""

import json
import queue
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).parent / "web" / "index.html"
FONTS = Path(__file__).parent / "web" / "fonts"  # Fraunces + IBM Plex Sans (SIL OFL), served locally so the page works offline


class EventHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.subscribers = []
        self.history = []        # non-level events, replayed to new clients
        self.recent_levels = deque(maxlen=400)  # last 20 s at 20/s, so a reloaded page can redraw the voice strip

    def publish(self, ev):
        with self.lock:
            if ev["type"] == "level":
                self.recent_levels.append(ev)
            else:
                self.history.append(ev)
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:  # a stalled browser tab only loses its own events
                pass

    def subscribe(self):
        q = queue.Queue(maxsize=1000)
        with self.lock:
            for ev in self.history:
                q.put_nowait(ev)
            for ev in self.recent_levels:
                q.put_nowait(ev)
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


def make_handler(hub: EventHub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the terminal clean
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = hub.subscribe()
                try:
                    while True:
                        try:
                            ev = q.get(timeout=15)
                            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    hub.unsubscribe(q)
            elif self.path.startswith("/fonts/"):
                name = self.path[len("/fonts/"):]
                font = FONTS / name
                # only plain file names from the fonts folder (no "..", no subfolders)
                if "/" in name or not name.endswith(".woff2") or not font.is_file():
                    self.send_error(404)
                    return
                body = font.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    return Handler


def serve(hub: EventHub, port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(hub))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
