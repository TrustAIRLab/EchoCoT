"""Loopback-only HTTP/SSE server. Model calls stay in one persistent session."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from config import DEFAULT_TARGET_MODEL, LENGTH_ERROR_THRESHOLD, TARGET_MODEL_OPTIONS, TOKEN_EM_THRESHOLD
from extractors import ARCHIVE_SUFFIX, FINAL_ANSWER_INJECTION, load_target, make_session
from injection_io import load_injections

from .records import response_view

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name("static")
EXAMPLES = Path(__file__).with_name("examples")


def saved_injections(key):
    path = ROOT / "injections" / f"{key}_expereince.md"
    return load_injections(path) if path.is_file() else [
        {"turn_idx": 1, "injection": ""}, {"turn_idx": 2, "injection": ""},
    ]


def catalog(datasets):
    models = []
    for key, config in TARGET_MODEL_OPTIONS.items():
        if key in ("o4-mini", "deepseek-v4-flash-20261017"):
            continue
        length_only = config["extractor"] in ("extractors.extract_gemini", "extractors.extract_claude")
        model = {
            "key": key, "model": config["model"], "notes": config["notes"],
            "length_only": length_only, "length_error_threshold": LENGTH_ERROR_THRESHOLD,
        }
        if not length_only:
            model["token_em_threshold"] = TOKEN_EM_THRESHOLD
        models.append(model)
    dataset_names = sorted({name for model_datasets in datasets.values() for name in model_datasets})
    return {"models": models, "datasets": [{"name": name} for name in dataset_names], "default_model": DEFAULT_TARGET_MODEL,
            "final_instruction": FINAL_ANSWER_INJECTION}


def json_bytes(value):
    # Missing provider token counts can produce NaN; browsers need valid JSON.
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, list):
            return [clean(v) for v in item]
        return item
    return json.dumps(clean(value), ensure_ascii=False).encode("utf-8")


class DemoUI:
    def __init__(self, data_path=EXAMPLES):
        # The dataset supplies only a question, never precomputed responses or metrics.
        self.datasets = {
            model.name: {
                p.parent.name: [{"sample_id": str(item["sample_id"]), "question": item["question"]}
                                for item in json.loads(p.read_text(encoding="utf-8"))]
                for p in sorted(model.glob("*/test.json"))
            }
            for model in sorted(Path(data_path).iterdir()) if model.is_dir()
        }
        self.catalog = catalog(self.datasets)
        self.condition = threading.Condition(threading.RLock())
        self.revision = 0
        self.busy = False
        self.config = None
        self.question = ""
        self.sample_id = ""
        self.phase = "ready"
        self.events = []
        self.sent_injections = 0
        self.response = None
        self.error = None
        self.session = None
        self.extractor = None

    def _changed(self):
        self.revision += 1
        self.condition.notify_all()

    def _next_action(self):
        if not self.config or self.busy:
            return None
        if self.phase in ("complete", "error"):
            return None
        if self.phase == "ready":
            return "start"
        return "inject" if self.sent_injections < len(self.config["injections"]) else "final"

    def snapshot(self):
        with self.condition:
            return copy.deepcopy({
                "revision": self.revision, "busy": self.busy, "config": self.config,
                "question": self.question, "sample_id": self.sample_id,
                "phase": self.phase, "events": self.events, "response": self.response,
                "error": self.error, "sent_injections": self.sent_injections,
                "next_action": self._next_action(),
            })

    def configure(self, data):
        with self.condition:
            if self.busy:
                raise ValueError("Wait for the current model response before changing the selection.")
            model = next((m for m in self.catalog["models"] if m["key"] == data["model"]), None)
            if not model:
                raise ValueError("Choose a configured target model.")
            dataset = data["dataset"]
            model_datasets = self.datasets[model["key"]]
            if dataset not in model_datasets:
                raise ValueError("Choose an existing test dataset.")
            samples = model_datasets[dataset]
            index = int(data.get("sample_idx", 0))
            if not 0 <= index < len(samples):
                raise ValueError("Sample index is outside this dataset.")
            sample = samples[index]
            self.config = {**model, "id": uuid4().hex, "dataset": dataset, "sample_idx": index,
                           "sample_count": len(samples),
                           "injections": saved_injections(model["key"]),
                           "final_instruction": FINAL_ANSWER_INJECTION}
            self.question, self.sample_id = sample["question"], sample["sample_id"]
            self.phase, self.events, self.sent_injections = "ready", [], 0
            self.response, self.error, self.session, self.extractor = None, None, None, None
            self._changed()
            return self.snapshot()

    def act(self, action, revision, injection=None):
        with self.condition:
            if self.busy or revision != self.revision or action != self._next_action() or action is None:
                raise ValueError("This step is not available. Wait for the current state to update.")
            if action == "start":
                text, label = self.question + ARCHIVE_SUFFIX, "Initial response"
            elif action == "inject":
                saved = self.config["injections"][self.sent_injections]
                text = saved["injection"] if injection is None else injection
                label = f"Turn {saved['turn_idx']} · Injection"
            else:
                text = self.config["final_instruction"] if injection is None else injection
                label = f"Turn {len(self.config['injections']) + 1} · Final instruction"
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Enter an injection before sending this turn.")
            event = {"action": action, "label": label, "input": text, "status": "running",
                     "started_at": time.time(), "elapsed": None, "turns": [], "answer": ""}
            self.events.append(event)
            self.phase, self.busy = "running", True
            self._changed()
            threading.Thread(target=self._execute, args=(action, event), daemon=True).start()
            return self.snapshot()

    def _execute(self, action, event):
        try:
            if action == "start":
                if self.extractor is None:
                    self.extractor = load_target(self.config["key"])
                self.session = make_session(event["input"], self.extractor)
                self.session.retain_reasoning_context = True
                self.session.skip_cot_summary = True
                self.session.sample_id = self.sample_id
            previous_turns = len(self.session.trajectory.turns)
            if action == "start":
                self.session.start()
            elif action == "inject":
                self.session.send(event["input"])
            else:
                self.session.final_send(event["input"])
            trajectory = self.session.trajectory
            if self.session.done and trajectory.turns:
                trajectory.turns[-1].stopped = True
            response = response_view(trajectory, self.config["length_only"])
            with self.condition:
                event["status"], event["answer"] = "complete", trajectory.final_output
                event["turns"] = [t.turn_idx for t in trajectory.turns[previous_turns:]]
                self.response = response
                if action == "inject":
                    self.sent_injections += 1
                self.phase = "complete" if self.session.done else "waiting"
        except Exception as exc:
            traceback.print_exc()
            message = f"{type(exc).__name__}: {exc}"
            for key, value in os.environ.items():
                if value and any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                    message = message.replace(value, "[redacted]")
            with self.condition:
                self.phase, self.error = "error", message
                event["status"] = "error"
        finally:
            with self.condition:
                event["elapsed"] = round(time.time() - event["started_at"], 2)
                self.busy = False
                self._changed()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _reply(self, status, body, content_type="application/json; charset=utf-8"):
        body = json_bytes(body) if not isinstance(body, bytes) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _local_request(self):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in allowed and (not origin or origin == f"http://{host}")

    def do_GET(self):
        if not self._local_request():
            return self._reply(403, {"error": "Use the local server URL."})
        path = urlparse(self.path).path
        app = self.server.app
        if path == "/api/catalog":
            return self._reply(200, app.catalog)
        if path == "/api/state":
            return self._reply(200, app.snapshot())
        if path == "/api/events":
            return self._events()
        assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                  "/style.css": ("style.css", "text/css")}
        if path in assets:
            filename, mime = assets[path]
            return self._reply(200, (STATIC / filename).read_bytes(), mime + "; charset=utf-8")
        self._reply(404, {"error": "Not found"})

    def do_POST(self):
        if not self._local_request() or self.headers.get_content_type() != "application/json":
            return self._reply(403, {"error": "Use the local interface with JSON requests."})
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path == "/api/configure":
                state = self.server.app.configure(data)
            elif self.path == "/api/action":
                state = self.server.app.act(data["action"], data["revision"], data.get("injection"))
            else:
                return self._reply(404, {"error": "Not found"})
            self._reply(200, state)
        except (ValueError, KeyError, TypeError) as exc:
            self._reply(409, {"error": str(exc)})

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        app, seen = self.server.app, -1
        try:
            while True:
                with app.condition:
                    if seen == app.revision:
                        app.condition.wait(timeout=15)
                    state = app.snapshot() if seen != app.revision else None
                if state is None:
                    payload = b": keep-alive\n\n"
                else:
                    seen = state["revision"]
                    payload = b"event: state\ndata: " + json_bytes(state) + b"\n\n"
                self.wfile.write(payload)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(app, port):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.app = app
    print(f"EchoCoT interactive demo: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Explore one question and its injection process in a browser.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(DemoUI(), args.port)
