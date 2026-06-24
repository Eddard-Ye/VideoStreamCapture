# -*- coding: utf-8 -*-
"""Minimal HTTP server: MJPEG stream + water-cut control API."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


@dataclass
class CaptureRequest:
    name: str
    height: str
    temperature: str
    weight: str
    water_cut: bool
    event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] = field(default_factory=dict)


class StreamHub:
    """Thread-safe shared state between capture loop and HTTP handlers."""

    def __init__(self, target_fps: float = 10.0):
        self.target_fps = max(0.1, float(target_fps))
        self._lock = threading.Lock()
        self._jpeg = b""
        self._status: dict[str, Any] = {"state": "starting", "fps": 0.0, "instances": 0}
        self._water_cut_request = threading.Event()
        self._clear_water_cut_request = threading.Event()
        self._frame_ready = threading.Event()
        self._capture_lock = threading.Lock()
        self._pending_capture: CaptureRequest | None = None
        self.computing_water_cut = False

    def set_frame(self, jpeg: bytes, status: dict[str, Any]) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._status = status
        self._frame_ready.set()

    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def request_water_cut(self) -> None:
        self._water_cut_request.set()

    def consume_water_cut_request(self) -> bool:
        if not self._water_cut_request.is_set():
            return False
        self._water_cut_request.clear()
        return True

    def request_clear_water_cut(self) -> None:
        self._clear_water_cut_request.set()

    def consume_clear_water_cut_request(self) -> bool:
        if not self._clear_water_cut_request.is_set():
            return False
        self._clear_water_cut_request.clear()
        return True

    def wait_for_frame(self, timeout: float) -> bool:
        ok = self._frame_ready.wait(timeout=timeout)
        self._frame_ready.clear()
        return ok

    def submit_capture(self, request: CaptureRequest) -> CaptureRequest | None:
        with self._capture_lock:
            if self._pending_capture is not None:
                return None
            self._pending_capture = request
            return request

    def consume_capture_request(self) -> CaptureRequest | None:
        with self._capture_lock:
            request = self._pending_capture
            self._pending_capture = None
            return request

    def finish_capture(self, request: CaptureRequest) -> None:
        request.event.set()


def resolve_capture_file(capture_dir: str, filename: str) -> Path | None:
    """Resolve a capture filename safely inside capture_dir."""
    name = unquote(filename).strip()
    if not name or name != os.path.basename(name):
        return None
    if name.startswith(".") or ".." in name:
        return None

    base = Path(capture_dir).resolve()
    target = (base / name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    if not target.is_file():
        return None
    return target


def make_handler(hub: StreamHub, capture_output_dir: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/help"):
                text = (
                    "Endpoints:\n"
                    "  GET  /video          MJPEG stream\n"
                    "  GET  /snapshot.jpg   latest JPEG frame\n"
                    "  GET  /status         JSON status\n"
                    "  POST /water-cut      run water-cut on current frame\n"
                    "  POST /water-cut/clear clear water-cut overlay\n"
                    "  POST /capture        capture annotated frame to disk\n"
                    "  GET  /captures/{fileName}  saved capture JPEG\n"
                )
                self._send_bytes(200, text.encode("utf-8"), "text/plain; charset=utf-8")
                return

            if path == "/video":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Connection", "close")
                self.end_headers()
                interval = 1.0 / hub.target_fps
                try:
                    while True:
                        if not hub.wait_for_frame(timeout=max(interval, 0.5)):
                            jpeg = hub.get_jpeg()
                        else:
                            jpeg = hub.get_jpeg()
                        if not jpeg:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(interval)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                except OSError:
                    return

            if path == "/snapshot.jpg":
                jpeg = hub.get_jpeg()
                if not jpeg:
                    self._send_json(503, {"ok": False, "error": "no frame yet"})
                    return
                self._send_bytes(200, jpeg, "image/jpeg")
                return

            if path == "/status":
                self._send_json(200, {"ok": True, **hub.get_status()})
                return

            if path.startswith("/captures/"):
                filename = path[len("/captures/") :]
                capture_path = resolve_capture_file(capture_output_dir, filename)
                if capture_path is None:
                    self._send_json(404, {"ok": False, "error": "capture not found"})
                    return
                body = capture_path.read_bytes()
                self._send_bytes(200, body, "image/jpeg")
                return

            self._send_json(404, {"ok": False, "error": "not found"})

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/water-cut":
                hub.request_water_cut()
                self._send_json(202, {"ok": True, "message": "water-cut queued"})
                return
            if path == "/water-cut/clear":
                hub.request_clear_water_cut()
                self._send_json(200, {"ok": True, "message": "water-cut cleared"})
                return
            if path == "/capture":
                try:
                    payload = self._read_json_body()
                except (json.JSONDecodeError, ValueError) as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return

                name = payload.get("name")
                if not isinstance(name, str) or not name.strip():
                    self._send_json(400, {"ok": False, "error": "name is required"})
                    return

                def _field(key: str) -> str:
                    value = payload.get(key, "")
                    return "" if value is None else str(value)

                water_cut_raw = payload.get("water_cut", False)
                if isinstance(water_cut_raw, str):
                    water_cut = water_cut_raw.strip().lower() in ("1", "true", "yes", "on")
                else:
                    water_cut = bool(water_cut_raw)

                request = CaptureRequest(
                    name=name.strip(),
                    height=_field("height"),
                    temperature=_field("temperature"),
                    weight=_field("weight"),
                    water_cut=water_cut,
                )
                if hub.submit_capture(request) is None:
                    self._send_json(409, {"ok": False, "error": "capture already in progress"})
                    return

                timeout_s = 120.0
                if not request.event.wait(timeout=timeout_s):
                    with hub._capture_lock:
                        if hub._pending_capture is request:
                            hub._pending_capture = None
                    self._send_json(504, {"ok": False, "error": "capture timed out"})
                    return

                result = dict(request.result)
                code = 200 if result.get("ok") else 500
                self._send_json(code, result)
                return
            self._send_json(404, {"ok": False, "error": "not found"})

    return Handler


def start_stream_server(
    hub: StreamHub,
    host: str,
    port: int,
    *,
    capture_output_dir: str = "output/captures",
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(hub, capture_output_dir))
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="stream-http")
    thread.start()
    return server
