# -*- coding: utf-8 -*-
"""CLI client for capture_2d_stream control API."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _request(method: str, url: str, timeout: float = 30.0, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc

    if not body.strip():
        return {"ok": True}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"ok": True, "raw": body}


def main() -> int:
    parser = argparse.ArgumentParser(description="Control capture_2d_stream HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "command",
        choices=("status", "water-cut", "clear", "snapshot", "capture"),
        help=(
            "status: GET /status; water-cut: POST /water-cut; clear: POST /water-cut/clear; "
            "snapshot: GET /snapshot.jpg; capture: POST /capture"
        ),
    )
    parser.add_argument("--output", default=None, help="For snapshot, save JPEG to this path.")
    parser.add_argument("--name", default=None, help="For capture, record name prefix.")
    parser.add_argument("--height", default="", help="For capture, height text (e.g. 10.0mm).")
    parser.add_argument("--temperature", default="", help="For capture, temperature text.")
    parser.add_argument("--weight", default="", help="For capture, weight text (e.g. 14.2g).")
    parser.add_argument(
        "--water-cut",
        action="store_true",
        help="For capture, compute and render water-cut width.",
    )
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    if args.command == "status":
        result = _request("GET", f"{base}/status")
    elif args.command == "water-cut":
        result = _request("POST", f"{base}/water-cut", timeout=5.0)
    elif args.command == "clear":
        result = _request("POST", f"{base}/water-cut/clear", timeout=5.0)
    elif args.command == "capture":
        if not args.name or not str(args.name).strip():
            parser.error("capture requires --name")
        result = _request(
            "POST",
            f"{base}/capture",
            timeout=120.0,
            payload={
                "name": args.name.strip(),
                "height": args.height,
                "temperature": args.temperature,
                "weight": args.weight,
                "water_cut": bool(args.water_cut),
            },
        )
    else:
        req = urllib.request.Request(f"{base}/snapshot.jpg")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = resp.read()
        if args.output:
            with open(args.output, "wb") as handle:
                handle.write(data)
            print(f"Saved snapshot: {args.output}")
            return 0
        sys.stdout.buffer.write(data)
        return 0

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
