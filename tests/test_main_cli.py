from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from PIL import Image

import main


class _LMStudioHandler(BaseHTTPRequestHandler):
    post_count = 0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        payload = {"object": "list", "data": [{"id": "test-vlm", "object": "model"}]}
        self._send(payload)

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        type(self).post_count += 1
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        assert request["model"] == "test-vlm"
        classification = {
            "result": "PASS",
            "confidence": 0.96,
            "scores": {"anatomy": 9, "hands": 9, "face": 9, "artifacts": 9, "composition": 9},
            "problems": [],
            "summary": "No obvious issues found.",
        }
        self._send(
            {
                "id": "chatcmpl-test",
                "choices": [{"message": {"role": "assistant", "content": json.dumps(classification)}}],
            }
        )

    def _send(self, payload: object) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _write_config(path: Path, port: int) -> None:
    path.write_text(
        f"""
watch:
  paths: [incoming]
  recursive: true
  mode: polling
  polling_interval_seconds: 0.1
  file_stable_seconds: 0
output:
  directory: output
  operation: copy
  preserve_relative_paths: true
lmstudio:
  base_url: http://127.0.0.1:{port}/v1
  model: test-vlm
  timeout_seconds: 5
  retries: 0
processing:
  parallel_workers: 1
  extensions: [.png, .jpg, .jpeg, .webp]
logs:
  directory: logs
cache:
  directory: cache
report:
  filename: review.html
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_scan_cli_end_to_end_with_mock_lmstudio(tmp_path: Path) -> None:
    _LMStudioHandler.post_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LMStudioHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        incoming = tmp_path / "incoming" / "nested"
        incoming.mkdir(parents=True)
        image_path = incoming / "sample.png"
        Image.new("RGB", (24, 24), color=(20, 120, 220)).save(image_path)
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, server.server_port)

        assert main.run(["--config", str(config_path), "test-lmstudio"]) == 0
        assert main.run(["--config", str(config_path), "scan"]) == 0
        assert (tmp_path / "output" / "pass" / "nested" / "sample.png").is_file()
        assert (tmp_path / "logs" / "results.jsonl").is_file()
        assert (tmp_path / "logs" / "latest_summary.csv").is_file()
        assert (tmp_path / "review.html").is_file()
        assert "sample.png" in (tmp_path / "review.html").read_text(encoding="utf-8")
        assert _LMStudioHandler.post_count == 1

        # The content-hash cache prevents a second API request.
        assert main.run(["--config", str(config_path), "scan"]) == 0
        assert _LMStudioHandler.post_count == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _CodexClient:
    classify_count = 0

    def __init__(self, _config: object) -> None:
        pass

    def get_status(self, *, refresh: bool = False) -> dict[str, str]:
        return {
            "version": "codex-cli test",
            "authentication": "chatgpt",
            "login_status": "Logged in using ChatGPT",
            "model": "gpt-5.6-luna",
        }

    def classify_image(self, _image: Path, *, image_name: str | None = None) -> dict[str, object]:
        type(self).classify_count += 1
        return {
            "result": "PASS",
            "confidence": 0.96,
            "scores": {"anatomy": 9, "hands": 9, "face": 9, "artifacts": 9, "composition": 9},
            "problems": [],
            "summary": f"Checked {image_name}",
        }

    def close(self) -> None:
        pass


def test_scan_cli_end_to_end_with_codex_backend(tmp_path: Path, monkeypatch: object) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    image_path = incoming / "sample.png"
    Image.new("RGB", (24, 24), color=(20, 120, 220)).save(image_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
classifier:
  backend: codex_cli
codex_cli:
  model: gpt-5.6-luna
  working_directory: cache/codex-cli
  require_chatgpt_login: true
watch:
  paths: [incoming]
  file_stable_seconds: 0
output:
  directory: output
  operation: copy
logs:
  directory: logs
cache:
  directory: cache
processing:
  parallel_workers: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _CodexClient.classify_count = 0
    monkeypatch.setattr(main, "CodexCLIClient", _CodexClient)  # type: ignore[attr-defined]

    assert main.run(["--config", str(config_path), "test-codex"]) == 0
    assert main.run(["--config", str(config_path), "scan"]) == 0
    assert _CodexClient.classify_count == 1
    assert (tmp_path / "output" / "pass" / "sample.png").is_file()
