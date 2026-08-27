from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import pytest

from src.codex_cli_client import (
    CodexCLIAuthError,
    CodexCLIClient,
    CodexCLIClientConfig,
)
from src.models import ResultLabel


def _payload() -> dict[str, object]:
    return {
        "result": "PASS",
        "confidence": 0.96,
        "scores": {
            "anatomy": 9,
            "hands": 9,
            "face": 9,
            "artifacts": 9,
            "composition": 9,
        },
        "problems": [],
        "summary": "No obvious issues found.",
    }


class _Runner:
    def __init__(self, *, login: str = "Logged in using ChatGPT", malformed_first: bool = False) -> None:
        self.login = login
        self.malformed_first = malformed_first
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.schemas: list[dict[str, Any]] = []
        self.image_sizes: list[tuple[int, int]] = []
        self.exec_count = 0

    def __call__(self, args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        command = list(args)
        self.calls.append((command, kwargs))
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, "codex-cli test", "")
        if command[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(command, 0, self.login, "")
        assert command[1] == "exec"
        self.exec_count += 1
        schema_path = Path(command[command.index("--output-schema") + 1])
        output_path = Path(command[command.index("--output-last-message") + 1])
        image_path = Path(command[command.index("--image") + 1])
        try:
            from PIL import Image

            with Image.open(image_path) as opened:
                self.image_sizes.append(opened.size)
        except OSError:
            pass
        self.schemas.append(json.loads(schema_path.read_text(encoding="utf-8")))
        if self.malformed_first and self.exec_count == 1:
            output_path.write_text("not json", encoding="utf-8")
        else:
            output_path.write_text(json.dumps(_payload()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")


def _config(tmp_path: Path, **changes: Any) -> CodexCLIClientConfig:
    values: dict[str, Any] = {
        "executable": "codex-test",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "timeout_seconds": 5,
        "retries": 0,
        "retry_delay_seconds": 0,
        "working_directory": tmp_path / "work",
        "require_chatgpt_login": True,
        "ignore_user_config": True,
        "ephemeral": True,
    }
    values.update(changes)
    return CodexCLIClientConfig(**values)


def test_classify_uses_image_schema_read_only_and_chatgpt_auth(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"fake image content")
    runner = _Runner()
    client = CodexCLIClient(_config(tmp_path), runner=runner, executable_path="codex-test")

    result = client.classify_image(image)

    assert result.result is ResultLabel.PASS
    assert runner.exec_count == 1
    exec_args, kwargs = next(call for call in runner.calls if call[0][1] == "exec")
    assert exec_args[exec_args.index("--model") + 1] == "gpt-5.6-luna"
    assert exec_args[exec_args.index("--image") + 1] == str(image.resolve())
    assert exec_args[exec_args.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in exec_args
    assert "--ephemeral" in exec_args
    assert exec_args[-1] == "-"
    assert "Analyze only the attached image" in kwargs["input"]
    assert "Inspection mode: STANDARD" in kwargs["input"]
    assert "intentional anatomical exaggeration" in kwargs["input"]
    if sys.platform == "win32":
        assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in kwargs
    assert runner.schemas[0]["additionalProperties"] is False
    assert set(runner.schemas[0]["required"]) == {
        "result",
        "confidence",
        "scores",
        "problems",
        "summary",
    }


def test_api_key_authentication_is_rejected_before_model_request(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"fake image content")
    runner = _Runner(login="Logged in using an API key")
    client = CodexCLIClient(_config(tmp_path), runner=runner, executable_path="codex-test")

    with pytest.raises(CodexCLIAuthError, match="avoid OpenAI Platform API billing"):
        client.classify_image(image)

    assert runner.exec_count == 0


def test_malformed_result_is_retried(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"fake image content")
    runner = _Runner(malformed_first=True)
    client = CodexCLIClient(
        _config(tmp_path, retries=1),
        runner=runner,
        executable_path="codex-test",
    )

    result = client.classify_image(image)

    assert result.result is ResultLabel.PASS
    assert runner.exec_count == 2


def test_authentication_is_rechecked_for_each_image(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    runner = _Runner()
    client = CodexCLIClient(_config(tmp_path), runner=runner, executable_path="codex-test")

    client.classify_image(first)
    client.classify_image(second)

    login_calls = [call for call, _ in runner.calls if call[1:] == ["login", "status"]]
    assert len(login_calls) == 2
    assert runner.exec_count == 2


def test_oversized_image_is_downscaled_without_changing_source(tmp_path: Path) -> None:
    from PIL import Image

    image = tmp_path / "large.png"
    Image.new("RGB", (512, 256), color=(10, 20, 30)).save(image)
    runner = _Runner()
    client = CodexCLIClient(
        _config(tmp_path, max_image_dimension=256),
        runner=runner,
        executable_path="codex-test",
    )

    client.classify_image(image)

    assert runner.image_sizes == [(256, 128)]
    with Image.open(image) as original:
        assert original.size == (512, 256)
