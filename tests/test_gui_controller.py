from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import src.gui_controller as gui_controller
from src.config import ConfigError
from src.gui_controller import (
    ConfigStore,
    ReviewEngine,
    ScanRecord,
    check_backend_connection,
    validate_desktop_settings,
)


def _write_base_config(root: Path) -> ConfigStore:
    incoming = root / "incoming"
    incoming.mkdir()
    (root / "config.yaml").write_text(
        """
classifier:
  backend: codex_cli
codex_cli:
  executable: codex
  model: gpt-5.6-luna
  reasoning_effort: low
  timeout_seconds: 321
  working_directory: cache/codex-cli
  require_chatgpt_login: true
lmstudio:
  base_url: http://127.0.0.1:1234/v1
  model: local-vision
watch:
  paths: [incoming]
  recursive: true
  file_stable_seconds: 0
output:
  directory: output
  operation: copy
  preserve_relative_paths: true
logs:
  directory: logs
cache:
  directory: cache
processing:
  parallel_workers: 1
  extensions: [.png, .jpg]
report:
  filename: review.html
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return ConfigStore(root)


def test_config_store_saves_gui_fields_without_weakening_auth_guard(tmp_path: Path) -> None:
    store = _write_base_config(tmp_path)
    original = store.load_settings()
    second_input = tmp_path / "second"
    second_input.mkdir()
    settings = replace(
        original,
        input_paths=(original.input_paths[0], second_input),
        output_path=tmp_path / "sorted",
        backend="lmstudio",
        reasoning_effort="medium",
        review_mode="strict",
        lmstudio_url="http://192.168.0.104:1234/v1/",
        lmstudio_model="vision-model",
        operation="move",
        recursive=False,
    )

    config = store.save(settings)

    assert store.user_path.is_file()
    assert config.classifier.backend == "lmstudio"
    assert config.watch.paths == (original.input_paths[0], second_input)
    assert config.output.directory == (tmp_path / "sorted").resolve()
    assert config.output.operation == "move"
    assert config.codex_cli.reasoning_effort == "medium"
    assert config.rules.mode == "strict"
    assert config.rules.score_fail_below == 3
    assert config.codex_cli.timeout_seconds == 321
    assert config.codex_cli.require_chatgpt_login is True
    assert config.lmstudio.base_url == "http://192.168.0.104:1234/v1"
    assert config.lmstudio.model == "vision-model"


def test_desktop_validation_uses_official_luna_efforts_and_safe_paths(tmp_path: Path) -> None:
    store = _write_base_config(tmp_path)
    settings = store.load_settings()

    validate_desktop_settings(replace(settings, reasoning_effort="ultra"))
    with pytest.raises(ConfigError, match="推論設定"):
        validate_desktop_settings(replace(settings, reasoning_effort="extreme"))
    with pytest.raises(ConfigError, match="判定基準"):
        validate_desktop_settings(replace(settings, review_mode="unknown"))

    output = tmp_path / "output"
    nested_input = output / "incoming"
    nested_input.mkdir(parents=True)
    with pytest.raises(ConfigError, match="出力フォルダの内側"):
        validate_desktop_settings(
            replace(settings, input_paths=(nested_input,), output_path=output)
        )


def test_codex_connection_check_reports_chatgpt_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _write_base_config(tmp_path)
    config = store.load_config()

    class FakeCodex:
        closed = False

        def __init__(self, _config: object) -> None:
            pass

        def get_status(self, *, refresh: bool = False) -> dict[str, str]:
            assert refresh is True
            return {
                "version": "codex-cli test",
                "authentication": "chatgpt",
                "model": "gpt-5.6-luna",
            }

        def close(self) -> None:
            type(self).closed = True

    monkeypatch.setattr(gui_controller, "CodexCLIClient", FakeCodex)

    result = check_backend_connection(config)

    assert result.ok is True
    assert result.authentication == "chatgpt"
    assert "ChatGPT認証済み" in result.message
    assert FakeCodex.closed is True


def test_lmstudio_connection_returns_models_and_missing_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _write_base_config(tmp_path)
    config = store.save(replace(store.load_settings(), backend="lmstudio"))

    class FakeLMStudio:
        def __init__(self, _config: object) -> None:
            pass

        def get_models(self) -> list[dict[str, str]]:
            return [{"id": "other-vision"}, {"id": "another-model"}]

        def close(self) -> None:
            pass

    monkeypatch.setattr(gui_controller, "LMStudioClient", FakeLMStudio)

    result = check_backend_connection(config)

    assert result.ok is False
    assert result.models == ("other-vision", "another-model")
    assert "一覧にありません" in result.message


def test_review_engine_reports_progress_and_stops_between_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _write_base_config(tmp_path)
    incoming = tmp_path / "incoming"
    for index in range(3):
        (incoming / f"{index}.png").write_bytes(b"image")
    config = store.load_config()

    class DummyClient:
        closed = False

        def close(self) -> None:
            type(self).closed = True

    class DummyScanner:
        def _is_output_path(self, _path: Path) -> bool:
            return False

        def process_file(self, path: Path, *, force: bool = False) -> ScanRecord:
            assert force is True
            return ScanRecord(
                source_path=str(path),
                destination_path=str(tmp_path / "output" / "pass" / path.name),
                file_hash=path.name,
                result="PASS",
                confidence=0.9,
                summary="ok",
            )

    class DummyReport:
        builds = 0

        def build(self) -> dict[str, Path]:
            type(self).builds += 1
            return {"html": tmp_path / "review.html"}

    engine = ReviewEngine(config)
    client = DummyClient()
    monkeypatch.setattr(
        engine,
        "_create_pipeline",
        lambda _roots: (client, DummyScanner(), DummyReport()),
    )
    updates = []

    def on_progress(progress: object) -> None:
        updates.append(progress)
        if getattr(progress, "completed", 0) == 1:
            engine.stop()

    summary = engine.run(force=True, on_progress=on_progress)

    assert summary.cancelled is True
    assert len(summary.records) == 1
    assert summary.counts["PASS"] == 1
    assert updates[0].total == 3
    assert updates[-1].completed == 1
    assert DummyClient.closed is True
    assert DummyReport.builds >= 1


def test_review_engine_honors_stop_requested_before_worker_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _write_base_config(tmp_path)
    (tmp_path / "incoming" / "one.png").write_bytes(b"image")
    engine = ReviewEngine(store.load_config())

    class DummyClient:
        def close(self) -> None:
            pass

    class DummyScanner:
        calls = 0

        def _is_output_path(self, _path: Path) -> bool:
            return False

        def process_file(self, _path: Path, *, force: bool = False) -> None:
            type(self).calls += 1

    class DummyReport:
        def build(self) -> dict[str, Path]:
            return {"html": tmp_path / "review.html"}

    monkeypatch.setattr(
        engine,
        "_create_pipeline",
        lambda _roots: (DummyClient(), DummyScanner(), DummyReport()),
    )
    engine.stop()

    summary = engine.run()

    assert summary.cancelled is True
    assert DummyScanner.calls == 0
