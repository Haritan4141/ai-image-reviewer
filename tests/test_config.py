from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError, load_config


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_resolves_relative_paths_from_yaml_directory(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        """
watch:
  paths: [images/incoming]
  recursive: false
  mode: polling
output:
  directory: out
  operation: copy
logs:
  directory: state/logs
cache:
  directory: state/cache
""",
    )

    config = load_config(config_path)

    assert config.watch.paths == ((tmp_path / "images" / "incoming").resolve(),)
    assert config.output.directory == (tmp_path / "out").resolve()
    assert config.results_jsonl_path == (tmp_path / "state" / "logs" / "results.jsonl").resolve()
    assert config.processed_cache_path == (tmp_path / "state" / "cache" / "processed.json").resolve()


def test_ensure_directories_creates_only_application_directories(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        "watch:\n  paths: [input]\noutput:\n  directory: output\nlogs:\n  directory: logs\ncache:\n  directory: cache\n",
    )
    config = load_config(config_path)

    config.ensure_directories()

    assert (tmp_path / "output/pass").is_dir()
    assert (tmp_path / "output/review").is_dir()
    assert (tmp_path / "output/fail").is_dir()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "cache").is_dir()
    assert not (tmp_path / "input").exists()


@pytest.mark.parametrize(
    "body",
    [
        "watch:\n  paths: [input]\noutput:\n  operation: delete\n",
        "watch:\n  paths: []\n",
        "watch:\n  paths: [input]\nrules:\n  threshold_review: 0.9\n  threshold_pass: 0.5\n",
        "watch:\n  paths: [input]\nprocessing:\n  parallel_workers: 0\n",
        "watch:\n  paths: [input]\nrules:\n  fail_score_count: 0\n",
        "watch:\n  paths: [output/incoming]\noutput:\n  directory: output\n",
    ],
)
def test_invalid_configuration_raises_config_error(tmp_path: Path, body: str) -> None:
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path / "config.yaml", body))


def test_missing_configuration_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_codex_cli_backend_configuration(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path / "config.yaml",
            """
classifier:
  backend: codex_cli
codex_cli:
  executable: codex
  model: gpt-5.6-luna
  reasoning_effort: low
  timeout_seconds: 90
  retries: 0
  working_directory: state/codex
  require_chatgpt_login: true
watch:
  paths: [input]
""",
        )
    )

    assert config.classifier.backend == "codex_cli"
    assert config.codex_cli.model == "gpt-5.6-luna"
    assert config.codex_cli.timeout_seconds == 90
    assert config.codex_cli.working_directory == (tmp_path / "state" / "codex").resolve()
    assert config.codex_cli.require_chatgpt_login is True
