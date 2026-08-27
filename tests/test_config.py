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
    assert config.rules.mode == "standard"


def test_crop_recheck_defaults_are_safe_and_future_targets_are_disabled(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path / "config.yaml",
            "watch:\n  paths: [input]\n",
        )
    )

    crop = config.crop_recheck
    assert crop.enabled is False
    assert crop.mode == "balanced"
    assert crop.keep_crop_files is True
    assert crop.crop_cache_dir == (tmp_path / "cache" / "crops").resolve()
    assert crop.planner.trigger_low_scores == ("hands", "face", "anatomy")
    assert crop.planner.review_problem_keywords == (
        "finger",
        "hand",
        "face",
        "eye",
        "foot",
        "toe",
    )
    assert crop.planner.max_hand_crops == 4
    assert crop.planner.max_face_crops == 2
    assert crop.planner.max_foot_crops == 4
    assert crop.detectors.provider == "auto"
    assert crop.detectors.detector_failure_policy == "review"
    assert crop.targets["face"].enabled is True
    assert crop.targets["hand"].enabled is True
    assert crop.targets["foot"].enabled is True
    assert crop.targets["upper_body"].enabled is False
    assert crop.targets["lower_body"].enabled is False


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
        "watch:\n  paths: [input]\nrules:\n  mode: unknown\n",
        "watch:\n  paths: [output/incoming]\noutput:\n  directory: output\n",
        "watch:\n  paths: [input]\ncrop_recheck:\n  enabled: 'false'\n",
        "watch:\n  paths: [input]\ncrop_recheck:\n  mode: turbo\n",
        "watch:\n  paths: [input]\ncrop_recheck:\n  planner:\n    min_confidence_for_skip: .inf\n",
        "watch:\n  paths: [input]\ncrop_recheck:\n  planner:\n    max_hand_crops: 0\n",
        "watch:\n  paths: [input]\ncrop_recheck:\n  planner:\n    trigger_low_scores: [typo]\n",
        "watch:\n  paths: [input]\ncrop_recheck:\n  planner:\n    dedup_iou: 0\n",
        "watch:\n  paths: [input]\ncrop_recheck:\n  detectors:\n    detector_failure_policy: ignore\n",
        "watch:\n  paths: [input]\ncrop_recheck:\n  targets:\n    upper_body:\n      enabled: true\n",
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


def test_crop_recheck_nested_settings_are_loaded_and_paths_resolved(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path / "config.yaml",
            """
watch:
  paths: [input]
crop_recheck:
  enabled: true
  mode: strict
  keep_crop_files: false
  crop_cache_dir: state/crops
  planner:
    run_on_review_only: true
    min_confidence_for_skip: 0.95
    max_hand_crops: 3
    max_face_crops: 1
    max_foot_crops: 2
    min_crop_size: 128
    crop_padding_ratio: 0.2
    dedup_iou: 0.6
    min_detector_confidence: 0.85
    large_foot_area: 0.05
  detectors:
    provider: vlm
    allow_fallback: false
    person_required_for_balanced: false
  targets:
    face: {enabled: true}
    hand: {enabled: false}
    foot: {enabled: true}
""",
        )
    )

    crop = config.crop_recheck
    assert crop.enabled is True
    assert crop.mode == "strict"
    assert crop.keep_crop_files is False
    assert crop.crop_cache_dir == (tmp_path / "state" / "crops").resolve()
    assert crop.planner.run_on_review_only is True
    assert crop.planner.min_confidence_for_skip == 0.95
    assert crop.planner.max_hand_crops == 3
    assert crop.planner.min_crop_size == 128
    assert crop.planner.dedup_iou == 0.6
    assert crop.detectors.provider == "vlm"
    assert crop.detectors.allow_fallback is False
    assert crop.detectors.person_required_for_balanced is False
    assert crop.targets["hand"].enabled is False
    # Omitted reserved keys are still present and disabled.
    assert crop.targets["upper_body"].enabled is False
