"""Configuration loading and validation for ai-image-reviewer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


class ConfigError(ValueError):
    """Raised when ``config.yaml`` contains an invalid value."""


@dataclass(frozen=True, slots=True)
class LMStudioConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "qwen3-vl-8b"
    timeout_seconds: float = 120.0
    retries: int = 2
    retry_delay_seconds: float = 2.0
    max_image_dimension: int = 2048
    jpeg_quality: int = 90


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    backend: str = "lmstudio"


@dataclass(frozen=True, slots=True)
class CodexCLIConfig:
    executable: str = "codex"
    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "low"
    timeout_seconds: float = 180.0
    retries: int = 1
    retry_delay_seconds: float = 2.0
    max_image_dimension: int = 2048
    jpeg_quality: int = 90
    working_directory: Path = Path("cache/codex-cli")
    require_chatgpt_login: bool = True
    ignore_user_config: bool = True
    ephemeral: bool = True


@dataclass(frozen=True, slots=True)
class RulesConfig:
    threshold_pass: float = 0.85
    threshold_review: float = 0.50
    score_review_below: int = 6
    score_fail_below: int = 3
    fail_score_count: int = 2
    review_problem_keywords: tuple[str, ...] = (
        "extra finger",
        "missing finger",
        "fused finger",
        "deformed hand",
        "extra limb",
        "missing limb",
        "fused body",
        "fused object",
        "malformed anatomy",
    )
    fail_problem_keywords: tuple[str, ...] = (
        "severe deformation",
        "major anatomy failure",
        "multiple extra limbs",
        "unrecognizable face",
        "heavy generation noise",
    )


@dataclass(frozen=True, slots=True)
class WatchConfig:
    paths: tuple[Path, ...] = ()
    recursive: bool = True
    mode: str = "polling"
    polling_interval_seconds: float = 5.0
    file_stable_seconds: float = 2.0


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: Path = Path("output")
    operation: str = "copy"
    preserve_relative_paths: bool = True


@dataclass(frozen=True, slots=True)
class LogsConfig:
    directory: Path = Path("logs")
    results_jsonl: str = "results.jsonl"
    summary_csv: str = "latest_summary.csv"
    application_log: str = "app.log"


@dataclass(frozen=True, slots=True)
class CacheConfig:
    directory: Path = Path("cache")
    processed_file: str = "processed.json"
    use_content_hash: bool = True


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    parallel_workers: int = 1
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS


@dataclass(frozen=True, slots=True)
class ReportConfig:
    filename: str = "review.html"
    thumbnail_width: int = 320
    thumbnail_height: int = 320


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated application configuration with absolute filesystem paths."""

    project_root: Path
    config_path: Path
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
    codex_cli: CodexCLIConfig = field(default_factory=CodexCLIConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    watch: WatchConfig = field(default_factory=WatchConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logs: LogsConfig = field(default_factory=LogsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    @property
    def results_jsonl_path(self) -> Path:
        return self.logs.directory / self.logs.results_jsonl

    @property
    def summary_csv_path(self) -> Path:
        return self.logs.directory / self.logs.summary_csv

    @property
    def application_log_path(self) -> Path:
        return self.logs.directory / self.logs.application_log

    @property
    def processed_cache_path(self) -> Path:
        return self.cache.directory / self.cache.processed_file

    @property
    def report_path(self) -> Path:
        return self.project_root / self.report.filename

    def ensure_directories(self) -> None:
        """Create application-owned directories, never any configured input path."""
        self.output.directory.mkdir(parents=True, exist_ok=True)
        for result in ("pass", "review", "fail"):
            (self.output.directory / result).mkdir(parents=True, exist_ok=True)
        self.logs.directory.mkdir(parents=True, exist_ok=True)
        self.cache.directory.mkdir(parents=True, exist_ok=True)


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a YAML mapping")
    return value


def _resolve_path(value: str | Path, root: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if not expanded:
        raise ConfigError("path values must not be empty")
    path = Path(expanded)
    return path if path.is_absolute() else (root / path).resolve()


def _as_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"'{name}' must be a list of strings")
    return tuple(value)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load YAML, resolve relative paths against the YAML directory, and validate."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read configuration: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ConfigError("config.yaml must contain a mapping at its root")

    root = config_path.parent
    classifier = _section(loaded, "classifier")
    lm = _section(loaded, "lmstudio")
    codex = _section(loaded, "codex_cli")
    rules = _section(loaded, "rules")
    watch = _section(loaded, "watch")
    output = _section(loaded, "output")
    logs = _section(loaded, "logs")
    cache = _section(loaded, "cache")
    processing = _section(loaded, "processing")
    report = _section(loaded, "report")

    raw_watch_paths = watch.get("paths", ["samples/incoming"])
    if not isinstance(raw_watch_paths, list):
        raise ConfigError("'watch.paths' must be a list")
    watch_paths = tuple(_resolve_path(item, root) for item in raw_watch_paths)

    extensions = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in _as_tuple(
            processing.get("extensions", list(SUPPORTED_EXTENSIONS)),
            name="processing.extensions",
        )
    )

    config = AppConfig(
        project_root=root,
        config_path=config_path,
        classifier=ClassifierConfig(
            backend=str(classifier.get("backend", "lmstudio")).strip().lower(),
        ),
        lmstudio=LMStudioConfig(
            base_url=str(lm.get("base_url", "http://127.0.0.1:1234/v1")).rstrip("/"),
            model=str(lm.get("model", "qwen3-vl-8b")),
            timeout_seconds=float(lm.get("timeout_seconds", 120)),
            retries=int(lm.get("retries", 2)),
            retry_delay_seconds=float(lm.get("retry_delay_seconds", 2)),
            max_image_dimension=int(lm.get("max_image_dimension", 2048)),
            jpeg_quality=int(lm.get("jpeg_quality", 90)),
        ),
        codex_cli=CodexCLIConfig(
            executable=str(codex.get("executable", "codex")).strip(),
            model=str(codex.get("model", "gpt-5.6-luna")).strip(),
            reasoning_effort=str(codex.get("reasoning_effort", "low")).strip().lower(),
            timeout_seconds=float(codex.get("timeout_seconds", 180)),
            retries=int(codex.get("retries", 1)),
            retry_delay_seconds=float(codex.get("retry_delay_seconds", 2)),
            max_image_dimension=int(codex.get("max_image_dimension", 2048)),
            jpeg_quality=int(codex.get("jpeg_quality", 90)),
            working_directory=_resolve_path(
                codex.get("working_directory", "cache/codex-cli"),
                root,
            ),
            require_chatgpt_login=bool(codex.get("require_chatgpt_login", True)),
            ignore_user_config=bool(codex.get("ignore_user_config", True)),
            ephemeral=bool(codex.get("ephemeral", True)),
        ),
        rules=RulesConfig(
            threshold_pass=float(rules.get("threshold_pass", 0.85)),
            threshold_review=float(rules.get("threshold_review", 0.50)),
            score_review_below=int(rules.get("score_review_below", 6)),
            score_fail_below=int(rules.get("score_fail_below", 3)),
            fail_score_count=int(rules.get("fail_score_count", 2)),
            review_problem_keywords=_as_tuple(
                rules.get("review_problem_keywords", list(RulesConfig().review_problem_keywords)),
                name="rules.review_problem_keywords",
            ),
            fail_problem_keywords=_as_tuple(
                rules.get("fail_problem_keywords", list(RulesConfig().fail_problem_keywords)),
                name="rules.fail_problem_keywords",
            ),
        ),
        watch=WatchConfig(
            paths=watch_paths,
            recursive=bool(watch.get("recursive", True)),
            mode=str(watch.get("mode", "polling")).lower(),
            polling_interval_seconds=float(watch.get("polling_interval_seconds", 5)),
            file_stable_seconds=float(watch.get("file_stable_seconds", 2)),
        ),
        output=OutputConfig(
            directory=_resolve_path(output.get("directory", "output"), root),
            operation=str(output.get("operation", "copy")).lower(),
            preserve_relative_paths=bool(output.get("preserve_relative_paths", True)),
        ),
        logs=LogsConfig(
            directory=_resolve_path(logs.get("directory", "logs"), root),
            results_jsonl=str(logs.get("results_jsonl", "results.jsonl")),
            summary_csv=str(logs.get("summary_csv", "latest_summary.csv")),
            application_log=str(logs.get("application_log", "app.log")),
        ),
        cache=CacheConfig(
            directory=_resolve_path(cache.get("directory", "cache"), root),
            processed_file=str(cache.get("processed_file", "processed.json")),
            use_content_hash=bool(cache.get("use_content_hash", True)),
        ),
        processing=ProcessingConfig(
            parallel_workers=int(processing.get("parallel_workers", 1)),
            extensions=extensions,
        ),
        report=ReportConfig(
            filename=str(report.get("filename", "review.html")),
            thumbnail_width=int(report.get("thumbnail_width", 320)),
            thumbnail_height=int(report.get("thumbnail_height", 320)),
        ),
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.classifier.backend not in {"lmstudio", "codex_cli"}:
        raise ConfigError("classifier.backend must be 'lmstudio' or 'codex_cli'")
    if config.output.operation not in {"copy", "move"}:
        raise ConfigError("output.operation must be 'copy' or 'move'")
    if config.watch.mode not in {"polling", "watchdog"}:
        raise ConfigError("watch.mode must be 'polling' or 'watchdog'")
    if not config.watch.paths:
        raise ConfigError("watch.paths must contain at least one path")
    if config.processing.parallel_workers < 1:
        raise ConfigError("processing.parallel_workers must be at least 1")
    if config.rules.fail_score_count < 1:
        raise ConfigError("rules.fail_score_count must be at least 1")
    if config.lmstudio.retries < 0:
        raise ConfigError("lmstudio.retries must not be negative")
    if config.codex_cli.retries < 0:
        raise ConfigError("codex_cli.retries must not be negative")
    if not config.codex_cli.executable or not config.codex_cli.model:
        raise ConfigError("codex_cli.executable and codex_cli.model must not be empty")
    if config.codex_cli.reasoning_effort not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    }:
        raise ConfigError("codex_cli.reasoning_effort is not supported")
    if config.codex_cli.timeout_seconds <= 0:
        raise ConfigError("codex_cli.timeout_seconds must be positive")
    if config.codex_cli.max_image_dimension < 256:
        raise ConfigError("codex_cli.max_image_dimension must be at least 256")
    if not 1 <= config.codex_cli.jpeg_quality <= 100:
        raise ConfigError("codex_cli.jpeg_quality must be between 1 and 100")
    if config.lmstudio.timeout_seconds <= 0 or config.watch.polling_interval_seconds <= 0:
        raise ConfigError("timeout and polling interval values must be positive")
    if not 0 <= config.rules.threshold_review <= config.rules.threshold_pass <= 1:
        raise ConfigError("rules thresholds must satisfy 0 <= review <= pass <= 1")
    if not 1 <= config.rules.score_fail_below <= config.rules.score_review_below <= 10:
        raise ConfigError("score thresholds must satisfy 1 <= fail <= review <= 10")
    if not 1 <= config.lmstudio.jpeg_quality <= 100:
        raise ConfigError("lmstudio.jpeg_quality must be between 1 and 100")
    if config.lmstudio.max_image_dimension < 256:
        raise ConfigError("lmstudio.max_image_dimension must be at least 256")
    if not config.processing.extensions:
        raise ConfigError("processing.extensions must not be empty")
    output_key = os.path.normcase(os.path.normpath(os.fspath(config.output.directory)))
    for watch_path in config.watch.paths:
        try:
            common = os.path.commonpath([os.fspath(config.output.directory), os.fspath(watch_path)])
        except ValueError:
            continue
        if os.path.normcase(os.path.normpath(common)) == output_key:
            raise ConfigError("watch paths must not be inside the configured output directory")
