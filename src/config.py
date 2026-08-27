"""Configuration loading and validation for ai-image-reviewer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
import math
import numbers
from pathlib import Path
from typing import Any, Mapping

import yaml


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
REVIEW_MODES = ("lenient", "standard", "strict")
CROP_MODES = ("fast", "balanced", "strict")
CROP_DETECTOR_PROVIDERS = ("auto", "vlm", "none")
CROP_DETECTOR_FAILURE_POLICIES = ("review",)
CROP_TARGETS = ("face", "hand", "foot", "upper_body", "lower_body")
SUPPORTED_CROP_TARGETS = ("face", "hand", "foot")
RULE_MODE_PRESETS: dict[str, dict[str, float | int]] = {
    "lenient": {
        "threshold_pass": 0.70,
        "threshold_review": 0.40,
        "score_review_below": 4,
        "score_fail_below": 2,
        "fail_score_count": 3,
    },
    "standard": {
        "threshold_pass": 0.80,
        "threshold_review": 0.50,
        "score_review_below": 5,
        "score_fail_below": 2,
        "fail_score_count": 2,
    },
    "strict": {
        "threshold_pass": 0.85,
        "threshold_review": 0.50,
        "score_review_below": 6,
        "score_fail_below": 3,
        "fail_score_count": 2,
    },
}
RULE_MODE_FAIL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lenient": (
        "severe deformation",
        "multiple extra limbs",
        "unrecognizable face",
        "heavy generation noise",
    ),
    "standard": (
        "severe deformation",
        "multiple extra limbs",
        "unrecognizable face",
        "heavy generation noise",
    ),
    "strict": (
        "severe deformation",
        "major anatomy failure",
        "multiple extra limbs",
        "unrecognizable face",
        "heavy generation noise",
    ),
}


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
    mode: str = "standard"
    threshold_pass: float = 0.80
    threshold_review: float = 0.50
    score_review_below: int = 5
    score_fail_below: int = 2
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
        "multiple extra limbs",
        "unrecognizable face",
        "heavy generation noise",
    )


@dataclass(frozen=True, slots=True)
class CropTargetConfig:
    """Whether a region kind participates in crop re-checking.

    ``upper_body`` and ``lower_body`` are deliberately represented here even
    though they are reserved for a future detector/prompt implementation.
    The loader rejects enabling those reserved targets so an apparently
    successful configuration can never silently skip requested checks.
    """

    enabled: bool = False


@dataclass(frozen=True, slots=True)
class CropPlannerConfig:
    """Rules controlling when and how region crops are planned."""

    run_on_review_only: bool = False
    min_confidence_for_skip: float = 0.90
    trigger_low_scores: tuple[str, ...] = ("hands", "face", "anatomy")
    review_problem_keywords: tuple[str, ...] = (
        "finger",
        "hand",
        "face",
        "eye",
        "foot",
        "toe",
    )
    max_hand_crops: int = 4
    max_face_crops: int = 2
    max_foot_crops: int = 4
    min_crop_size: int = 96
    crop_padding_ratio: float = 0.15
    dedup_iou: float = 0.50
    min_detector_confidence: float = 0.80
    large_foot_area: float = 0.04


@dataclass(frozen=True, slots=True)
class CropDetectorsConfig:
    """Region detector selection and fail-safe behavior."""

    provider: str = "auto"
    allow_fallback: bool = True
    detector_failure_policy: str = "review"
    person_required_for_balanced: bool = True


def _default_crop_targets() -> dict[str, CropTargetConfig]:
    return {
        "face": CropTargetConfig(enabled=True),
        "hand": CropTargetConfig(enabled=True),
        "foot": CropTargetConfig(enabled=True),
        "upper_body": CropTargetConfig(enabled=False),
        "lower_body": CropTargetConfig(enabled=False),
    }


@dataclass(frozen=True, slots=True)
class CropRecheckConfig:
    """Optional second-pass face/hand/foot inspection configuration."""

    enabled: bool = False
    mode: str = "balanced"
    keep_crop_files: bool = True
    crop_cache_dir: Path = Path("cache/crops")
    planner: CropPlannerConfig = field(default_factory=CropPlannerConfig)
    detectors: CropDetectorsConfig = field(default_factory=CropDetectorsConfig)
    targets: Mapping[str, CropTargetConfig] = field(default_factory=_default_crop_targets)


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
    crop_recheck: CropRecheckConfig = field(default_factory=CropRecheckConfig)
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
        if self.crop_recheck.enabled:
            self.crop_recheck.crop_cache_dir.mkdir(parents=True, exist_ok=True)


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


def _strict_bool(value: Any, *, name: str) -> bool:
    """Read a YAML boolean without accepting truthy strings such as ``"false"``."""

    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be a boolean (true or false)")
    return value


def _strict_number(value: Any, *, name: str) -> float:
    """Read a finite, non-boolean YAML number."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigError(f"'{name}' must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"'{name}' must be finite")
    return number


def _strict_int(value: Any, *, name: str) -> int:
    """Read a positive-count integer without truncating floats or strings."""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ConfigError(f"'{name}' must be an integer")
    return int(value)


def _strict_enum(value: Any, *, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"'{name}' must be one of: {', '.join(choices)}")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ConfigError(f"'{name}' must be one of: {', '.join(choices)}")
    return normalized


def _crop_targets(raw: Mapping[str, Any]) -> dict[str, CropTargetConfig]:
    """Parse target entries while retaining all future-reserved keys."""

    defaults = _default_crop_targets()
    unknown = sorted(set(raw) - set(CROP_TARGETS))
    if unknown:
        raise ConfigError(
            "crop_recheck.targets contains unsupported target(s): " + ", ".join(unknown)
        )
    parsed = dict(defaults)
    for name, value in raw.items():
        if not isinstance(value, Mapping):
            raise ConfigError(f"'crop_recheck.targets.{name}' must be a YAML mapping")
        enabled = value.get("enabled", defaults[name].enabled)
        parsed[name] = CropTargetConfig(
            enabled=_strict_bool(enabled, name=f"crop_recheck.targets.{name}.enabled")
        )
    return parsed


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
    crop = _section(loaded, "crop_recheck")
    crop_planner = _section(crop, "planner")
    crop_detectors = _section(crop, "detectors")
    crop_targets = _section(crop, "targets")
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
            mode=str(rules.get("mode", "standard")).strip().lower(),
            threshold_pass=float(rules.get("threshold_pass", 0.80)),
            threshold_review=float(rules.get("threshold_review", 0.50)),
            score_review_below=int(rules.get("score_review_below", 5)),
            score_fail_below=int(rules.get("score_fail_below", 2)),
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
        crop_recheck=CropRecheckConfig(
            enabled=_strict_bool(crop.get("enabled", False), name="crop_recheck.enabled"),
            mode=_strict_enum(crop.get("mode", "balanced"), name="crop_recheck.mode", choices=CROP_MODES),
            keep_crop_files=_strict_bool(
                crop.get("keep_crop_files", True), name="crop_recheck.keep_crop_files"
            ),
            crop_cache_dir=_resolve_path(crop.get("crop_cache_dir", "cache/crops"), root),
            planner=CropPlannerConfig(
                run_on_review_only=_strict_bool(
                    crop_planner.get("run_on_review_only", False),
                    name="crop_recheck.planner.run_on_review_only",
                ),
                min_confidence_for_skip=_strict_number(
                    crop_planner.get("min_confidence_for_skip", 0.90),
                    name="crop_recheck.planner.min_confidence_for_skip",
                ),
                trigger_low_scores=_as_tuple(
                    crop_planner.get(
                        "trigger_low_scores",
                        list(CropPlannerConfig().trigger_low_scores),
                    ),
                    name="crop_recheck.planner.trigger_low_scores",
                ),
                review_problem_keywords=_as_tuple(
                    crop_planner.get(
                        "review_problem_keywords",
                        list(CropPlannerConfig().review_problem_keywords),
                    ),
                    name="crop_recheck.planner.review_problem_keywords",
                ),
                max_hand_crops=_strict_int(
                    crop_planner.get("max_hand_crops", 4),
                    name="crop_recheck.planner.max_hand_crops",
                ),
                max_face_crops=_strict_int(
                    crop_planner.get("max_face_crops", 2),
                    name="crop_recheck.planner.max_face_crops",
                ),
                max_foot_crops=_strict_int(
                    crop_planner.get("max_foot_crops", 4),
                    name="crop_recheck.planner.max_foot_crops",
                ),
                min_crop_size=_strict_int(
                    crop_planner.get("min_crop_size", 96),
                    name="crop_recheck.planner.min_crop_size",
                ),
                crop_padding_ratio=_strict_number(
                    crop_planner.get("crop_padding_ratio", 0.15),
                    name="crop_recheck.planner.crop_padding_ratio",
                ),
                dedup_iou=_strict_number(
                    crop_planner.get("dedup_iou", 0.50),
                    name="crop_recheck.planner.dedup_iou",
                ),
                min_detector_confidence=_strict_number(
                    crop_planner.get("min_detector_confidence", 0.80),
                    name="crop_recheck.planner.min_detector_confidence",
                ),
                large_foot_area=_strict_number(
                    crop_planner.get("large_foot_area", 0.04),
                    name="crop_recheck.planner.large_foot_area",
                ),
            ),
            detectors=CropDetectorsConfig(
                provider=_strict_enum(
                    crop_detectors.get("provider", "auto"),
                    name="crop_recheck.detectors.provider",
                    choices=CROP_DETECTOR_PROVIDERS,
                ),
                allow_fallback=_strict_bool(
                    crop_detectors.get("allow_fallback", True),
                    name="crop_recheck.detectors.allow_fallback",
                ),
                detector_failure_policy=_strict_enum(
                    crop_detectors.get("detector_failure_policy", "review"),
                    name="crop_recheck.detectors.detector_failure_policy",
                    choices=CROP_DETECTOR_FAILURE_POLICIES,
                ),
                person_required_for_balanced=_strict_bool(
                    crop_detectors.get("person_required_for_balanced", True),
                    name="crop_recheck.detectors.person_required_for_balanced",
                ),
            ),
            targets=_crop_targets(crop_targets),
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
    if config.rules.mode not in REVIEW_MODES:
        raise ConfigError("rules.mode must be 'lenient', 'standard', or 'strict'")
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
    _validate_crop_recheck(config.crop_recheck)
    output_key = os.path.normcase(os.path.normpath(os.fspath(config.output.directory)))
    for watch_path in config.watch.paths:
        try:
            common = os.path.commonpath([os.fspath(config.output.directory), os.fspath(watch_path)])
        except ValueError:
            continue
        if os.path.normcase(os.path.normpath(common)) == output_key:
            raise ConfigError("watch paths must not be inside the configured output directory")


def _validate_crop_recheck(config: CropRecheckConfig) -> None:
    """Validate crop settings independently from the legacy full-image rules."""

    if not isinstance(config.enabled, bool):
        raise ConfigError("crop_recheck.enabled must be a boolean")
    if config.mode not in CROP_MODES:
        raise ConfigError("crop_recheck.mode must be 'fast', 'balanced', or 'strict'")
    if not isinstance(config.keep_crop_files, bool):
        raise ConfigError("crop_recheck.keep_crop_files must be a boolean")
    if not isinstance(config.crop_cache_dir, Path) or not config.crop_cache_dir.is_absolute():
        raise ConfigError("crop_recheck.crop_cache_dir must resolve to an absolute path")

    planner = config.planner
    if not isinstance(planner.run_on_review_only, bool):
        raise ConfigError("crop_recheck.planner.run_on_review_only must be a boolean")
    _range_number(
        planner.min_confidence_for_skip,
        name="crop_recheck.planner.min_confidence_for_skip",
        lower=0,
        upper=1,
    )
    _string_tuple(planner.trigger_low_scores, name="crop_recheck.planner.trigger_low_scores")
    if set(planner.trigger_low_scores) - {"hands", "face", "anatomy", "artifacts", "composition"}:
        raise ConfigError("crop_recheck.planner.trigger_low_scores contains an unknown score name")
    _string_tuple(
        planner.review_problem_keywords,
        name="crop_recheck.planner.review_problem_keywords",
    )
    for field_name in ("max_hand_crops", "max_face_crops", "max_foot_crops", "min_crop_size"):
        value = getattr(planner, field_name)
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1:
            raise ConfigError(f"crop_recheck.planner.{field_name} must be a positive integer")
    _range_number(
        planner.crop_padding_ratio,
        name="crop_recheck.planner.crop_padding_ratio",
        lower=0,
        upper=1,
    )
    _range_number(planner.dedup_iou, name="crop_recheck.planner.dedup_iou", lower=0, upper=1,
                  inclusive_lower=False)
    _range_number(
        planner.min_detector_confidence,
        name="crop_recheck.planner.min_detector_confidence",
        lower=0,
        upper=1,
    )
    _range_number(
        planner.large_foot_area,
        name="crop_recheck.planner.large_foot_area",
        lower=0,
        upper=1,
        inclusive_lower=False,
    )

    detectors = config.detectors
    if detectors.provider not in CROP_DETECTOR_PROVIDERS:
        raise ConfigError("crop_recheck.detectors.provider must be 'auto', 'vlm', or 'none'")
    if not isinstance(detectors.allow_fallback, bool):
        raise ConfigError("crop_recheck.detectors.allow_fallback must be a boolean")
    if detectors.detector_failure_policy not in CROP_DETECTOR_FAILURE_POLICIES:
        raise ConfigError(
            "crop_recheck.detectors.detector_failure_policy must be 'review'"
        )
    if not isinstance(detectors.person_required_for_balanced, bool):
        raise ConfigError(
            "crop_recheck.detectors.person_required_for_balanced must be a boolean"
        )

    if not isinstance(config.targets, Mapping):
        raise ConfigError("crop_recheck.targets must be a mapping")
    unknown = sorted(set(config.targets) - set(CROP_TARGETS))
    if unknown:
        raise ConfigError(
            "crop_recheck.targets contains unsupported target(s): " + ", ".join(unknown)
        )
    for target_name, target in config.targets.items():
        if not isinstance(target, CropTargetConfig):
            raise ConfigError(f"crop_recheck.targets.{target_name} must be a target mapping")
        if not isinstance(target.enabled, bool):
            raise ConfigError(f"crop_recheck.targets.{target_name}.enabled must be a boolean")
    for reserved in ("upper_body", "lower_body"):
        target = config.targets.get(reserved)
        if target is not None and target.enabled:
            raise ConfigError(
                f"crop_recheck.targets.{reserved} is reserved for a future implementation; "
                "leave enabled: false"
            )


def _range_number(
    value: Any,
    *,
    name: str,
    lower: float,
    upper: float,
    inclusive_lower: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be a finite number")
    lower_ok = number >= lower if inclusive_lower else number > lower
    if not lower_ok or number > upper:
        bracket = "<=" if inclusive_lower else "<"
        raise ConfigError(f"{name} must satisfy {lower} {bracket} value <= {upper}")


def _string_tuple(value: Any, *, name: str) -> None:
    if not isinstance(value, tuple) or not value or any(not item.strip() for item in value):
        raise ConfigError(f"{name} must be a non-empty list of strings")
