"""Tk-independent settings and execution controller for the desktop app."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import threading
from typing import Callable, Mapping, Sequence

import yaml

from .classifier import ImageClassifier, LocalRulesConfig
from .codex_cli_client import CodexCLIClient
from .config import (
    AppConfig,
    ConfigError,
    REVIEW_MODES,
    RULE_MODE_FAIL_KEYWORDS,
    RULE_MODE_PRESETS,
    load_config,
)
from .file_watcher import ImageWatcher
from .lmstudio_client import LMStudioClient
from .report_builder import ReportBuilder
from .scanner import ImageScanner, ScanRecord
from .sorter import ImageSorter
from .utils import atomic_write_text, iter_image_paths


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
BACKENDS = ("codex_cli", "lmstudio")
OPERATIONS = ("copy", "move")
REVIEW_MODE_LABELS = {
    "lenient": "緩め",
    "standard": "標準（推奨）",
    "strict": "厳格",
}
REVIEW_MODE_DESCRIPTIONS = {
    "lenient": "軽微な簡略化や曖昧さを許容し、明白な重大破綻だけを強く警告します。",
    "standard": "アニメ調・誇張・遠近・重なりを許容し、根拠がそろった場合だけFAILにします。",
    "strict": "小さな手指や輪郭まで厳しく確認し、従来に近い基準で判定します。",
}


@dataclass(frozen=True, slots=True)
class DesktopSettings:
    """Editable settings shown by the desktop application."""

    input_paths: tuple[Path, ...]
    output_path: Path
    backend: str = "codex_cli"
    codex_model: str = "gpt-5.6-luna"
    reasoning_effort: str = "low"
    review_mode: str = "standard"
    lmstudio_url: str = "http://127.0.0.1:1234/v1"
    lmstudio_model: str = "qwen3-vl-8b"
    operation: str = "copy"
    recursive: bool = True
    preserve_relative_paths: bool = True


@dataclass(frozen=True, slots=True)
class BackendConnection:
    """Result of a non-classifying backend connectivity check."""

    backend: str
    ok: bool
    message: str
    models: tuple[str, ...] = ()
    authentication: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewProgress:
    """One progress update safe to pass across a GUI queue."""

    completed: int
    total: int
    current_path: Path | None
    record: ScanRecord | None
    counts: Mapping[str, int]
    monitoring: bool = False
    event: str = "scan"


@dataclass(frozen=True, slots=True)
class ReviewSummary:
    """Final state of a batch or monitoring session."""

    records: tuple[ScanRecord, ...]
    counts: Mapping[str, int]
    report_path: Path
    cancelled: bool
    monitoring: bool


def validate_desktop_settings(settings: DesktopSettings) -> None:
    """Reject settings that would be invalid or unsafe to execute."""

    if settings.backend not in BACKENDS:
        raise ConfigError("判定バックエンドを選択してください")
    if settings.operation not in OPERATIONS:
        raise ConfigError("出力操作はcopyまたはmoveを選択してください")
    if not settings.input_paths:
        raise ConfigError("入力フォルダを1つ以上追加してください")
    missing = [path for path in settings.input_paths if not path.is_dir()]
    if missing:
        raise ConfigError("入力フォルダが見つかりません: " + ", ".join(os.fspath(path) for path in missing))
    output = settings.output_path.expanduser().resolve()
    for input_path in settings.input_paths:
        source = input_path.expanduser().resolve()
        if output == source:
            raise ConfigError("入力フォルダと出力フォルダは別にしてください")
        if source.is_relative_to(output):
            raise ConfigError("入力フォルダを出力フォルダの内側には配置できません")
    if not settings.codex_model.strip():
        raise ConfigError("Codexモデル名を入力してください")
    if settings.reasoning_effort not in REASONING_EFFORTS:
        raise ConfigError("GPT-5.6 Lunaの推論設定を選択してください")
    if settings.review_mode not in REVIEW_MODES:
        raise ConfigError("判定基準は緩め、標準、厳格から選択してください")
    if not settings.lmstudio_url.strip().lower().startswith(("http://", "https://")):
        raise ConfigError("LM Studio URLはhttp://またはhttps://から入力してください")
    if not settings.lmstudio_model.strip():
        raise ConfigError("LM Studioモデルを選択または入力してください")


class ConfigStore:
    """Load the template config and persist GUI choices to an ignored file."""

    def __init__(
        self,
        project_root: str | os.PathLike[str] | Path,
        *,
        base_filename: str = "config.yaml",
        user_filename: str = "config.local.yaml",
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.base_path = self.project_root / base_filename
        self.user_path = self.project_root / user_filename

    @property
    def active_path(self) -> Path:
        return self.user_path if self.user_path.is_file() else self.base_path

    def load_config(self) -> AppConfig:
        return load_config(self.active_path)

    def load_settings(self) -> DesktopSettings:
        config = self.load_config()
        return DesktopSettings(
            input_paths=config.watch.paths,
            output_path=config.output.directory,
            backend=config.classifier.backend,
            codex_model=config.codex_cli.model,
            reasoning_effort=config.codex_cli.reasoning_effort,
            review_mode=config.rules.mode,
            lmstudio_url=config.lmstudio.base_url,
            lmstudio_model=config.lmstudio.model,
            operation=config.output.operation,
            recursive=config.watch.recursive,
            preserve_relative_paths=config.output.preserve_relative_paths,
        )

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, object]:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"設定ファイルを読み込めません: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ConfigError("設定ファイルのルートはYAMLマッピングである必要があります")
        return dict(loaded)

    @staticmethod
    def _section(raw: dict[str, object], name: str) -> dict[str, object]:
        current = raw.get(name)
        if current is None:
            section: dict[str, object] = {}
            raw[name] = section
            return section
        if not isinstance(current, Mapping):
            raise ConfigError(f"設定セクション'{name}'が不正です")
        section = dict(current)
        raw[name] = section
        return section

    def save(self, settings: DesktopSettings) -> AppConfig:
        """Persist GUI-owned fields while preserving all advanced settings."""

        validate_desktop_settings(settings)
        source = self.user_path if self.user_path.is_file() else self.base_path
        raw = self._read_mapping(source)

        classifier = self._section(raw, "classifier")
        classifier["backend"] = settings.backend

        codex = self._section(raw, "codex_cli")
        codex["model"] = settings.codex_model.strip()
        codex["reasoning_effort"] = settings.reasoning_effort
        # Never let the GUI weaken the no-API-billing guard.
        codex["require_chatgpt_login"] = True

        rules = self._section(raw, "rules")
        previous_mode = str(rules.get("mode", "")).strip().lower()
        rules["mode"] = settings.review_mode
        if previous_mode != settings.review_mode:
            rules.update(RULE_MODE_PRESETS[settings.review_mode])
            rules["fail_problem_keywords"] = list(RULE_MODE_FAIL_KEYWORDS[settings.review_mode])

        lmstudio = self._section(raw, "lmstudio")
        lmstudio["base_url"] = settings.lmstudio_url.strip().rstrip("/")
        lmstudio["model"] = settings.lmstudio_model.strip()

        watch = self._section(raw, "watch")
        watch["paths"] = [os.fspath(path.expanduser().resolve()) for path in settings.input_paths]
        watch["recursive"] = settings.recursive

        output = self._section(raw, "output")
        output["directory"] = os.fspath(settings.output_path.expanduser().resolve())
        output["operation"] = settings.operation
        output["preserve_relative_paths"] = settings.preserve_relative_paths

        rendered = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False, default_flow_style=False)
        atomic_write_text(self.user_path, rendered)
        return load_config(self.user_path)


def check_backend_connection(config: AppConfig) -> BackendConnection:
    """Check the selected backend without classifying an image."""

    if config.classifier.backend == "codex_cli":
        client = CodexCLIClient(config)
        try:
            status = client.get_status(refresh=True)
        finally:
            client.close()
        authentication = status.get("authentication")
        ok = authentication == "chatgpt"
        message = (
            f"{status.get('version', 'Codex CLI')} / ChatGPT認証済み / {config.codex_cli.model}"
            if ok
            else "ChatGPT認証ではないため、API料金防止のため使用できません"
        )
        return BackendConnection(
            backend="codex_cli",
            ok=ok,
            message=message,
            authentication=authentication,
        )

    client = LMStudioClient(config)
    try:
        items = client.get_models()
    finally:
        client.close()
    models = tuple(str(item.get("id", "")).strip() for item in items if str(item.get("id", "")).strip())
    selected = config.lmstudio.model
    ok = selected in models
    if ok:
        message = f"LM Studio接続成功 / モデル{len(models)}件 / 選択モデルを確認"
    elif models:
        message = f"LM Studio接続成功。ただし選択モデル'{selected}'は現在の一覧にありません"
    else:
        message = "LM Studio接続成功。ただし利用可能なモデルがありません"
    return BackendConnection(backend="lmstudio", ok=ok, message=message, models=models)


ProgressCallback = Callable[[ReviewProgress], None]


class ReviewEngine:
    """Run scanning/monitoring away from Tk's main thread."""

    def __init__(self, config: AppConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("ai_image_reviewer.gui")
        self._stop_event = threading.Event()
        self._watcher: ImageWatcher | None = None
        self._state_lock = threading.RLock()
        self._running = False

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    def stop(self) -> None:
        """Request cooperative stop; an active image finishes first."""

        self._stop_event.set()

    def _create_pipeline(
        self, roots: Sequence[Path]
    ) -> tuple[LMStudioClient | CodexCLIClient, ImageScanner, ReportBuilder]:
        if self.config.classifier.backend == "codex_cli":
            client: LMStudioClient | CodexCLIClient = CodexCLIClient(self.config)
            client.get_status()
            self.logger.info(
                "判定バックエンド: Codex CLI / model=%s / reasoning=%s",
                self.config.codex_cli.model,
                self.config.codex_cli.reasoning_effort,
            )
        else:
            client = LMStudioClient(self.config)
            models = tuple(
                str(item.get("id", "")).strip()
                for item in client.get_models()
                if str(item.get("id", "")).strip()
            )
            if self.config.lmstudio.model not in models:
                client.close()
                raise ConfigError(
                    f"LM Studioモデル'{self.config.lmstudio.model}'が利用可能モデル一覧にありません"
                )
            self.logger.info(
                "判定バックエンド: LM Studio / model=%s / url=%s",
                self.config.lmstudio.model,
                self.config.lmstudio.base_url,
            )
        classifier = ImageClassifier(client, LocalRulesConfig.from_settings(self.config.rules))
        self.logger.info("判定基準: %s", self.config.rules.mode)
        sorter = ImageSorter(config=self.config, source_roots=roots)
        report = ReportBuilder(config=self.config)
        scanner = ImageScanner(
            roots=roots,
            classifier=classifier,
            sorter=sorter,
            config=self.config,
            report_builder=report,
            logger=self.logger,
        )
        return client, scanner, report

    @staticmethod
    def _counts(records: Sequence[ScanRecord]) -> dict[str, int]:
        counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0, "error": 0}
        for record in records:
            if record.status == "error":
                counts["error"] += 1
            if record.result in counts:
                counts[record.result] += 1
        return counts

    def _source_files(self, roots: Sequence[Path], scanner: ImageScanner) -> list[Path]:
        return [
            path
            for path in iter_image_paths(
                roots,
                recursive=self.config.watch.recursive,
                extensions=self.config.processing.extensions,
            )
            if not scanner._is_output_path(path)
        ]

    def run(
        self,
        *,
        force: bool = False,
        monitor: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> ReviewSummary:
        """Run an initial batch and optionally continue watching for new files."""

        with self._state_lock:
            if self._running:
                raise RuntimeError("レビュー処理は既に実行中です")
            self._running = True
        self.config.ensure_directories()
        roots = self.config.watch.paths
        client: LMStudioClient | CodexCLIClient | None = None
        records: list[ScanRecord] = []
        counts = self._counts(records)
        report = ReportBuilder(config=self.config)

        def emit(
            completed: int,
            total: int,
            current: Path | None,
            record: ScanRecord | None,
            *,
            monitoring: bool = False,
            event: str = "scan",
        ) -> None:
            if on_progress is not None:
                on_progress(
                    ReviewProgress(
                        completed=completed,
                        total=total,
                        current_path=current,
                        record=record,
                        counts=dict(counts),
                        monitoring=monitoring,
                        event=event,
                    )
                )

        try:
            client, scanner, report = self._create_pipeline(roots)
            files = self._source_files(roots, scanner)
            emit(0, len(files), None, None)
            completed = 0
            for path in files:
                if self._stop_event.is_set():
                    break
                record = scanner.process_file(path, force=force)
                completed += 1
                if record is not None:
                    records.append(record)
                    counts = self._counts(records)
                    self.logger.info("%s -> %s", path, record.result)
                emit(completed, len(files), path, record)
            report.build()

            if monitor and not self._stop_event.is_set():
                counter_lock = threading.RLock()

                def process_watched(path: Path, event: str = "created") -> None:
                    nonlocal counts
                    if self._stop_event.is_set():
                        return
                    record = scanner.process_file(path)
                    if record is None:
                        return
                    with counter_lock:
                        records.append(record)
                        counts = self._counts(records)
                        watched_count = len(records)
                    report.build()
                    self.logger.info("%s -> %s (%s)", path, record.result, event)
                    emit(watched_count, 0, path, record, monitoring=True, event=event)

                watcher = ImageWatcher(
                    roots=roots,
                    callback=process_watched,
                    config=self.config,
                    logger=self.logger,
                    include_existing=False,
                )
                with self._state_lock:
                    self._watcher = watcher
                watcher.start()
                emit(len(records), 0, None, None, monitoring=True, event="started")
                while not self._stop_event.wait(0.25):
                    if not watcher.is_running:
                        break
                watcher.stop()
            report.build()
            return ReviewSummary(
                records=tuple(records),
                counts=dict(counts),
                report_path=self.config.report_path,
                cancelled=self._stop_event.is_set(),
                monitoring=monitor,
            )
        finally:
            with self._state_lock:
                watcher, self._watcher = self._watcher, None
            if watcher is not None:
                watcher.stop()
            if client is not None:
                client.close()
            with self._state_lock:
                self._running = False


__all__ = [
    "BACKENDS",
    "OPERATIONS",
    "REASONING_EFFORTS",
    "REVIEW_MODE_DESCRIPTIONS",
    "REVIEW_MODE_LABELS",
    "BackendConnection",
    "ConfigStore",
    "DesktopSettings",
    "ReviewEngine",
    "ReviewProgress",
    "ReviewSummary",
    "check_backend_connection",
    "validate_desktop_settings",
]
