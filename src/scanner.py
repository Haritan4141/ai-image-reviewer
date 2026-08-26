"""Batch image scanning and content-hash deduplication."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence

from .models import ClassificationResult, ResultLabel
from .utils import (
    atomic_write_text,
    file_fingerprint,
    iter_image_paths,
    normalize_path,
    read_jsonl,
    relative_path,
    sha256_file,
    wait_for_file_stable,
)

try:  # classifier.py is part of the application, but keep this module useful standalone.
    from .classifier import apply_local_rules
except ImportError:  # pragma: no cover - only relevant for isolated copying of this module
    apply_local_rules = None  # type: ignore[assignment]


def _get(value: object | None, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
        return dict(result) if isinstance(result, Mapping) else {}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        result = value.model_dump()
        return dict(result) if isinstance(result, Mapping) else {}
    if is_dataclass(value):
        result = asdict(value)
        return dict(result) if isinstance(result, Mapping) else {}
    result: dict[str, Any] = {}
    for name in ("result", "confidence", "scores", "problems", "summary"):
        if hasattr(value, name):
            result[name] = getattr(value, name)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        return _json_safe(getattr(value, "value"))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(slots=True)
class ScanRecord:
    """JSONL-friendly outcome for one image."""

    source_path: str
    destination_path: str | None
    file_hash: str
    result: str
    confidence: float
    scores: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    summary: str = ""
    relative_path: str | None = None
    size_bytes: int | None = None
    modified_at: str | None = None
    processed_at: str = ""
    status: str = "processed"
    error: str | None = None

    # Compatibility aliases make reports and callers less dependent on naming.
    @property
    def path(self) -> str:
        return self.source_path

    @property
    def destination(self) -> str | None:
        return self.destination_path

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def normalize_classification(value: object, rules: object | None = None) -> ClassificationResult:
    """Convert classifier output to the project's validated model."""

    if isinstance(value, ClassificationResult):
        result = value
    else:
        payload = _as_dict(value)
        try:
            result = ClassificationResult.from_mapping(payload)
        except (TypeError, ValueError):
            result = ClassificationResult(result=ResultLabel.REVIEW, problems=["invalid classifier response"])
    if apply_local_rules is not None:
        try:
            result = apply_local_rules(result, rules)
        except (TypeError, ValueError):
            # A malformed optional rules object must never turn a safe scan
            # into a crash; the validated classifier result remains REVIEW/
            # PASS according to its own value.
            pass
    return result


class ImageScanner:
    """Scan one or more local/UNC roots and process each image once.

    The scanner is deliberately callback-based: a classifier can be the
    project's ``ImageClassifier``, a test double, or any callable accepting a
    path.  Likewise, a sorter only needs ``sort(path, classification)``.
    """

    def __init__(
        self,
        roots: str | os.PathLike[str] | Path | Sequence[str | os.PathLike[str] | Path] | object | None = None,
        classifier: object | None = None,
        sorter: object | None = None,
        *,
        config: object | None = None,
        report_builder: object | None = None,
        recursive: bool | None = None,
        stable_seconds: float | None = None,
        extensions: Iterable[str] | None = None,
        cache_path: str | os.PathLike[str] | Path | None = None,
        use_content_hash: bool | None = None,
        max_workers: int | None = None,
        logger: object | None = None,
    ) -> None:
        if config is None and roots is not None and not isinstance(roots, (str, os.PathLike, Path, list, tuple, set)):
            config = roots
            roots = None
        self.config = config
        watch = _get(config, "watch", None)
        processing = _get(config, "processing", None)
        cache = _get(config, "cache", None)
        self.roots = self._coerce_roots(roots if roots is not None else _get(watch, "paths", ()))
        self.recursive = bool(recursive if recursive is not None else _get(watch, "recursive", True))
        self.stable_seconds = float(stable_seconds if stable_seconds is not None else _get(watch, "file_stable_seconds", 1.0))
        configured_extensions = extensions if extensions is not None else _get(processing, "extensions", None)
        self.extensions = tuple(configured_extensions or (".png", ".jpg", ".jpeg", ".webp"))
        self.max_workers = max(1, int(max_workers if max_workers is not None else _get(processing, "parallel_workers", 1)))
        self.classifier = classifier
        self.sorter = sorter
        self.logger = logger
        self.rules = _get(config, "rules", None)
        output = _get(config, "output", None)
        configured_output = _get(output, "directory", None)
        sorter_output = _get(sorter, "output_dir", None)
        output_value = configured_output if configured_output is not None else sorter_output
        self.output_dir = normalize_path(output_value) if output_value is not None else None
        self.use_content_hash = bool(use_content_hash if use_content_hash is not None else _get(cache, "use_content_hash", True))
        configured_cache_path = cache_path if cache_path is not None else _get(config, "processed_cache_path", None)
        if configured_cache_path is None and cache is not None:
            directory = _get(cache, "directory", None)
            filename = _get(cache, "processed_file", "processed.json")
            if directory is not None:
                configured_cache_path = normalize_path(directory) / str(filename)
        self.cache_path = normalize_path(configured_cache_path) if configured_cache_path is not None else None
        self.report_builder = report_builder
        if self.report_builder is None and config is not None:
            # Lazy import avoids making report generation a hard dependency for
            # callers that only want a scanner in a small script.
            try:
                from .report_builder import ReportBuilder

                self.report_builder = ReportBuilder(config=config)
            except (ImportError, OSError, TypeError):
                self.report_builder = None
        self.records: list[ScanRecord] = []
        self._seen_hashes: set[str] = set()
        self._in_progress: set[str] = set()
        self._lock = threading.RLock()
        self._load_cache()

    @property
    def processed_hashes(self) -> frozenset[str]:
        """Read-only view of content fingerprints already accepted."""

        with self._lock:
            return frozenset(self._seen_hashes)

    @staticmethod
    def _coerce_roots(value: object) -> tuple[Path, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, os.PathLike, Path)):
            return (normalize_path(value),)
        return tuple(normalize_path(item) for item in value)  # type: ignore[union-attr]

    def _load_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.is_file():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        values: object = payload.get("hashes", []) if isinstance(payload, Mapping) else payload
        if isinstance(values, Mapping):
            values = values.keys()
        if isinstance(values, (list, tuple, set)):
            self._seen_hashes.update(str(item) for item in values if str(item))

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        payload = {"version": 1, "hashes": sorted(self._seen_hashes)}
        try:
            atomic_write_text(self.cache_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        except OSError:
            self._log("warning", "could not update processed-file cache: %s", self.cache_path)

    def _log(self, level: str, message: str, *args: object) -> None:
        if self.logger is None:
            return
        method = getattr(self.logger, level, None)
        if callable(method):
            method(message, *args)

    def _call_classifier(self, image: Path) -> object:
        if self.classifier is None:
            return ClassificationResult(
                result=ResultLabel.REVIEW,
                confidence=0.0,
                problems=["classifier is not configured"],
                summary="Manual review is required because no classifier is configured.",
            )
        for name in ("classify", "classify_image", "classify_file", "predict", "analyse", "analyze"):
            method = getattr(self.classifier, name, None)
            if callable(method):
                try:
                    return method(image, image_name=image.name)
                except TypeError as first_error:
                    try:
                        return method(image)
                    except TypeError:
                        raise first_error
        if callable(self.classifier):
            try:
                return self.classifier(image, image_name=image.name)
            except TypeError:
                return self.classifier(image)
        raise TypeError("classifier must be callable or expose classify/classify_image")

    def _call_sorter(self, image: Path, classification: ClassificationResult, relative: str | None) -> object:
        if self.sorter is None:
            return None
        method = getattr(self.sorter, "sort", None) or getattr(self.sorter, "sort_image", None) or getattr(self.sorter, "sort_file", None)
        if not callable(method):
            raise TypeError("sorter must expose sort/sort_image/sort_file")
        try:
            return method(image, classification, relative_path_value=relative)
        except TypeError as first_error:
            try:
                return method(image, classification, relative_path=relative)
            except TypeError:
                try:
                    return method(image, classification)
                except TypeError:
                    raise first_error

    def _root_for(self, image: Path) -> Path | None:
        for root in self.roots:
            if root.is_file() and os.path.normcase(os.fspath(root)) == os.path.normcase(os.fspath(image)):
                return root.parent
            try:
                if os.path.commonpath([os.fspath(root), os.fspath(image)]) == os.fspath(root):
                    return root
            except ValueError:
                continue
        return self.roots[0] if self.roots else None

    def _is_output_path(self, image: Path) -> bool:
        """Prevent a broad input root from recursively consuming sorted output."""

        if self.output_dir is None:
            return False
        try:
            common = os.path.commonpath([os.fspath(self.output_dir), os.fspath(image)])
            return os.path.normcase(os.path.normpath(common)) == os.path.normcase(
                os.path.normpath(os.fspath(self.output_dir))
            )
        except ValueError:
            return False

    def _record(
        self,
        image: Path,
        digest: str,
        classification: ClassificationResult,
        operation: object,
        relative: str | None,
        status: str = "processed",
        error: str | None = None,
        *,
        size: int | None = None,
        modified: str | None = None,
    ) -> ScanRecord:
        operation_dict = _as_dict(operation) if operation is not None else {}
        destination = operation_dict.get("destination_path", operation_dict.get("destination"))
        if destination is None and operation is not None:
            destination = _get(operation, "destination_path", _get(operation, "destination", None))
        if size is None or modified is None:
            try:
                # In move mode the original path no longer exists.  copy2/move
                # preserve the source size and timestamp at the destination.
                stat_path = image if image.exists() else normalize_path(destination) if destination is not None else image
                stat = stat_path.stat()
                if size is None:
                    size = int(stat.st_size)
                if modified is None:
                    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            except OSError:
                pass
        scores = classification.scores.to_dict() if hasattr(classification.scores, "to_dict") else _as_dict(classification.scores)
        return ScanRecord(
            source_path=os.fspath(image),
            destination_path=os.fspath(destination) if destination is not None else None,
            file_hash=digest,
            result=classification.result.value if hasattr(classification.result, "value") else str(classification.result),
            confidence=float(classification.confidence),
            scores={str(k): int(v) for k, v in scores.items()},
            problems=list(classification.problems),
            summary=classification.summary,
            relative_path=relative,
            size_bytes=size,
            modified_at=modified,
            processed_at=datetime.now(timezone.utc).isoformat(),
            status=status,
            error=error,
        )

    def _append_report(self, record: ScanRecord) -> None:
        if self.report_builder is None:
            return
        for name in ("append_record", "append", "write_record"):
            method = getattr(self.report_builder, name, None)
            if callable(method):
                method(record.to_dict())
                return

    def process_file(
        self,
        image: str | os.PathLike[str] | Path,
        *,
        force: bool = False,
        allow_output: bool = False,
    ) -> ScanRecord | None:
        """Process one image; return ``None`` when its content was deduplicated."""

        target = normalize_path(image)
        if not allow_output and self._is_output_path(target):
            self._log("debug", "ignoring image inside the configured output tree: %s", target)
            return None
        if target.suffix.lower() not in {
            (str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}") for ext in self.extensions
        }:
            return None
        if not wait_for_file_stable(target, stable_seconds=self.stable_seconds):
            self._log("warning", "file did not become stable before timeout: %s", target)
            return None
        try:
            digest = sha256_file(target) if self.use_content_hash else file_fingerprint(target, include_hash=False)
        except OSError as exc:
            self._log("warning", "cannot hash image %s: %s", target, exc)
            return None
        with self._lock:
            if digest in self._in_progress or (not force and digest in self._seen_hashes):
                return None
            self._in_progress.add(digest)
        try:
            # Capture source metadata before a move operation removes it.
            try:
                source_stat = target.stat()
                source_size = int(source_stat.st_size)
                source_modified = datetime.fromtimestamp(source_stat.st_mtime, tz=timezone.utc).isoformat()
            except OSError:
                source_size, source_modified = None, None
            classification = normalize_classification(self._call_classifier(target), self.rules)
            root = self._root_for(target)
            rel = relative_path(target, root) if root is not None else None
            operation = self._call_sorter(target, classification, rel)
            record = self._record(
                target,
                digest,
                classification,
                operation,
                rel,
                size=source_size,
                modified=source_modified,
            )
            with self._lock:
                self._seen_hashes.add(digest)
                self.records.append(record)
                self._save_cache()
            try:
                self._append_report(record)
            except Exception as exc:
                self._log("warning", "could not append report record for %s: %s", target, exc)
            return record
        except Exception as exc:
            self._log("exception", "failed to process image %s", target)
            record = self._record(
                target,
                digest,
                ClassificationResult(result=ResultLabel.REVIEW, problems=[f"processing error: {type(exc).__name__}"]),
                None,
                None,
                status="error",
                error=str(exc),
            )
            self.records.append(record)
            try:
                self._append_report(record)
            except Exception as report_error:
                self._log("warning", "could not append error record for %s: %s", target, report_error)
            return record
        finally:
            with self._lock:
                self._in_progress.discard(digest)

    scan_file = process_file

    def scan(
        self,
        paths: str | os.PathLike[str] | Path | Sequence[str | os.PathLike[str] | Path] | None = None,
        *,
        recursive: bool | None = None,
        force: bool = False,
        allow_output: bool = False,
    ) -> list[ScanRecord]:
        """Scan configured or explicitly supplied roots."""

        roots = self._coerce_roots(paths if paths is not None else self.roots)
        files = [
            image
            for image in iter_image_paths(
                roots,
                recursive=self.recursive if recursive is None else recursive,
                extensions=self.extensions,
            )
            if allow_output or not self._is_output_path(image)
        ]
        if self.max_workers <= 1 or len(files) <= 1:
            return [
                record
                for file in files
                if (record := self.process_file(file, force=force, allow_output=allow_output)) is not None
            ]
        results: list[ScanRecord] = []
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="image-scan") as executor:
            futures: list[Future[ScanRecord | None]] = [
                executor.submit(self.process_file, file, force=force, allow_output=allow_output)
                for file in files
            ]
            for future in as_completed(futures):
                record = future.result()
                if record is not None:
                    results.append(record)
        results.sort(key=lambda item: item.processed_at)
        return results

    scan_folder = scan
    scan_directory = scan

    def rescan_review(self, review_dir: str | os.PathLike[str] | Path | None = None) -> list[ScanRecord]:
        """Force a fresh classification of images currently in ``output/review``."""

        target = normalize_path(review_dir) if review_dir is not None else getattr(self.sorter, "review_dir", None)
        if target is None:
            target = Path("output") / "review"
        return self.scan(target, force=True, allow_output=True)


Scanner = ImageScanner


def scan_images(
    paths: str | os.PathLike[str] | Path | Sequence[str | os.PathLike[str] | Path],
    classifier: object,
    sorter: object | None = None,
    **kwargs: Any,
) -> list[ScanRecord]:
    """One-shot convenience function used by small scripts and tests."""

    return ImageScanner(paths, classifier, sorter, **kwargs).scan()


__all__ = [
    "ImageScanner",
    "ScanRecord",
    "Scanner",
    "normalize_classification",
    "scan_images",
]
