"""Move/copy classified images into PASS/REVIEW/FAIL directories."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

from .utils import (
    ensure_directory,
    normalize_path,
    path_key,
    relative_path,
    safe_filename,
    transfer_file,
    unique_destination_path,
)


_BUCKETS = {"PASS": "pass", "REVIEW": "review", "FAIL": "fail"}


def _value(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def coerce_label(value: object) -> str:
    """Coerce an arbitrary classifier label to a safe output bucket."""

    if hasattr(value, "value"):
        value = getattr(value, "value")
    label = str(value or "REVIEW").strip().upper().strip("`* _-.")
    return label if label in _BUCKETS else "REVIEW"


@dataclass(slots=True)
class SortResult:
    """Description of one copy/move operation."""

    source_path: str
    destination_path: str
    result: str
    operation: str
    relative_path: str | None = None
    collision_renamed: bool = False

    @property
    def destination(self) -> str:
        return self.destination_path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImageSorter:
    """Sort images while keeping the source tree safe from overwrites.

    ``config`` may be an :class:`src.config.AppConfig`, an output section, or
    a plain mapping.  For direct use, ``output_dir`` and ``mode`` are enough.
    By default the source tree is flattened under each result directory; when
    ``preserve_relative_paths`` is enabled, relative source roots are kept and
    same-name files in different folders remain easy to identify.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str] | Path | object = "output",
        mode: str = "copy",
        *,
        operation: str | None = None,
        preserve_relative_paths: bool | None = None,
        source_roots: Sequence[str | os.PathLike[str] | Path] | None = None,
        config: object | None = None,
    ) -> None:
        # Accept ImageSorter(config) as a convenient integration path.
        candidate = config if config is not None else output_dir
        if config is None and not isinstance(output_dir, (str, os.PathLike, Path)):
            config = output_dir
            candidate = output_dir

        output_section = _value(candidate, "output", candidate)
        configured_dir = _value(output_section, "directory", None)
        if configured_dir is not None:
            output_dir = configured_dir
        elif not isinstance(output_dir, (str, os.PathLike, Path)):
            output_dir = "output"

        configured_operation = _value(output_section, "operation", None)
        if operation is not None:
            mode = operation
        elif configured_operation is not None:
            mode = configured_operation
        self.output_dir = normalize_path(output_dir)  # type: ignore[arg-type]
        self.mode = str(mode).lower().strip()
        if self.mode not in {"copy", "move"}:
            raise ValueError("operation/mode must be 'copy' or 'move'")

        configured_preserve = _value(output_section, "preserve_relative_paths", None)
        self.preserve_relative_paths = (
            bool(preserve_relative_paths)
            if preserve_relative_paths is not None
            else bool(configured_preserve) if configured_preserve is not None else False
        )

        roots = source_roots
        if roots is None and config is not None:
            watch_section = _value(config, "watch", None)
            roots = _value(watch_section, "paths", None) if watch_section is not None else None
        self.source_roots = tuple(normalize_path(item) for item in (roots or ()))
        self._lock = RLock()
        for bucket in _BUCKETS.values():
            ensure_directory(self.output_dir / bucket)

    @property
    def pass_dir(self) -> Path:
        return self.output_dir / "pass"

    @property
    def review_dir(self) -> Path:
        return self.output_dir / "review"

    @property
    def fail_dir(self) -> Path:
        return self.output_dir / "fail"

    def bucket_for(self, classification: object) -> str:
        return _BUCKETS[coerce_label(_value(classification, "result", classification))]

    def _relative_destination_dir(self, source: Path, bucket: str, relative: str | Path | None) -> Path:
        destination = self.output_dir / bucket
        if not self.preserve_relative_paths:
            return destination
        relative_value: str | None = os.fspath(relative) if relative is not None else None
        if relative_value is None:
            for root in self.source_roots:
                try:
                    relative_value = os.path.relpath(os.fspath(source), os.fspath(root))
                except (ValueError, OSError):
                    continue
                if relative_value != os.fspath(source) and not relative_value.startswith(".." + os.sep):
                    break
        if relative_value is None or relative_value in {".", ""}:
            return destination
        parent = Path(relative_value).parent
        # A path supplied by a classifier is not allowed to escape output.
        if str(parent) in {"", "."} or any(part == ".." for part in parent.parts):
            return destination
        return destination / parent

    def destination_for(
        self,
        source: str | os.PathLike[str] | Path,
        classification: object,
        *,
        relative_path_value: str | Path | None = None,
    ) -> Path:
        """Return the next free destination for an image without copying it."""

        original = normalize_path(source)
        bucket = self.bucket_for(classification)
        target_dir = self._relative_destination_dir(original, bucket, relative_path_value)
        return unique_destination_path(target_dir, safe_filename(original.name), source=original)

    def sort(
        self,
        source: str | os.PathLike[str] | Path,
        classification: object = None,
        *,
        result: object | None = None,
        relative_path_value: str | Path | None = None,
        relative_path: str | Path | None = None,
        mode: str | None = None,
    ) -> SortResult:
        """Copy or move ``source`` according to the classification result.

        ``result=`` and ``relative_path=`` are accepted as keyword aliases to
        make this method convenient for scanner and CLI integrations.
        """

        if classification is None:
            classification = result
        if classification is None:
            classification = {"result": "REVIEW"}
        original = normalize_path(source)
        operation = str(mode or self.mode).lower().strip()
        if operation not in {"copy", "move"}:
            raise ValueError("operation/mode must be 'copy' or 'move'")
        relative_value = relative_path_value if relative_path_value is not None else relative_path
        with self._lock:
            destination = self.destination_for(original, classification, relative_path_value=relative_value)
            collision_renamed = destination.name != safe_filename(original.name)
            # The helper returns the exact source for the unusual case where a
            # caller points output at the source's current directory.
            if path_key(destination) == path_key(original):
                actual = original
            else:
                actual = transfer_file(original, destination.parent, operation, destination_name=destination.name)
        relative_display = None
        if relative_value is not None:
            relative_display = os.fspath(relative_value).replace("\\", "/")
        return SortResult(
            source_path=os.fspath(original),
            destination_path=os.fspath(actual),
            result=coerce_label(_value(classification, "result", classification)),
            operation=operation,
            relative_path=relative_display,
            collision_renamed=collision_renamed,
        )

    sort_image = sort
    sort_file = sort


# Names used by older examples and simple integrations.
Sorter = ImageSorter
FileSorter = ImageSorter


def sort_image(
    source: str | os.PathLike[str] | Path,
    classification: object,
    output_dir: str | os.PathLike[str] | Path = "output",
    mode: str = "copy",
    **kwargs: Any,
) -> SortResult:
    """One-shot convenience wrapper around :class:`ImageSorter`."""

    return ImageSorter(output_dir, mode, **kwargs).sort(source, classification)


__all__ = [
    "FileSorter",
    "ImageSorter",
    "SortResult",
    "Sorter",
    "coerce_label",
    "sort_image",
]
