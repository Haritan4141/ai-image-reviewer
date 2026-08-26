"""Small, dependency-free helpers shared by the file processing pipeline.

The application is intended to run on Windows, but most of these helpers use
the normal :mod:`os`/``pathlib`` APIs so that unit tests can also run on
another platform.  In particular, no assumption is made that an input path
is local: a UNC path is just another path to the Windows filesystem APIs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote


IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def normalize_path(path: str | os.PathLike[str] | Path) -> Path:
    """Return an expanded path without requiring it to exist.

    ``Path.resolve`` is deliberately not used here.  Resolving a not-yet
    created UNC path can turn a useful configured path into an exception on
    some Windows versions.
    """

    if isinstance(path, Path):
        return path.expanduser()
    value = os.fspath(path)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    return Path(os.path.expandvars(value).strip()).expanduser()


def path_key(path: str | os.PathLike[str] | Path) -> str:
    """Return a case-insensitive, normalised key suitable for dictionaries."""

    return os.path.normcase(os.path.normpath(os.fspath(normalize_path(path))))


def is_supported_image(path: str | os.PathLike[str] | Path, extensions: Iterable[str] | None = None) -> bool:
    """Return whether *path* has one of the supported image extensions."""

    allowed = IMAGE_EXTENSIONS if extensions is None else {
        (str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}")
        for ext in extensions
    }
    return normalize_path(path).suffix.lower() in allowed


# Common alternate name used by integrations.
is_image_file = is_supported_image


def _iter_directory(path: Path, recursive: bool) -> Iterator[Path]:
    """Yield files below *path*, tolerating inaccessible directories."""

    if recursive:
        # ``Path.rglob`` is concise but raises if a network directory becomes
        # unavailable during enumeration.  Walking with an error callback is
        # more useful for a long-running UNC watcher.
        def on_error(_: OSError) -> None:
            return None

        try:
            for root, _dirs, files in os.walk(path, onerror=on_error):
                for name in files:
                    yield Path(root) / name
        except (OSError, PermissionError):
            return
    else:
        try:
            for child in path.iterdir():
                try:
                    if child.is_file():
                        yield child
                except OSError:
                    continue
        except (OSError, PermissionError):
            return


def iter_image_files(
    root: str | os.PathLike[str] | Path,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield supported images under a file or directory in stable order.

    A missing/inaccessible root produces an empty iterator.  This is useful
    for watchers where a disconnected UNC share may come back later.
    """

    path = normalize_path(root)
    if path.is_file():
        if is_supported_image(path, extensions):
            yield path
        return
    if not path.is_dir():
        return
    candidates = (item for item in _iter_directory(path, recursive) if is_supported_image(item, extensions))
    # Sort before yielding so reports and tests remain deterministic, even
    # when Windows returns directory entries in a different order.
    yield from sorted(candidates, key=lambda item: os.fspath(item).casefold())


def iter_image_paths(
    roots: str | os.PathLike[str] | Path | Sequence[str | os.PathLike[str] | Path],
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield images below one or more roots, de-duplicated by path."""

    if isinstance(roots, (str, os.PathLike, Path)):
        values: Sequence[str | os.PathLike[str] | Path] = (roots,)
    else:
        values = roots
    seen: set[str] = set()
    for root in values:
        for item in iter_image_files(root, recursive=recursive, extensions=extensions):
            key = path_key(item)
            if key not in seen:
                seen.add(key)
                yield item


def relative_path(path: str | os.PathLike[str] | Path, root: str | os.PathLike[str] | Path) -> str:
    """Return a portable relative path, or the filename when roots differ."""

    source = os.fspath(normalize_path(path))
    base = os.fspath(normalize_path(root))
    try:
        value = os.path.relpath(source, base)
    except (ValueError, OSError):
        value = os.path.basename(source)
    return value.replace("\\", "/")


def wait_for_file_stable(
    path: str | os.PathLike[str] | Path,
    stable_seconds: float = 1.0,
    timeout_seconds: float = 60.0,
    poll_interval: float = 0.25,
) -> bool:
    """Wait until a file exists and its size/mtime stay unchanged.

    A stable file is also opened for a small read, which catches a transient
    network/share error.  The function returns ``False`` on timeout instead
    of raising, allowing a watcher to retry on the next poll.
    """

    target = normalize_path(path)
    timeout = max(0.0, float(timeout_seconds))
    stable_for = max(0.0, float(stable_seconds))
    interval = max(0.01, float(poll_interval))
    started = time.monotonic()
    previous: tuple[int, int] | None = None
    unchanged_since: float | None = None

    while True:
        now = time.monotonic()
        if now - started > timeout:
            return False
        try:
            stat = target.stat()
            signature = (int(stat.st_size), int(getattr(stat, "st_mtime_ns", stat.st_mtime * 1_000_000_000)))
            with target.open("rb") as stream:
                stream.read(1)
        except (FileNotFoundError, PermissionError, OSError):
            previous = None
            unchanged_since = None
            time.sleep(min(interval, max(0.01, timeout - (time.monotonic() - started))))
            continue

        if signature != previous:
            previous = signature
            unchanged_since = time.monotonic()
        elif unchanged_since is None:
            unchanged_since = time.monotonic()
        elif time.monotonic() - unchanged_since >= stable_for:
            return True

        if stable_for <= 0:
            return True
        time.sleep(min(interval, max(0.01, timeout - (time.monotonic() - started))))


# Alternate spelling retained for callers that use "settled" terminology.
wait_until_stable = wait_for_file_stable


def sha256_file(path: str | os.PathLike[str] | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""

    digest = hashlib.sha256()
    with normalize_path(path).open("rb") as stream:
        while True:
            chunk = stream.read(max(1024, int(chunk_size)))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


file_hash = sha256_file
hash_file = sha256_file


def quick_fingerprint(path: str | os.PathLike[str] | Path) -> tuple[int, int]:
    """Return a cheap fingerprint used by the polling watcher."""

    stat = normalize_path(path).stat()
    return int(stat.st_size), int(getattr(stat, "st_mtime_ns", stat.st_mtime * 1_000_000_000))


def file_fingerprint(path: str | os.PathLike[str] | Path, include_hash: bool = True) -> str:
    """Return a stable fingerprint for cache/deduplication purposes."""

    target = normalize_path(path)
    stat = target.stat()
    digest = sha256_file(target) if include_hash else ""
    return f"{int(stat.st_size)}:{int(getattr(stat, 'st_mtime_ns', stat.st_mtime * 1_000_000_000))}:{digest}"


def ensure_directory(path: str | os.PathLike[str] | Path) -> Path:
    """Create *path* (including parents) and return it as a ``Path``."""

    target = normalize_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(name: str, fallback: str = "image") -> str:
    """Make a basename safe for a Windows destination directory."""

    cleaned = _INVALID_FILENAME_CHARS.sub("_", str(name)).strip().rstrip(".")
    if not cleaned:
        cleaned = fallback
    stem = cleaned.rsplit(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def unique_destination_path(
    directory: str | os.PathLike[str] | Path,
    filename: str,
    *,
    source: str | os.PathLike[str] | Path | None = None,
) -> Path:
    """Choose a destination filename without overwriting an existing file."""

    target_dir = ensure_directory(directory)
    original = target_dir / safe_filename(Path(filename).name)
    source_key = path_key(source) if source is not None else None
    if not original.exists() or (source_key is not None and path_key(original) == source_key):
        return original

    stem, suffix = original.stem, original.suffix
    for index in range(1, 100_000):
        candidate = target_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find a free destination name for {original}")


def transfer_file(
    source: str | os.PathLike[str] | Path,
    destination_dir: str | os.PathLike[str] | Path,
    mode: str = "copy",
    *,
    destination_name: str | None = None,
) -> Path:
    """Copy or move *source* to a collision-free destination.

    ``copy2`` preserves generated-image timestamps.  ``shutil.move`` also
    works across volumes/UNC shares (falling back to copy-and-delete as needed
    by Python), which is useful for a configured move operation.
    """

    original = normalize_path(source)
    if not original.is_file():
        raise FileNotFoundError(original)
    mode_value = str(mode).lower().strip()
    if mode_value not in {"copy", "move"}:
        raise ValueError("mode must be 'copy' or 'move'")
    target_dir = ensure_directory(destination_dir)
    destination = unique_destination_path(target_dir, destination_name or original.name, source=original)
    if path_key(destination) == path_key(original):
        return original
    if mode_value == "move":
        return Path(shutil.move(os.fspath(original), os.fspath(destination)))
    shutil.copy2(os.fspath(original), os.fspath(destination))
    return destination


def path_to_file_url(path: str | os.PathLike[str] | Path) -> str:
    """Return a browser-safe file URL for a local or UNC path."""

    target = normalize_path(path)
    try:
        return target.resolve(strict=False).as_uri()
    except (ValueError, OSError):
        # ``Path.as_uri`` is strict about the platform's path syntax.  This
        # fallback is primarily for cross-platform tests with Windows paths.
        text = os.fspath(target).replace("\\", "/")
        if text.startswith("//"):
            return "file:" + quote(text, safe="/:@")
        return "file:///" + quote(text.lstrip("/"), safe="/:@")


def json_default(value: Any) -> Any:
    """JSON encoder hook for paths, enums, dataclasses, and datetimes."""

    if isinstance(value, Path):
        return os.fspath(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def atomic_write_text(path: str | os.PathLike[str] | Path, text: str, encoding: str = "utf-8") -> Path:
    """Atomically replace a text file in its containing directory."""

    target = normalize_path(path)
    ensure_directory(target.parent)
    handle = tempfile.NamedTemporaryFile("w", encoding=encoding, dir=target.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(os.fspath(temporary), os.fspath(target))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def read_jsonl(path: str | os.PathLike[str] | Path) -> list[dict[str, Any]]:
    """Read valid JSON objects from a JSONL file, ignoring malformed lines."""

    target = normalize_path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with target.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping):
                    records.append(dict(value))
    except OSError:
        return records
    return records


__all__ = [
    "IMAGE_EXTENSIONS",
    "atomic_write_text",
    "ensure_directory",
    "file_fingerprint",
    "file_hash",
    "hash_file",
    "is_image_file",
    "is_supported_image",
    "iter_image_files",
    "iter_image_paths",
    "json_default",
    "normalize_path",
    "path_key",
    "path_to_file_url",
    "quick_fingerprint",
    "read_jsonl",
    "relative_path",
    "safe_filename",
    "sha256_file",
    "transfer_file",
    "unique_destination_path",
    "wait_for_file_stable",
    "wait_until_stable",
]
