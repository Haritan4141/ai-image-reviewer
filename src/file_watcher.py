"""Watch local and UNC image folders with watchdog or a polling fallback."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .utils import iter_image_paths, normalize_path, path_key, quick_fingerprint, is_supported_image, wait_for_file_stable

try:  # Optional dependency; polling remains fully functional without it.
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # pragma: no cover - exercised only without optional dependency
    FileSystemEventHandler = object  # type: ignore[misc,assignment]
    Observer = None  # type: ignore[assignment]


Callback = Callable[..., Any]


def _get(value: object | None, name: str, default: Any = None) -> Any:
    """Read a setting from either a mapping or an attribute-based config."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class _WatchdogHandler(FileSystemEventHandler):
    def __init__(self, owner: "ImageWatcher") -> None:
        super().__init__()
        self.owner = owner

    def on_created(self, event: Any) -> None:
        if not getattr(event, "is_directory", False):
            self.owner._dispatch(Path(event.src_path), "created")

    def on_modified(self, event: Any) -> None:
        if not getattr(event, "is_directory", False):
            self.owner._dispatch(Path(event.src_path), "modified")

    def on_moved(self, event: Any) -> None:
        if not getattr(event, "is_directory", False):
            self.owner._dispatch(Path(event.dest_path), "moved")


class ImageWatcher:
    """Dispatch new/changed images to a scanner or callback.

    Parameters are intentionally permissive so the watcher can be used with
    ``AppConfig`` directly or with a small standalone script.  ``backend`` may
    be ``"polling"``, ``"watchdog"``, or ``"auto"``.  If watchdog is not
    installed, ``auto`` and ``watchdog`` both fall back to polling.
    """

    def __init__(
        self,
        roots: str | os.PathLike[str] | Path | Sequence[str | os.PathLike[str] | Path] | object | None = None,
        callback: Callback | None = None,
        *,
        scanner: object | None = None,
        config: object | None = None,
        recursive: bool | None = None,
        backend: str | None = None,
        mode: str | None = None,
        interval: float | None = None,
        polling_interval: float | None = None,
        stable_seconds: float | None = None,
        include_existing: bool = True,
        extensions: Iterable[str] | None = None,
        logger: object | None = None,
    ) -> None:
        if config is None and roots is not None and not isinstance(roots, (str, os.PathLike, Path, list, tuple, set)):
            config = roots
            roots = None
        self.config = config
        watch = _get(config, "watch", None)
        processing = _get(config, "processing", None)
        self.roots = self._coerce_roots(roots if roots is not None else _get(watch, "paths", ()))
        self.recursive = bool(recursive if recursive is not None else _get(watch, "recursive", True))
        self.backend = str(backend or mode or _get(watch, "mode", "polling")).lower().strip()
        if self.backend not in {"auto", "polling", "watchdog"}:
            raise ValueError("watch backend/mode must be 'auto', 'polling', or 'watchdog'")
        configured_interval = interval if interval is not None else polling_interval
        self.interval = max(0.05, float(configured_interval if configured_interval is not None else _get(watch, "polling_interval_seconds", 5.0)))
        self.stable_seconds = max(0.0, float(stable_seconds if stable_seconds is not None else _get(watch, "file_stable_seconds", 1.0)))
        self.extensions = tuple(extensions or _get(processing, "extensions", (".png", ".jpg", ".jpeg", ".webp")))
        self.callback = callback
        self.scanner = scanner
        if self.callback is None and scanner is not None:
            self.callback = getattr(scanner, "process_file", None) or getattr(scanner, "scan_file", None)
        self.include_existing = include_existing
        self.logger = logger
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer: Any = None
        self._seen: dict[str, tuple[int, int]] = {}
        self._lock = threading.RLock()
        self._running = False

    @staticmethod
    def _coerce_roots(value: object) -> tuple[Path, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, os.PathLike, Path)):
            return (normalize_path(value),)
        return tuple(normalize_path(item) for item in value)  # type: ignore[union-attr]

    def _log(self, level: str, message: str, *args: object) -> None:
        method = getattr(self.logger, level, None) if self.logger is not None else None
        if callable(method):
            method(message, *args)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def active_backend(self) -> str:
        if self.backend == "auto":
            return "watchdog" if Observer is not None else "polling"
        if self.backend == "watchdog" and Observer is None:
            return "polling"
        return self.backend

    def _callback(self, path: Path, event: str) -> Any:
        if self.callback is None:
            return None
        try:
            return self.callback(path, event=event)
        except TypeError as first_error:
            try:
                return self.callback(path, event)
            except TypeError:
                try:
                    return self.callback(path)
                except TypeError:
                    raise first_error

    def _dispatch(self, path: Path, event: str = "created") -> None:
        if not is_supported_image(path, self.extensions):
            return
        try:
            signature = quick_fingerprint(path)
        except (FileNotFoundError, PermissionError, OSError):
            return
        key = path_key(path)
        with self._lock:
            if self._seen.get(key) == signature:
                return
            self._seen[key] = signature
        if self.stable_seconds and not wait_for_file_stable(path, stable_seconds=self.stable_seconds):
            # Leave it eligible for the next polling/event pass; a writer may
            # simply need a little more time to finish a large network copy.
            with self._lock:
                if self._seen.get(key) == signature:
                    self._seen.pop(key, None)
            self._log("warning", "watched file was not stable before timeout: %s", path)
            return
        try:
            self._callback(path, event)
        except Exception:
            with self._lock:
                if self._seen.get(key) == signature:
                    self._seen.pop(key, None)
            self._log("exception", "watch callback failed for %s", path)

    def _poll_once(self, initial: bool = False) -> None:
        current: set[str] = set()
        for path in iter_image_paths(self.roots, recursive=self.recursive, extensions=self.extensions):
            key = path_key(path)
            current.add(key)
            try:
                signature = quick_fingerprint(path)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            with self._lock:
                old = self._seen.get(key)
                if initial and not self.include_existing and old is None:
                    # Establish a baseline without treating files that were
                    # already present at startup as newly created.
                    self._seen[key] = signature
                    continue
            if (initial and self.include_existing) or old != signature:
                self._dispatch(path, "created" if old is None else "modified")
        # Forget disappeared files so that a recreated file is dispatched.
        with self._lock:
            for key in list(self._seen):
                if key not in current:
                    self._seen.pop(key, None)

    def _poll_loop(self) -> None:
        self._poll_once(initial=True)
        while not self._stop_event.wait(self.interval):
            self._poll_once()

    def _start_watchdog(self) -> bool:
        if Observer is None:
            return False
        observer = Observer()
        handler = _WatchdogHandler(self)
        scheduled = False
        for root in self.roots:
            if root.is_file():
                root = root.parent
            try:
                observer.schedule(handler, os.fspath(root), recursive=self.recursive)
                scheduled = True
            except (OSError, ValueError):
                self._log("warning", "cannot watch path: %s", root)
        if not scheduled:
            return False
        observer.start()
        self._observer = observer
        return True

    def start(self, *, blocking: bool = False) -> "ImageWatcher":
        """Start monitoring; set ``blocking=True`` for a CLI-style loop."""

        if self._running:
            if blocking:
                self.wait()
            return self
        self._stop_event.clear()
        self._running = True
        active = self.active_backend
        if active == "watchdog" and self._start_watchdog():
            pass
        else:
            self._thread = threading.Thread(target=self._poll_loop, name="image-poll-watcher", daemon=True)
            self._thread.start()
        if blocking:
            self.wait()
        return self

    def run_forever(self) -> None:
        """Start and block until :meth:`stop` is called or Ctrl+C is pressed."""

        self.start()
        try:
            self.wait()
        except KeyboardInterrupt:
            self.stop()

    run = run_forever
    watch = run_forever

    def wait(self, timeout: float | None = None) -> None:
        """Wait for the polling thread/observer to finish."""

        if self._thread is not None:
            self._thread.join(timeout)
        elif self._observer is not None:
            self._observer.join(timeout)

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        observer, self._observer = self._observer, None
        if observer is not None:
            observer.stop()
            observer.join(timeout=max(1.0, self.interval + 1.0))
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.interval + 1.0))
        self._running = False

    close = stop

    def __enter__(self) -> "ImageWatcher":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()


class PollingWatcher(ImageWatcher):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["backend"] = "polling"
        super().__init__(*args, **kwargs)


class WatchdogWatcher(ImageWatcher):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["backend"] = "watchdog"
        super().__init__(*args, **kwargs)


FileWatcher = ImageWatcher
FolderWatcher = ImageWatcher


def watch_folder(
    roots: str | os.PathLike[str] | Path | Sequence[str | os.PathLike[str] | Path],
    callback: Callback,
    **kwargs: Any,
) -> ImageWatcher:
    """Create and start a watcher, returning it so callers can stop it later."""

    return ImageWatcher(roots, callback, **kwargs).start()


__all__ = [
    "FileWatcher",
    "FolderWatcher",
    "ImageWatcher",
    "PollingWatcher",
    "WatchdogWatcher",
    "watch_folder",
]
