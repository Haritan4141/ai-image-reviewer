from __future__ import annotations

from pathlib import Path

import pytest

from src.file_watcher import ImageWatcher, PollingWatcher


def test_polling_watcher_dispatches_new_and_changed_supported_images(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"one")
    events: list[tuple[Path, str]] = []

    def callback(path: Path, event: str = "created") -> None:
        events.append((path, event))

    watcher = PollingWatcher(tmp_path, callback, include_existing=True, stable_seconds=0, interval=0.05)
    watcher._poll_once(initial=True)
    watcher._poll_once()
    # Use a different size as well as different bytes so coarse filesystem
    # mtime resolution cannot hide the modification on Windows.
    image.write_bytes(b"two-two")
    watcher._poll_once()

    assert events == [(image, "created"), (image, "modified")]
    assert watcher.active_backend == "polling"


def test_watcher_filters_unsupported_files_and_forgets_deleted_paths(tmp_path: Path) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("ignore", encoding="utf-8")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    seen: list[Path] = []
    watcher = ImageWatcher(tmp_path, seen.append, include_existing=True, stable_seconds=0)

    watcher._poll_once(initial=True)
    assert seen == [image]
    image.unlink()
    watcher._poll_once()
    image.write_bytes(b"recreated")
    watcher._poll_once()
    assert seen == [image, image]


def test_watcher_start_stop_and_invalid_backend() -> None:
    watcher = ImageWatcher([], backend="polling", interval=0.05)

    assert not watcher.is_running
    watcher.start()
    assert watcher.is_running
    watcher.stop()
    assert not watcher.is_running

    with pytest.raises(ValueError):
        ImageWatcher([], backend="unknown")
