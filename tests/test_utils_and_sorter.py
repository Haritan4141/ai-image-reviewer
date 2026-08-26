from __future__ import annotations

import json
from pathlib import Path

from src.sorter import ImageSorter, coerce_label
from src.utils import (
    file_fingerprint,
    iter_image_files,
    iter_image_paths,
    path_to_file_url,
    read_jsonl,
    safe_filename,
    sha256_file,
    transfer_file,
    wait_for_file_stable,
)


def test_image_enumeration_is_recursive_filtered_and_deduplicated(tmp_path: Path) -> None:
    root = tmp_path / "images"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "b.PNG").write_bytes(b"b")
    (nested / "a.jpg").write_bytes(b"a")
    (nested / "ignore.txt").write_text("x", encoding="utf-8")

    recursive = list(iter_image_files(root, recursive=True))
    shallow = list(iter_image_files(root, recursive=False))
    deduped = list(iter_image_paths([root, nested], recursive=True))

    assert [item.name for item in recursive] == ["b.PNG", "a.jpg"]
    assert [item.name for item in shallow] == ["b.PNG"]
    assert len(deduped) == 2


def test_hash_fingerprint_and_stability(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"sample")

    assert len(sha256_file(image)) == 64
    assert file_fingerprint(image, include_hash=True).endswith(sha256_file(image))
    assert wait_for_file_stable(image, stable_seconds=0, timeout_seconds=0.5, poll_interval=0.01)


def test_safe_filename_and_transfer_file_avoid_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "source" / "source-file.png"
    destination = tmp_path / "destination"
    source.parent.mkdir()
    source.write_bytes(b"one")

    assert safe_filename("CON") == "_CON"
    assert safe_filename("a:b?.png") == "a_b_.png"
    first = transfer_file(source, destination, "copy")
    second = transfer_file(source, destination, "copy")

    assert first.read_bytes() == b"one"
    assert second.name != first.name
    assert source.is_file()


def test_read_jsonl_ignores_invalid_lines_and_path_url_is_usable(tmp_path: Path) -> None:
    log = tmp_path / "results.jsonl"
    log.write_text('{"result":"PASS"}\nnot-json\n\n[1,2]\n{"result":"REVIEW"}\n', encoding="utf-8")

    assert read_jsonl(log) == [{"result": "PASS"}, {"result": "REVIEW"}]
    assert path_to_file_url(log).startswith("file:")


def test_sorter_copies_into_result_bucket_preserving_relative_path(tmp_path: Path) -> None:
    source_root = tmp_path / "incoming"
    source = source_root / "batch-1" / "image.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image")
    sorter = ImageSorter(
        tmp_path / "output",
        mode="copy",
        preserve_relative_paths=True,
        source_roots=[source_root],
    )

    result = sorter.sort(source, {"result": "PASS"})

    assert result.result == "PASS"
    assert Path(result.destination_path) == tmp_path / "output/pass/batch-1/image.png"
    assert Path(result.destination_path).read_bytes() == b"image"
    assert source.is_file()


def test_sorter_renames_collision_and_move_removes_source(tmp_path: Path) -> None:
    source = tmp_path / "image.jpg"
    source.write_bytes(b"image")
    sorter = ImageSorter(tmp_path / "output", mode="copy")

    first = sorter.sort(source, {"result": "REVIEW"})
    second = sorter.sort(source, {"result": "REVIEW"})

    assert Path(first.destination_path).exists()
    assert Path(second.destination_path).exists()
    assert Path(first.destination_path).name != Path(second.destination_path).name

    movable = tmp_path / "move.png"
    movable.write_bytes(b"move")
    moved = ImageSorter(tmp_path / "output", mode="move").sort(movable, {"result": "FAIL"})

    assert not movable.exists()
    assert Path(moved.destination_path).exists()


def test_unknown_sort_label_is_safe_review_bucket() -> None:
    assert coerce_label("PASS") == "PASS"
    assert coerce_label("not-valid") == "REVIEW"
