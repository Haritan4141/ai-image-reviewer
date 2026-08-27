from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.scanner import ImageScanner, normalize_classification, scan_images
from src.models import ResultLabel
from src.sorter import ImageSorter


def _payload(label: str = "PASS", confidence: float = 0.95) -> dict[str, object]:
    return {
        "result": label,
        "confidence": confidence,
        "scores": {"anatomy": 9, "hands": 9, "face": 9, "artifacts": 9, "composition": 9},
        "problems": [],
        "summary": "ok",
    }


class _Classifier:
    def __init__(self, label: str = "PASS") -> None:
        self.label = label
        self.calls: list[Path] = []

    def classify(self, image: Path, *, image_name: str | None = None) -> dict[str, object]:
        self.calls.append(image)
        return _payload(self.label)


class _Sorter:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str | None]] = []
        self.review_dir = Path("output/review")

    def sort(self, image: Path, classification: object, *, relative_path_value: str | None = None) -> dict[str, str]:
        self.calls.append((image, relative_path_value))
        return {"destination_path": f"output/{getattr(classification.result, 'value', classification.result).lower()}/{image.name}"}


def test_scanner_processes_once_by_content_hash_and_persists_cache(tmp_path: Path) -> None:
    root = tmp_path / "incoming"
    root.mkdir()
    image = root / "batch" / "one.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    cache = tmp_path / "cache" / "processed.json"
    classifier = _Classifier()
    sorter = _Sorter()
    scanner = ImageScanner(
        root,
        classifier,
        sorter,
        stable_seconds=0,
        cache_path=cache,
        use_content_hash=True,
    )

    first = scanner.process_file(image)
    duplicate = scanner.process_file(image)

    assert first is not None
    assert first.result == "PASS"
    assert first.model_result == "PASS"
    assert first.final_result == "PASS"
    assert first.decision_source == "model"
    assert first.review_mode == "standard"
    assert first.relative_path == "batch/one.png"
    assert duplicate is None
    assert classifier.calls == [image]
    assert cache.is_file()

    restarted = ImageScanner(root, _Classifier(), _Sorter(), stable_seconds=0, cache_path=cache)
    assert restarted.process_file(image) is None


def test_scanner_skips_unsupported_files_and_can_force_rescan(tmp_path: Path) -> None:
    root = tmp_path / "incoming"
    root.mkdir()
    image = root / "one.jpg"
    image.write_bytes(b"image")
    text = root / "notes.txt"
    text.write_text("ignore", encoding="utf-8")
    classifier = _Classifier()
    scanner = ImageScanner(root, classifier, stable_seconds=0, cache_path=tmp_path / "cache.json")

    records = scanner.scan()
    forced = scanner.process_file(image, force=True)
    assert len(records) == 1
    assert forced is not None
    assert len(classifier.calls) == 2
    assert scanner.process_file(text) is None


def test_scanner_records_errors_as_review_instead_of_crashing(tmp_path: Path) -> None:
    image = tmp_path / "broken.webp"
    image.write_bytes(b"image")

    class BrokenClassifier:
        def classify(self, image: Path, *, image_name: str | None = None) -> object:
            raise RuntimeError("API unavailable")

    scanner = ImageScanner(image.parent, BrokenClassifier(), stable_seconds=0)
    record = scanner.process_file(image)

    assert record is not None
    assert record.status == "error"
    assert record.result == "REVIEW"
    assert record.error == "API unavailable"


def test_normalize_classification_applies_safe_default_for_invalid_payload() -> None:
    result = normalize_classification({"result": "not-a-label"})

    assert result.result is ResultLabel.REVIEW
    assert result.problems


def test_scan_images_convenience_function(tmp_path: Path) -> None:
    image = tmp_path / "one.png"
    image.write_bytes(b"image")
    classifier = _Classifier("REVIEW")

    records = scan_images(
        tmp_path,
        classifier,
        stable_seconds=0,
        cache_path=tmp_path / "cache.json",
    )

    assert len(records) == 1
    assert records[0].result == "REVIEW"


def test_scanner_does_not_consume_configured_output_tree(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    output = incoming / "output"
    (output / "review").mkdir(parents=True)
    (incoming / "new.png").write_bytes(b"new")
    (output / "review" / "already-sorted.png").write_bytes(b"sorted")
    classifier = _Classifier()
    scanner = ImageScanner(
        incoming,
        classifier,
        config={"output": {"directory": output}},
        report_builder=False,
        stable_seconds=0,
        cache_path=tmp_path / "cache.json",
    )

    records = scanner.scan()

    assert [Path(record.source_path).name for record in records] == ["new.png"]
    assert scanner.process_file(output / "review" / "already-sorted.png") is None
    assert scanner.process_file(output / "review" / "already-sorted.png", force=True, allow_output=True) is not None


def test_move_mode_preserves_source_metadata_in_record(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    image = incoming / "one.png"
    payload = b"image bytes"
    image.write_bytes(payload)
    sorter = ImageSorter(tmp_path / "output", mode="move")
    scanner = ImageScanner(incoming, _Classifier(), sorter, stable_seconds=0)

    record = scanner.process_file(image)

    assert record is not None
    assert record.size_bytes == len(payload)
    assert record.modified_at is not None
    assert not image.exists()
    assert Path(record.destination_path or "").is_file()


def test_scanner_retains_crop_checks_and_pipeline_metadata_without_raw_payload(tmp_path: Path) -> None:
    image = tmp_path / "one.png"
    image.write_bytes(b"image")
    scanner = ImageScanner(image.parent, _Classifier(), stable_seconds=0)
    classification = SimpleNamespace(
        result=ResultLabel.REVIEW,
        model_result=ResultLabel.PASS,
        confidence=0.72,
        scores={"hands": 8},
        problems=["hand requires review"],
        summary="full result",
        decision_source="crop_merge",
        review_mode="balanced",
        low_scores={},
        keyword_hits={},
        rule_reasons=["crop review"],
        pipeline_stage="full_plus_crop",
        pipeline_version="crop-v1",
        crop_mode="balanced",
        full_result_before_merge="PASS",
        crop_checks=[
            {
                "kind": "hand",
                "index": 0,
                "box": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
                "result": "REVIEW",
                "confidence": 0.4,
                "problems": ["finger ambiguity"],
                "summary": "needs review",
                "detector_name": "fallback",
                "detector_confidence": 0.5,
                "raw": {"secret": "must not be persisted"},
            }
        ],
    )

    record = scanner._record(image, "digest", classification, None, "one.png")

    assert record.result == "REVIEW"
    assert record.model_result == "PASS"
    assert record.pipeline_stage == "full_plus_crop"
    assert record.pipeline_version == "crop-v1"
    assert record.crop_mode == "balanced"
    assert record.full_result_before_merge == "PASS"
    assert record.crop_checks[0]["kind"] == "hand"
    assert record.crop_checks[0]["box"] == [0.1, 0.2, 0.8, 0.9]
    assert "raw" not in record.crop_checks[0]


def test_scanner_excludes_crop_cache_even_when_crop_recheck_is_disabled(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    crop_cache = incoming / "cache" / "crops"
    crop_cache.mkdir(parents=True)
    good = incoming / "good.png"
    generated = crop_cache / "generated.png"
    good.write_bytes(b"good")
    generated.write_bytes(b"generated")
    classifier = _Classifier()
    scanner = ImageScanner(
        incoming,
        classifier,
        config={"crop_recheck": {"crop_cache_dir": crop_cache}},
        report_builder=False,
        stable_seconds=0,
        cache_path=tmp_path / "processed.json",
    )

    records = scanner.scan()

    assert [Path(record.source_path).name for record in records] == ["good.png"]
    assert classifier.calls == [good]
    assert scanner.process_file(generated) is None


def test_explicit_crop_cache_input_cannot_feed_crops_back_into_pipeline(tmp_path: Path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    generated = crops / "hand.png"
    generated.write_bytes(b"generated")
    classifier = _Classifier()
    scanner = ImageScanner(crops, classifier, config={"crop_recheck": {"crop_cache_dir": crops}},
                           stable_seconds=0, report_builder=False)
    assert scanner.scan() == []
    assert not classifier.calls
