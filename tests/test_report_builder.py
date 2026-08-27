from __future__ import annotations

import csv
import json
from pathlib import Path

from src.report_builder import ReportBuilder


def _record(source: Path, destination: Path, label: str, timestamp: str) -> dict[str, object]:
    return {
        "source_path": str(source),
        "destination_path": str(destination),
        "file_hash": f"hash-{label}",
        "result": label,
        "model_result": label,
        "final_result": label,
        "decision_source": "model",
        "review_mode": "standard",
        "low_scores": {},
        "keyword_hits": {},
        "rule_reasons": [],
        "confidence": 0.93 if label == "PASS" else 0.45,
        "scores": {"hands": 8},
        "problems": [] if label == "PASS" else ["deformed hand"],
        "summary": "safe <summary>" if label == "PASS" else "needs review",
        "processed_at": timestamp,
        "status": "processed",
    }


def test_report_builder_persists_jsonl_csv_and_filterable_html(tmp_path: Path) -> None:
    output = tmp_path / "output"
    pass_image = output / "pass" / "pass.png"
    review_image = output / "review" / "review.png"
    pass_image.parent.mkdir(parents=True)
    review_image.parent.mkdir(parents=True)
    pass_image.write_bytes(b"pass")
    review_image.write_bytes(b"review")
    source_pass = tmp_path / "incoming" / "pass.png"
    source_review = tmp_path / "incoming" / "review.png"
    builder = ReportBuilder(
        output,
        tmp_path / "logs",
        report_path=tmp_path / "review.html",
    )

    builder.append_record(_record(source_pass, pass_image, "PASS", "2026-01-02T00:00:00+00:00"))
    builder.append_record(_record(source_review, review_image, "REVIEW", "2026-01-01T00:00:00+00:00"))
    paths = builder.build()

    assert paths["jsonl"].is_file()
    assert paths["csv"].is_file()
    assert paths["html"].is_file()
    records = [json.loads(line) for line in paths["jsonl"].read_text(encoding="utf-8").splitlines()]
    assert [item["result"] for item in records] == ["PASS", "REVIEW"]
    assert records[0]["confidence"] == 0.93

    with paths["csv"].open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["result"] for row in rows] == ["PASS", "REVIEW"]
    assert rows[1]["problems"] == "deformed hand"
    assert rows[0]["model_result"] == "PASS"
    assert rows[0]["final_result"] == "PASS"

    html = paths["html"].read_text(encoding="utf-8")
    assert 'data-result="PASS"' in html
    assert 'data-result="REVIEW"' in html
    assert 'data-filter="FAIL"' in html
    assert "confidence 0.93" in html
    assert "<strong>Model</strong> PASS" in html
    assert "&lt;summary&gt;" in html
    assert "output/pass/pass.png" in html


def test_report_builder_clamps_bad_confidence_and_loads_records(tmp_path: Path) -> None:
    builder = ReportBuilder(tmp_path / "output", tmp_path / "logs")

    value = builder.append_record(
        {
            "result": "unknown",
            "confidence": 5,
            "source_path": str(tmp_path / "image.png"),
            "problems": "unknown problem",
        }
    )

    assert value["result"] == "REVIEW"
    assert value["confidence"] == 1.0
    assert builder.load_records()[0]["result"] == "REVIEW"


def test_report_builder_persists_and_renders_crop_checks_safely(tmp_path: Path) -> None:
    output = tmp_path / "output"
    original = output / "review" / "original.png"
    crop = tmp_path / "cache" / "crops" / "hand-0.png"
    original.parent.mkdir(parents=True)
    crop.parent.mkdir(parents=True)
    original.write_bytes(b"original")
    crop.write_bytes(b"crop")
    builder = ReportBuilder(output, tmp_path / "logs", report_path=tmp_path / "review.html")
    record = {
        "source_path": str(tmp_path / "incoming" / "original.png"),
        "destination_path": str(original),
        "file_hash": "hash",
        "result": "REVIEW",
        "model_result": "PASS",
        "final_result": "REVIEW",
        "decision_source": "crop_merge",
        "review_mode": "balanced",
        "pipeline_stage": "full_plus_crop",
        "pipeline_version": "crop-v1",
        "crop_mode": "balanced",
        "full_result_before_merge": "PASS",
        "pipeline_note": "<not rendered>",
        "confidence": 0.55,
        "scores": {},
        "problems": [],
        "summary": "<unsafe summary>",
        "processed_at": "2026-08-27T00:00:00+00:00",
        "crop_checks": [
            {
                "kind": "hand",
                "index": 0,
                "box": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
                "result": "REVIEW",
                "confidence": 0.4,
                "problems": ["<finger ambiguity>"],
                "summary": "ignored in card",
                "rule_reasons": ["crop review"],
                "detector_name": "fallback <detector>",
                "detector_confidence": 0.5,
                "crop_path": str(crop),
                "raw": {"secret": "not serialized"},
            },
            {
                "kind": "face",
                "index": 1,
                "box": [0.0, 0.0, 1.0, 1.0],
                "result": "PASS",
                "confidence": 0.9,
                "detector_name": "removed",
                "crop_path": str(tmp_path / "removed.png"),
            },
        ],
    }

    appended = builder.append_record(record)
    paths = builder.build()

    assert appended["crop_checks"][0]["box"] == [0.1, 0.2, 0.8, 0.9]
    assert "raw" not in appended["crop_checks"][0]
    saved = json.loads(paths["jsonl"].read_text(encoding="utf-8").splitlines()[0])
    assert saved["pipeline_version"] == "crop-v1"
    assert saved["crop_checks"][1]["result"] == "PASS"
    with paths["csv"].open("r", encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert json.loads(row["crop_checks"])[0]["kind"] == "hand"
    assert row["pipeline_stage"] == "full_plus_crop"

    html = paths["html"].read_text(encoding="utf-8")
    assert "stage full_plus_crop" in html
    assert "version crop-v1" in html
    assert "crop mode balanced" in html
    assert "full before merge PASS" in html
    assert "[Hand 0]" in html
    assert "[Face 1]" in html
    assert "fallback &lt;detector&gt;" in html
    assert "&lt;finger ambiguity&gt;" in html
    assert "Crop thumbnail unavailable" in html
    assert "hand-0.png" in html
    assert "raw" not in html


def test_report_builder_legacy_records_remain_renderable_without_crop_fields(tmp_path: Path) -> None:
    builder = ReportBuilder(tmp_path / "output", tmp_path / "logs", report_path=tmp_path / "review.html")
    builder.append_record(
        {
            "source_path": str(tmp_path / "old.png"),
            "result": "PASS",
            "confidence": 0.9,
            "processed_at": "2026-08-27T00:00:00+00:00",
        }
    )
    paths = builder.build()
    assert paths["html"].is_file()
    assert "Crop checks" not in paths["html"].read_text(encoding="utf-8")
