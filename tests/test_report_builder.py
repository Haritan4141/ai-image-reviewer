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

    html = paths["html"].read_text(encoding="utf-8")
    assert 'data-result="PASS"' in html
    assert 'data-result="REVIEW"' in html
    assert 'data-filter="FAIL"' in html
    assert "confidence 0.93" in html
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
