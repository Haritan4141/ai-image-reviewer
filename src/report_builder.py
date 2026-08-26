"""JSONL/CSV persistence and a self-contained static review page."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from .utils import atomic_write_text, ensure_directory, json_default, normalize_path, path_to_file_url, read_jsonl


def _get(value: object | None, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _dict_record(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        raw = value.to_dict()
        result = dict(raw) if isinstance(raw, Mapping) else {}
    elif hasattr(value, "model_dump") and callable(value.model_dump):
        raw = value.model_dump()
        result = dict(raw) if isinstance(raw, Mapping) else {}
    elif is_dataclass(value):
        result = asdict(value)
    else:
        result = {}
        for name in (
            "source_path",
            "destination_path",
            "file_hash",
            "result",
            "confidence",
            "scores",
            "problems",
            "summary",
            "relative_path",
            "size_bytes",
            "modified_at",
            "processed_at",
            "status",
            "error",
        ):
            if hasattr(value, name):
                result[name] = getattr(value, name)
    return _json_normalize(result)


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        return _json_normalize(getattr(value, "value"))
    if isinstance(value, Mapping):
        return {str(k): _json_normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_normalize(v) for v in value]
    return value


def _label(value: object) -> str:
    value = getattr(value, "value", value)
    text = str(value or "REVIEW").strip().upper().strip("`* _-.")
    return text if text in {"PASS", "REVIEW", "FAIL"} else "REVIEW"


def _time_key(record: Mapping[str, Any]) -> str:
    return str(record.get("processed_at") or record.get("timestamp") or record.get("created_at") or "")


class ReportBuilder:
    """Persist scan records and render a local ``review.html`` page."""

    CSV_FIELDS = (
        "processed_at",
        "result",
        "confidence",
        "source_path",
        "destination_path",
        "relative_path",
        "file_hash",
        "scores",
        "problems",
        "summary",
        "status",
        "error",
    )

    def __init__(
        self,
        output_dir: str | os.PathLike[str] | Path | object = "output",
        logs_dir: str | os.PathLike[str] | Path | None = "logs",
        *,
        jsonl_path: str | os.PathLike[str] | Path | None = None,
        csv_path: str | os.PathLike[str] | Path | None = None,
        report_path: str | os.PathLike[str] | Path | None = None,
        config: object | None = None,
        thumbnail_width: int | None = None,
        thumbnail_height: int | None = None,
    ) -> None:
        if config is None and not isinstance(output_dir, (str, os.PathLike, Path)):
            config = output_dir
        if config is not None:
            output_section = _get(config, "output", None)
            logs_section = _get(config, "logs", None)
            report_section = _get(config, "report", None)
            output_dir = _get(output_section, "directory", output_dir)
            logs_dir = _get(logs_section, "directory", logs_dir)
            jsonl_path = _get(config, "results_jsonl_path", jsonl_path)
            csv_path = _get(config, "summary_csv_path", csv_path)
            report_path = _get(config, "report_path", report_path)
            thumbnail_width = thumbnail_width if thumbnail_width is not None else _get(report_section, "thumbnail_width", 320)
            thumbnail_height = thumbnail_height if thumbnail_height is not None else _get(report_section, "thumbnail_height", 320)
        self.output_dir = normalize_path(output_dir)  # type: ignore[arg-type]
        self.logs_dir = normalize_path(logs_dir or self.output_dir.parent / "logs")
        self.jsonl_path = normalize_path(jsonl_path) if jsonl_path is not None else self.logs_dir / "results.jsonl"
        self.csv_path = normalize_path(csv_path) if csv_path is not None else self.logs_dir / "latest_summary.csv"
        # Keeping the report beside output/ and logs/ makes the generated
        # relative image links work when the HTML is double-clicked.
        self.report_path = normalize_path(report_path) if report_path is not None else self.output_dir.parent / "review.html"
        self.thumbnail_width = max(64, int(thumbnail_width or 320))
        self.thumbnail_height = max(64, int(thumbnail_height or 320))
        self._lock = threading.RLock()

    def append_record(self, record: object) -> dict[str, Any]:
        """Append one validated JSON object to ``results.jsonl``."""

        data = _dict_record(record)
        data.setdefault("processed_at", datetime.now(timezone.utc).isoformat())
        data["result"] = _label(data.get("result"))
        data["confidence"] = self._confidence(data.get("confidence"))
        data.setdefault("problems", [])
        data.setdefault("scores", {})
        line = json.dumps(data, ensure_ascii=False, default=json_default, separators=(",", ":"))
        with self._lock:
            ensure_directory(self.jsonl_path.parent)
            with self.jsonl_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")
        return data

    append = append_record
    write_record = append_record

    @staticmethod
    def _confidence(value: object) -> float:
        try:
            number = float(value or 0.0)
        except (TypeError, ValueError):
            number = 0.0
        return round(max(0.0, min(1.0, number)), 6)

    def load_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_dict_record(record) for record in read_jsonl(self.jsonl_path)]

    def write_jsonl(self, records: Iterable[object]) -> Path:
        """Replace the JSONL file with *records* (useful for migrations/tests)."""

        lines: list[str] = []
        for record in records:
            data = _dict_record(record)
            data["result"] = _label(data.get("result"))
            data["confidence"] = self._confidence(data.get("confidence"))
            lines.append(json.dumps(data, ensure_ascii=False, default=json_default, separators=(",", ":")))
        with self._lock:
            atomic_write_text(self.jsonl_path, "" if not lines else "\n".join(lines) + "\n")
        return self.jsonl_path

    def write_summary_csv(self, records: Sequence[object] | None = None) -> Path:
        """Write a Windows/Excel-friendly summary CSV in newest-first order."""

        values = [_dict_record(item) for item in (records if records is not None else self.load_records())]
        values.sort(key=_time_key, reverse=True)
        ensure_directory(self.csv_path.parent)
        temporary = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for item in values:
                    row = dict(item)
                    row["result"] = _label(row.get("result"))
                    row["confidence"] = self._confidence(row.get("confidence"))
                    row["scores"] = json.dumps(_json_normalize(row.get("scores", {})), ensure_ascii=False, separators=(",", ":"))
                    problems = row.get("problems", [])
                    row["problems"] = "; ".join(str(problem) for problem in problems) if isinstance(problems, (list, tuple)) else str(problems or "")
                    writer.writerow({field: row.get(field, "") for field in self.CSV_FIELDS})
            os.replace(os.fspath(temporary), os.fspath(self.csv_path))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return self.csv_path

    def _asset_href(self, value: object) -> str:
        if value is None or not str(value).strip():
            return ""
        raw = str(value)
        target = Path(raw)
        if not target.is_absolute():
            # Paths in records are normally absolute.  Treat a relative path
            # as relative to the HTML file so reports remain portable.
            return quote(raw.replace("\\", "/"), safe="/:@?&=+#%")
        try:
            relative = os.path.relpath(os.fspath(target), os.fspath(self.report_path.parent))
            if not relative.startswith(".." + os.sep + ".." + os.sep) and not (len(relative) >= 2 and relative[1] == ":"):
                return quote(relative.replace("\\", "/"), safe="/:@?&=+#%")
        except (ValueError, OSError):
            pass
        return html.escape(path_to_file_url(target), quote=True)

    def _card_html(self, record: Mapping[str, Any]) -> str:
        label = _label(record.get("result"))
        confidence = self._confidence(record.get("confidence"))
        problems = record.get("problems", [])
        if not isinstance(problems, (list, tuple)):
            problems = [problems] if problems else []
        problems_text = "<br>".join(html.escape(str(item)) for item in problems) or "None"
        source = str(record.get("source_path") or record.get("path") or "")
        destination = record.get("destination_path") or record.get("destination") or source
        href = self._asset_href(destination)
        image = (
            f'<img loading="lazy" src="{html.escape(href, quote=True)}" '
            f'alt="{html.escape(Path(source).name if source else label, quote=True)}" '
            f'width="{self.thumbnail_width}" height="{self.thumbnail_height}" '
            'onerror="this.classList.add(\'missing\'); this.alt=\'Thumbnail unavailable\';">'
            if href
            else '<div class="missing">Thumbnail unavailable</div>'
        )
        summary = html.escape(str(record.get("summary") or ""))
        source_text = html.escape(source)
        timestamp = html.escape(_time_key(record))
        return (
            f'<article class="card" data-result="{label}" data-time="{timestamp}">'
            f'<div class="thumb">{image}</div>'
            f'<div class="meta"><span class="badge {label.lower()}">{label}</span>'
            f'<span class="confidence">confidence {confidence:.2f}</span></div>'
            f'<div class="problems"><strong>Problems</strong><br>{problems_text}</div>'
            f'<p class="summary">{summary}</p>'
            f'<details><summary>File</summary><code>{source_text}</code></details>'
            f'<time datetime="{timestamp}">{timestamp}</time>'
            '</article>'
        )

    def build_html(self, records: Sequence[object] | None = None) -> Path:
        values = [_dict_record(item) for item in (records if records is not None else self.load_records())]
        values.sort(key=_time_key, reverse=True)
        cards = "\n".join(self._card_html(item) for item in values)
        generated = datetime.now(timezone.utc).isoformat()
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Image Review</title>
<style>
:root {{ color-scheme: light dark; --bg:#10141b; --panel:#1a2230; --text:#e8edf5; --muted:#9eabba; }}
* {{ box-sizing:border-box; }} body {{ margin:0; padding:24px; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }}
h1 {{ margin:0 0 6px; }} .updated {{ color:var(--muted); margin-bottom:18px; }}
.toolbar {{ position:sticky; top:0; z-index:2; padding:12px 0; background:color-mix(in srgb,var(--bg) 90%,transparent); backdrop-filter:blur(5px); }}
button {{ border:1px solid #4b5a70; border-radius:6px; background:var(--panel); color:var(--text); padding:7px 12px; margin-right:6px; cursor:pointer; }}
button.active {{ outline:2px solid #72a7ff; }} #grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:18px; }}
.card {{ background:var(--panel); border:1px solid #344153; border-radius:10px; padding:12px; overflow:hidden; }}
.thumb {{ display:flex; align-items:center; justify-content:center; min-height:{self.thumbnail_height}px; background:#07090d; border-radius:7px; overflow:hidden; }}
.thumb img {{ display:block; max-width:100%; height:auto; object-fit:contain; }} .thumb img.missing,.missing {{ color:#f5b8b8; padding:20px; }}
.meta {{ display:flex; justify-content:space-between; align-items:center; margin:10px 0; }} .badge {{ border-radius:999px; padding:3px 9px; font-weight:700; }}
.pass {{ background:#1c7845; color:#d5ffe7; }} .review {{ background:#89691b; color:#fff5cf; }} .fail {{ background:#932d3b; color:#ffe2e5; }}
.confidence {{ color:var(--muted); }} .problems {{ min-height:40px; }} .summary {{ color:var(--muted); }} details {{ margin-top:8px; }} code {{ display:block; white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }} time {{ display:block; color:var(--muted); font-size:12px; margin-top:9px; }}
</style></head><body>
<h1>AI Image Review</h1><div class="updated">{len(values)} images · generated {html.escape(generated)}</div>
<div class="toolbar" role="toolbar" aria-label="Filter results"><button class="active" data-filter="ALL">All</button><button data-filter="PASS">PASS</button><button data-filter="REVIEW">REVIEW</button><button data-filter="FAIL">FAIL</button></div>
<main id="grid">{cards}</main>
<script>
document.querySelectorAll('[data-filter]').forEach(function(button) {{ button.addEventListener('click', function() {{
  document.querySelectorAll('[data-filter]').forEach(function(item) {{ item.classList.remove('active'); }}); button.classList.add('active');
  var wanted=button.dataset.filter; document.querySelectorAll('.card').forEach(function(card) {{ card.hidden = wanted !== 'ALL' && card.dataset.result !== wanted; }});
}}); }});
</script></body></html>
"""
        with self._lock:
            atomic_write_text(self.report_path, document)
        return self.report_path

    def build(self, records: Sequence[object] | None = None) -> dict[str, Path]:
        """Write CSV and HTML from all current JSONL records."""

        values = [_dict_record(item) for item in (records if records is not None else self.load_records())]
        csv_path = self.write_summary_csv(values)
        html_path = self.build_html(values)
        return {"jsonl": self.jsonl_path, "csv": csv_path, "html": html_path}

    build_report = build
    generate = build
    build_review_html = build_html
    write_results_jsonl = write_jsonl
    write_latest_summary_csv = write_summary_csv


def build_report(
    records: Sequence[Mapping[str, Any]] | None = None,
    *,
    output_dir: str | os.PathLike[str] | Path = "output",
    **kwargs: Any,
) -> dict[str, Path]:
    """Convenience wrapper used by the ``build-report`` CLI command."""

    return ReportBuilder(output_dir=output_dir, **kwargs).build(records)


__all__ = ["ReportBuilder", "build_report"]
