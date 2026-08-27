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
        try:
            raw = value.to_dict()
            result = dict(raw) if isinstance(raw, Mapping) else {}
        except Exception:
            result = {}
    elif hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            raw = value.model_dump()
            result = dict(raw) if isinstance(raw, Mapping) else {}
        except Exception:
            result = {}
    elif is_dataclass(value):
        try:
            result = asdict(value)
        except Exception:
            result = {}
    else:
        result = {}
        for name in (
            "source_path",
            "destination_path",
            "file_hash",
            "result",
            "model_result",
            "final_result",
            "decision_source",
            "review_mode",
            "low_scores",
            "keyword_hits",
            "rule_reasons",
            "crop_checks",
            "pipeline_stage",
            "pipeline_version",
            "crop_mode",
            "full_result_before_merge",
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
    if "crop_checks" in result:
        result["crop_checks"] = _normalise_crop_checks(result.get("crop_checks"))
    return _json_normalize(result)


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        return _json_normalize(getattr(value, "value"))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_normalize(value.to_dict())
        except Exception:
            return None
    if is_dataclass(value):
        try:
            return _json_normalize(asdict(value))
        except Exception:
            return None
    if isinstance(value, Mapping):
        return {str(k): _json_normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_normalize(v) for v in value]
    return value


def _normalise_crop_box(value: object) -> Any:
    """Return the portable flat representation used by CSV/HTML reports."""

    if value is None:
        return None
    candidate: Any = value
    try:
        to_list = getattr(candidate, "to_list", None)
        if callable(to_list):
            candidate = to_list()
        elif hasattr(candidate, "to_dict") and callable(candidate.to_dict):
            candidate = candidate.to_dict()
    except Exception:
        return None
    if isinstance(candidate, Mapping):
        names = ("x1", "y1", "x2", "y2")
        if all(name in candidate for name in names):
            candidate = [candidate[name] for name in names]
        else:
            names = ("left", "top", "right", "bottom")
            if all(name in candidate for name in names):
                candidate = [candidate[name] for name in names]
    safe = _json_normalize(candidate)
    return list(safe) if isinstance(safe, (list, tuple)) else safe


def _normalise_crop_check(value: object) -> dict[str, Any] | None:
    """Convert one region result while dropping diagnostic-only ``raw`` data."""

    try:
        if isinstance(value, Mapping):
            payload = dict(value)
        elif hasattr(value, "to_dict") and callable(value.to_dict):
            raw = value.to_dict()
            payload = dict(raw) if isinstance(raw, Mapping) else {}
        elif hasattr(value, "model_dump") and callable(value.model_dump):
            raw = value.model_dump()
            payload = dict(raw) if isinstance(raw, Mapping) else {}
        elif is_dataclass(value):
            payload = asdict(value)
        else:
            payload = {
                name: getattr(value, name)
                for name in (
                    "kind",
                    "index",
                    "box",
                    "result",
                    "confidence",
                    "score",
                    "scores",
                    "problems",
                    "summary",
                    "decision_source",
                    "rule_reasons",
                    "detector_name",
                    "detector_confidence",
                    "crop_path",
                    "model_result",
                )
                if hasattr(value, name)
            }
        if not payload:
            return None
        payload.pop("raw", None)
        box = payload.get("box")
        normalized = _json_normalize(payload)
        if not isinstance(normalized, Mapping):
            return None
        result = dict(normalized)
        if "box" in result:
            result["box"] = _normalise_crop_box(box)
        return result
    except Exception:
        return None


def _normalise_crop_checks(value: object) -> list[dict[str, Any]]:
    """Normalize crop checks defensively; malformed optional data is ignored."""

    if value is None or isinstance(value, (str, bytes, bytearray)):
        return []
    if isinstance(value, Mapping):
        values: Iterable[object] = (value,)
    else:
        try:
            values = list(value)  # type: ignore[arg-type]
        except Exception:
            return []
    checks: list[dict[str, Any]] = []
    for item in values:
        check = _normalise_crop_check(item)
        if check is not None:
            checks.append(check)
    return checks


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
        "model_result",
        "final_result",
        "decision_source",
        "review_mode",
        "pipeline_stage",
        "pipeline_version",
        "crop_mode",
        "full_result_before_merge",
        "crop_checks",
        "confidence",
        "source_path",
        "destination_path",
        "relative_path",
        "file_hash",
        "scores",
        "low_scores",
        "keyword_hits",
        "rule_reasons",
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
        data.setdefault("final_result", data["result"])
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

    @staticmethod
    def _csv_cell(field: str, value: object) -> object:
        """Protect newly added text columns from spreadsheet formula injection."""

        if field not in {
            "pipeline_stage",
            "pipeline_version",
            "crop_mode",
            "full_result_before_merge",
            "crop_checks",
        }:
            return value
        if value is None:
            return ""
        text = str(value)
        if text.startswith(("=", "+", "-", "@")):
            return "'" + text
        return value

    def load_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_dict_record(record) for record in read_jsonl(self.jsonl_path)]

    def write_jsonl(self, records: Iterable[object]) -> Path:
        """Replace the JSONL file with *records* (useful for migrations/tests)."""

        lines: list[str] = []
        for record in records:
            data = _dict_record(record)
            data["result"] = _label(data.get("result"))
            data.setdefault("final_result", data["result"])
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
                    row["low_scores"] = json.dumps(_json_normalize(row.get("low_scores", {})), ensure_ascii=False, separators=(",", ":"))
                    row["keyword_hits"] = json.dumps(_json_normalize(row.get("keyword_hits", {})), ensure_ascii=False, separators=(",", ":"))
                    row["crop_checks"] = json.dumps(
                        _normalise_crop_checks(row.get("crop_checks", [])),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    problems = row.get("problems", [])
                    row["problems"] = "; ".join(str(problem) for problem in problems) if isinstance(problems, (list, tuple)) else str(problems or "")
                    reasons = row.get("rule_reasons", [])
                    row["rule_reasons"] = "; ".join(str(reason) for reason in reasons) if isinstance(reasons, (list, tuple)) else str(reasons or "")
                    writer.writerow({field: self._csv_cell(field, row.get(field, "")) for field in self.CSV_FIELDS})
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

    def _existing_asset_href(self, value: object) -> str:
        """Return an asset URL only when the referenced file still exists."""

        if value is None or not str(value).strip():
            return ""
        try:
            target = Path(str(value))
        except (TypeError, ValueError, OSError):
            return ""
        candidates = [target]
        if not target.is_absolute():
            candidates.append(self.report_path.parent / target)
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return self._asset_href(candidate)
            except OSError:
                continue
        return ""

    @staticmethod
    def _values(value: object) -> list[str]:
        if value is None or isinstance(value, (str, bytes, bytearray)):
            return [str(value)] if isinstance(value, str) and value.strip() else []
        try:
            return [str(item) for item in value if str(item).strip()]  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _kind_label(value: object) -> str:
        value = getattr(value, "value", value)
        text = str(value or "region").strip().lower().replace("_", " ")
        return text.title() if text else "Region"

    def _crop_checks_html(self, record: Mapping[str, Any]) -> str:
        checks = _normalise_crop_checks(record.get("crop_checks", []))
        if not checks:
            return ""
        items: list[str] = []
        for check in checks:
            kind = html.escape(self._kind_label(check.get("kind")), quote=True)
            index = html.escape(str(check.get("index", "")), quote=True)
            label = _label(check.get("result"))
            confidence = self._confidence(check.get("confidence"))
            reasons = self._values(check.get("rule_reasons"))
            problems = self._values(check.get("problems"))
            for problem in problems:
                if problem not in reasons:
                    reasons.append(problem)
            reasons_text = "; ".join(reasons) or "None"
            summary = html.escape(str(check.get("summary") or ""), quote=True)
            detector_name = str(check.get("detector_name") or "unknown")
            detector_confidence = check.get("detector_confidence")
            if detector_confidence is None or str(detector_confidence).strip() == "":
                detector_text = detector_name
            else:
                detector_text = f"{detector_name} ({self._confidence(detector_confidence):.2f})"
            detector_text = html.escape(detector_text, quote=True)
            crop_path = str(check.get("crop_path") or "")
            crop_href = self._existing_asset_href(crop_path)
            if crop_href:
                thumbnail = (
                    f'<a href="{html.escape(crop_href, quote=True)}" target="_blank" rel="noopener">'
                    f'<img loading="lazy" class="crop-thumb" src="{html.escape(crop_href, quote=True)}" '
                    f'alt="{kind} {index} crop" width="128" height="128"></a>'
                )
            else:
                thumbnail = '<span class="crop-missing">Crop thumbnail unavailable</span>'
            crop_path_html = (
                f'<div class="crop-path"><strong>Crop path</strong> '
                f'<code>{html.escape(crop_path, quote=True)}</code></div>'
                if crop_path
                else ""
            )
            items.append(
                f'<li class="crop-check"><div><strong>[{kind} {index}]</strong> '
                f'<span class="badge {label.lower()}">{label}</span> '
                f'<span class="confidence">confidence {confidence:.2f}</span></div>'
                f'<div class="crop-score"><strong>Score</strong> {html.escape(str(check.get("score", "unknown")))} · '
                f'<strong>Source</strong> {html.escape(str(check.get("decision_source") or "model"))}</div>'
                f'<div class="crop-detector"><strong>Detector</strong> {detector_text}</div>'
                f'<div class="crop-reasons"><strong>Reasons</strong> {html.escape(reasons_text)}</div>'
                f'<div class="crop-summary"><strong>Summary</strong> {summary or "None"}</div>'
                f'{crop_path_html}'
                f'<div class="crop-preview">{thumbnail}</div></li>'
            )
        return f'<details class="crop-details" open><summary>Crop checks ({len(items)})</summary><ul class="crop-list">{"".join(items)}</ul></details>'

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
        model_result = html.escape(str(record.get("model_result") or "unknown"))
        final_result = html.escape(str(record.get("final_result") or label))
        decision_source = html.escape(str(record.get("decision_source") or "model"))
        review_mode = html.escape(str(record.get("review_mode") or "standard"))
        pipeline_stage = html.escape(str(record.get("pipeline_stage") or ""), quote=True)
        pipeline_version = html.escape(str(record.get("pipeline_version") or ""), quote=True)
        crop_mode = html.escape(str(record.get("crop_mode") or ""), quote=True)
        full_before_merge = record.get("full_result_before_merge")
        pipeline_parts = []
        if pipeline_stage:
            pipeline_parts.append(f"stage {pipeline_stage}")
        if pipeline_version:
            pipeline_parts.append(f"version {pipeline_version}")
        if crop_mode:
            pipeline_parts.append(f"crop mode {crop_mode}")
        if full_before_merge:
            pipeline_parts.append(f"full before merge {_label(full_before_merge)}")
        pipeline_html = (
            f'<div class="pipeline"><strong>Pipeline</strong> {" · ".join(pipeline_parts)}</div>'
            if pipeline_parts
            else ""
        )
        reasons = record.get("rule_reasons", [])
        if not isinstance(reasons, (list, tuple)):
            reasons = [reasons] if reasons else []
        reasons_text = "<br>".join(html.escape(str(item)) for item in reasons) or "None"
        source_text = html.escape(source)
        timestamp = html.escape(_time_key(record))
        return (
            f'<article class="card" data-result="{label}" data-time="{timestamp}">'
            f'<div class="thumb">{image}</div>'
            f'<div class="meta"><span class="badge {label.lower()}">{label}</span>'
            f'<span class="confidence">confidence {confidence:.2f}</span></div>'
            f'<div class="audit"><strong>Model</strong> {model_result} · '
            f'<strong>Final</strong> {final_result} · <strong>Source</strong> {decision_source} · '
            f'<strong>Mode</strong> {review_mode}</div>'
            f'{pipeline_html}'
            f'<div class="problems"><strong>Problems</strong><br>{problems_text}</div>'
            f'<div class="reasons"><strong>Decision reasons</strong><br>{reasons_text}</div>'
            f'{self._crop_checks_html(record)}'
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
 .confidence {{ color:var(--muted); }} .audit,.pipeline,.reasons {{ color:var(--muted); margin:7px 0; }} .problems {{ min-height:40px; }} .summary {{ color:var(--muted); }} details {{ margin-top:8px; }} code {{ display:block; white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }} time {{ display:block; color:var(--muted); font-size:12px; margin-top:9px; }} .crop-details {{ border-top:1px solid #344153; padding-top:7px; }} .crop-list {{ list-style:none; padding:0; margin:8px 0 0; display:grid; gap:9px; }} .crop-check {{ border:1px solid #344153; border-radius:7px; padding:7px; }} .crop-detector,.crop-reasons {{ color:var(--muted); font-size:12px; margin-top:4px; overflow-wrap:anywhere; }} .crop-preview {{ margin-top:6px; min-height:24px; display:flex; align-items:center; justify-content:center; background:#07090d; border-radius:5px; }} .crop-thumb {{ display:block; max-width:128px; max-height:128px; object-fit:contain; }} .crop-missing {{ color:var(--muted); font-size:12px; padding:5px; }}
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
