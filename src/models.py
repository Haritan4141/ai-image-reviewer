"""Data models used by the image quality classifier.

The VLM is deliberately treated as an untrusted source of data.  The helper
constructors in this module normalise the model response before it is used by
the sorter or the report builder.  Keeping this conversion in one place also
makes it possible to change the wire format without changing the rest of the
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import re
from typing import Any, Mapping, Sequence


class ResultLabel(str, Enum):
    """The three decisions understood by the sorting pipeline."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"

    @classmethod
    def coerce(cls, value: object, default: "ResultLabel | None" = None) -> "ResultLabel":
        """Return a safe label for arbitrary model output.

        VLMs occasionally return lower-case labels or add punctuation.  Any
        value that is not one of the supported labels is treated as REVIEW,
        which is the safe choice when the model is uncertain.
        """

        fallback = default if default is not None else cls.REVIEW
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            candidate = value.strip().upper().strip("`* _-.")
            try:
                return cls(candidate)
            except ValueError:
                pass
        return fallback


# A couple of intuitive aliases make integrations less dependent on the exact
# enum name chosen by this module.
ClassificationLabel = ResultLabel
Decision = ResultLabel


def _bounded_number(value: object, lower: float, upper: float, default: float) -> float:
    """Convert a number-like value and clamp it to a safe range."""

    if isinstance(value, bool):
        return default
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(lower, min(upper, number))


def _score(value: object, default: int = 5) -> int:
    """Normalise an individual 1--10 score to an integer."""

    if isinstance(value, bool):
        return default
    try:
        # Models sometimes emit 8.0 or a numeric string.  Rounding is less
        # surprising than truncating, while the final value remains integral.
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(1, min(10, int(round(number))))


def _valid_numeric(value: object, lower: float, upper: float) -> bool:
    """Return whether *value* is a finite, non-boolean number in range."""

    if isinstance(value, bool):
        return False
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and lower <= number <= upper


@dataclass(slots=True)
class ScoreSet:
    """Quality scores returned by the VLM (all values are 1 through 10)."""

    anatomy: int = 5
    hands: int = 5
    face: int = 5
    artifacts: int = 5
    composition: int = 5

    def __post_init__(self) -> None:
        self.anatomy = _score(self.anatomy)
        self.hands = _score(self.hands)
        self.face = _score(self.face)
        self.artifacts = _score(self.artifacts)
        self.composition = _score(self.composition)

    @classmethod
    def from_mapping(cls, value: object) -> "ScoreSet":
        if not isinstance(value, Mapping):
            return cls()
        return cls(
            anatomy=_score(value.get("anatomy")),
            hands=_score(value.get("hands")),
            face=_score(value.get("face")),
            artifacts=_score(value.get("artifacts")),
            composition=_score(value.get("composition")),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "anatomy": self.anatomy,
            "hands": self.hands,
            "face": self.face,
            "artifacts": self.artifacts,
            "composition": self.composition,
        }

    as_dict = to_dict

    # Mapping-style access is convenient for local rules and report code.
    def __getitem__(self, name: str) -> int:
        return self.to_dict()[name]


class RegionKind(str, Enum):
    FACE = "face"
    HAND = "hand"
    FOOT = "foot"
    UPPER_BODY = "upper_body"
    LOWER_BODY = "lower_body"

    @classmethod
    def coerce(cls, value: object) -> "RegionKind":
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


@dataclass(frozen=True, slots=True)
class CropBox:
    """A nonempty normalized box in EXIF-upright image coordinates.

    Out-of-image coordinates are clamped. Invalid/inverted boxes are rejected,
    not repaired into guessed anatomy locations.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError("crop coordinates must be finite numbers")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("crop coordinates must be finite numbers") from exc
            if not math.isfinite(number):
                raise ValueError("crop coordinates must be finite numbers")
            object.__setattr__(self, name, max(0.0, min(1.0, number)))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("crop box must have positive width and height")

    @classmethod
    def from_mapping(cls, value: object) -> "CropBox":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(*(value.get(name) for name in ("x1", "y1", "x2", "y2")))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
            return cls(*value)
        raise ValueError("crop box must contain four normalized coordinates")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    to_dict = to_list

    def padded(self, ratio: float) -> "CropBox":
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError("padding must be finite and nonnegative")
        return CropBox(self.x1 - self.width * ratio, self.y1 - self.height * ratio,
                       self.x2 + self.width * ratio, self.y2 + self.height * ratio)

    def iou(self, other: "CropBox") -> float:
        intersection = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1)) * max(
            0.0, min(self.y2, other.y2) - max(self.y1, other.y1)
        )
        return intersection / (self.area + other.area - intersection)


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@dataclass(slots=True)
class RegionCheckResult:
    """A localized inspection and its evidence, never a whole-image verdict."""

    kind: RegionKind
    index: int = 0
    box: CropBox | None = None
    result: ResultLabel = ResultLabel.REVIEW
    confidence: float = 0.0
    score: int = 5
    scores: ScoreSet | None = None
    problems: list[str] = field(default_factory=list)
    summary: str = ""
    decision_source: str = "model"
    rule_reasons: list[str] = field(default_factory=list)
    detector_name: str = "unknown"
    detector_confidence: float = 0.0
    crop_path: str | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)
    model_result: ResultLabel | None = None

    def __post_init__(self) -> None:
        self.kind = RegionKind.coerce(self.kind)
        if isinstance(self.index, bool) or not _valid_numeric(self.index, 0, 100000):
            raise ValueError("region index must be a nonnegative integer")
        if float(self.index) != int(float(self.index)):
            raise ValueError("region index must be a nonnegative integer")
        self.index = int(float(self.index))
        if self.box is not None:
            self.box = CropBox.from_mapping(self.box)
        self.result = ResultLabel.coerce(self.result)
        self.confidence = _bounded_number(self.confidence, 0, 1, 0)
        self.detector_confidence = _bounded_number(self.detector_confidence, 0, 1, 0)
        self.score = _score(self.score)
        if self.scores is not None and not isinstance(self.scores, ScoreSet):
            self.scores = ScoreSet.from_mapping(self.scores)
        self.problems = _string_list(self.problems)
        self.rule_reasons = _string_list(self.rule_reasons)
        self.summary = str(self.summary or "").strip()
        self.decision_source = str(self.decision_source or "model")
        self.detector_name = str(self.detector_name or "unknown")
        self.crop_path = str(self.crop_path) if self.crop_path else None
        if self.model_result is not None:
            self.model_result = ResultLabel.coerce(self.model_result)
        if self.result is ResultLabel.PASS and self.model_result in {ResultLabel.REVIEW, ResultLabel.FAIL}:
            self.result = ResultLabel.REVIEW
            self.decision_source = "validation"
            self.rule_reasons.append("crop model verdict contradicts PASS; manual review required")

    @classmethod
    def from_mapping(cls, value: object) -> "RegionCheckResult":
        if not isinstance(value, Mapping):
            raise ValueError("region result must be a JSON object")
        invalid = []
        if value.get("box") is None:
            invalid.append("missing crop box")
        for name, low, high in (("confidence", 0, 1), ("detector_confidence", 0, 1), ("score", 1, 10)):
            if not _valid_numeric(value.get(name), low, high):
                invalid.append(f"invalid or missing region {name}")
        if not isinstance(value.get("problems"), list) or not isinstance(value.get("summary"), str):
            invalid.append("invalid region explanation")
        label = ResultLabel.coerce(value.get("result"))
        return cls(
            kind=RegionKind.coerce(value.get("kind")), index=value.get("index", 0),
            box=CropBox.from_mapping(value["box"]) if value.get("box") is not None else None,
            result=ResultLabel.REVIEW if invalid else label,
            confidence=value.get("confidence", 0), score=value.get("score", 5),
            scores=value.get("scores"), problems=invalid + _string_list(value.get("problems")),
            summary=value.get("summary", ""),
            decision_source="validation" if invalid else value.get("decision_source", "model"),
            rule_reasons=invalid + _string_list(value.get("rule_reasons")),
            detector_name=value.get("detector_name", "unknown"),
            detector_confidence=value.get("detector_confidence", 0),
            crop_path=value.get("crop_path"), model_result=value.get("model_result"),
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value, "index": self.index,
            "box": self.box.to_list() if self.box else None,
            "result": self.result.value, "model_result": self.model_result.value if self.model_result else None,
            "confidence": round(self.confidence, 6), "score": self.score,
            "scores": self.scores.to_dict() if self.scores else None,
            "problems": list(self.problems), "summary": self.summary,
            "decision_source": self.decision_source, "rule_reasons": list(self.rule_reasons),
            "detector_name": self.detector_name, "detector_confidence": round(self.detector_confidence, 6),
            "crop_path": self.crop_path,
        }

    as_dict = to_dict

    def copy_with(self, **changes: Any) -> "RegionCheckResult":
        values = self.to_dict()
        values["raw"] = self.raw
        values.update(changes)
        return type(self)(**values)


@dataclass(slots=True)
class ClassificationResult:
    """Normalised result for one image.

    ``result`` may first contain the VLM's decision and then be replaced by
    :func:`src.classifier.apply_local_rules`.  ``raw`` is retained only for
    diagnostics and is not emitted by :meth:`to_dict` unless a caller asks for
    it explicitly.
    """

    result: ResultLabel = ResultLabel.REVIEW
    confidence: float = 0.0
    scores: ScoreSet = field(default_factory=ScoreSet)
    problems: list[str] = field(default_factory=list)
    summary: str = ""
    raw: dict[str, Any] | None = field(default=None, repr=False)
    model_result: ResultLabel | None = None
    decision_source: str = "model"
    low_scores: dict[str, dict[str, int]] = field(default_factory=dict)
    keyword_hits: dict[str, list[str]] = field(default_factory=dict)
    rule_reasons: list[str] = field(default_factory=list)
    review_mode: str = "standard"
    local_rules_applied: bool = False
    crop_checks: list[RegionCheckResult] = field(default_factory=list)
    pipeline_stage: str | None = None
    pipeline_version: str | None = None
    crop_mode: str | None = None
    full_result_before_merge: str | None = None

    def __post_init__(self) -> None:
        self.result = ResultLabel.coerce(self.result)
        self.confidence = _bounded_number(self.confidence, 0.0, 1.0, 0.0)
        if not isinstance(self.scores, ScoreSet):
            self.scores = ScoreSet.from_mapping(self.scores)
        # Keep short, human-readable strings and never let a malformed value
        # cause report generation to fail.
        if not isinstance(self.problems, list):
            self.problems = list(self.problems) if isinstance(self.problems, Sequence) and not isinstance(self.problems, (str, bytes)) else []
        self.problems = [str(item).strip() for item in self.problems if str(item).strip()]
        self.summary = str(self.summary or "").strip()
        if self.model_result is not None:
            self.model_result = ResultLabel.coerce(self.model_result)
        self.decision_source = str(self.decision_source or "model").strip()
        self.low_scores = {
            str(group): {str(name): _score(score) for name, score in values.items()}
            for group, values in (self.low_scores if isinstance(self.low_scores, Mapping) else {}).items()
            if isinstance(values, Mapping)
        }
        self.keyword_hits = {
            str(group): [str(item) for item in values]
            for group, values in (self.keyword_hits if isinstance(self.keyword_hits, Mapping) else {}).items()
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
        }
        self.rule_reasons = _string_list(self.rule_reasons)
        self.review_mode = str(self.review_mode or "standard").strip().lower()
        checks = []
        crop_invalid = not isinstance(self.crop_checks, list)
        for check in self.crop_checks if isinstance(self.crop_checks, list) else []:
            try:
                checks.append(check if isinstance(check, RegionCheckResult) else RegionCheckResult.from_mapping(check))
            except (TypeError, ValueError):
                crop_invalid = True
        self.crop_checks = checks
        if self.result is ResultLabel.PASS and any(check.result is not ResultLabel.PASS for check in checks):
            self.result = ResultLabel.REVIEW
            self.rule_reasons.append("crop concerns contradict serialized PASS; manual review required")
            self.decision_source = "validation"
        if crop_invalid:
            if self.result is ResultLabel.PASS:
                self.result = ResultLabel.REVIEW
            self.rule_reasons.append("invalid crop_checks payload; manual review required")
            self.decision_source = "validation"
        for name in ("pipeline_stage", "pipeline_version", "crop_mode", "full_result_before_merge"):
            value = getattr(self, name)
            setattr(self, name, str(value) if value is not None else None)

    @classmethod
    def from_mapping(cls, value: object) -> "ClassificationResult":
        """Build a result from a decoded JSON object.

        Missing or invalid fields intentionally receive conservative defaults;
        an unknown decision is REVIEW rather than PASS.
        """

        if not isinstance(value, Mapping):
            raise ValueError("classification payload must be a JSON object")
        problems_value = value.get("problems", [])
        if isinstance(problems_value, str):
            problems: list[str] = [problems_value]
        elif isinstance(problems_value, Sequence) and not isinstance(problems_value, (bytes, str)):
            problems = [str(item) for item in problems_value]
        else:
            problems = []
        raw = dict(value)
        # A syntactically valid but structurally incomplete JSON response must
        # not accidentally become PASS.  The diagnostics are retained in the
        # normal problem list so logs and the HTML report make the reason clear.
        validation_problems: list[str] = []
        raw_result = value.get("result")
        result = ResultLabel.coerce(raw_result)
        normalized_result = raw_result.strip().upper().strip("`* _-.") if isinstance(raw_result, str) else ""
        valid_model_result = normalized_result in {item.value for item in ResultLabel}
        if not valid_model_result:
            validation_problems.append("invalid or missing result field")
            result = ResultLabel.REVIEW
        if "confidence" not in value:
            validation_problems.append("missing confidence field")
            result = ResultLabel.REVIEW
        elif not _valid_numeric(value.get("confidence"), 0.0, 1.0):
            validation_problems.append("invalid confidence field")
            result = ResultLabel.REVIEW
        score_payload = value.get("scores")
        if not isinstance(score_payload, Mapping):
            validation_problems.append("missing or invalid scores object")
            result = ResultLabel.REVIEW
        elif any(name not in score_payload for name in ("anatomy", "hands", "face", "artifacts", "composition")):
            validation_problems.append("missing score fields")
            result = ResultLabel.REVIEW
        else:
            invalid_scores = [
                name
                for name in ("anatomy", "hands", "face", "artifacts", "composition")
                if not _valid_numeric(score_payload.get(name), 1.0, 10.0)
            ]
            if invalid_scores:
                validation_problems.append("invalid score fields: " + ", ".join(invalid_scores))
                result = ResultLabel.REVIEW
        if "problems" not in value or not isinstance(value.get("problems"), list):
            validation_problems.append("missing or invalid problems array")
            result = ResultLabel.REVIEW
        if "summary" not in value or not isinstance(value.get("summary"), str):
            validation_problems.append("missing or invalid summary field")
            result = ResultLabel.REVIEW
        problems = validation_problems + problems
        return cls(
            result=result,
            confidence=_bounded_number(value.get("confidence"), 0.0, 1.0, 0.0),
            scores=ScoreSet.from_mapping(value.get("scores")),
            problems=problems,
            summary=str(value.get("summary", "") or ""),
            raw=raw,
            model_result=ResultLabel.coerce(value.get("model_result") or raw_result) if valid_model_result else None,
            decision_source="validation" if validation_problems else value.get("decision_source", "model"),
            rule_reasons=list(validation_problems) + _string_list(value.get("rule_reasons")),
            low_scores=value.get("low_scores", {}), keyword_hits=value.get("keyword_hits", {}),
            review_mode=value.get("review_mode", "standard"),
            crop_checks=value.get("crop_checks", []),
            pipeline_stage=value.get("pipeline_stage"), pipeline_version=value.get("pipeline_version"),
            crop_mode=value.get("crop_mode"), full_result_before_merge=value.get("full_result_before_merge"),
        )

    # Common aliases used by callers that prefer a JSON-oriented name.
    from_dict = from_mapping

    @classmethod
    def from_model_mapping(cls, value: object) -> "ClassificationResult":
        """Read only the public five-field wire contract from an untrusted VLM.

        Pipeline metadata belongs to this application, never to model output.
        Stored results can still be restored using from_mapping().
        """
        if not isinstance(value, Mapping):
            raise ValueError("classification payload must be a JSON object")
        return cls.from_mapping({key: value[key] for key in
                                 ("result", "confidence", "scores", "problems", "summary") if key in value})

    @classmethod
    def from_json_text(cls, text: str) -> "ClassificationResult":
        payload = decode_json_object(text)
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.value,
            "confidence": round(self.confidence, 6),
            "scores": self.scores.to_dict(),
            "problems": list(self.problems),
            "summary": self.summary,
            "model_result": self.model_result.value if self.model_result else None,
            "decision_source": self.decision_source,
            "low_scores": {group: dict(values) for group, values in self.low_scores.items()},
            "keyword_hits": {group: list(values) for group, values in self.keyword_hits.items()},
            "rule_reasons": list(self.rule_reasons), "review_mode": self.review_mode,
            "crop_checks": [check.to_dict() for check in self.crop_checks],
            "pipeline_stage": self.pipeline_stage, "pipeline_version": self.pipeline_version,
            "crop_mode": self.crop_mode, "full_result_before_merge": self.full_result_before_merge,
        }

    as_dict = to_dict

    def copy_with(self, **changes: Any) -> "ClassificationResult":
        """Return a normalised copy without exposing dataclasses.replace."""

        values: dict[str, Any] = {
            "result": self.result,
            "confidence": self.confidence,
            "scores": ScoreSet.from_mapping(self.scores.to_dict()),
            "problems": list(self.problems),
            "summary": self.summary,
            "raw": self.raw,
            "model_result": self.model_result,
            "decision_source": self.decision_source,
            "low_scores": {group: dict(values) for group, values in self.low_scores.items()},
            "keyword_hits": {group: list(values) for group, values in self.keyword_hits.items()},
            "rule_reasons": list(self.rule_reasons),
            "review_mode": self.review_mode,
            "local_rules_applied": self.local_rules_applied,
            "crop_checks": [check.copy_with() for check in self.crop_checks],
            "pipeline_stage": self.pipeline_stage, "pipeline_version": self.pipeline_version,
            "crop_mode": self.crop_mode, "full_result_before_merge": self.full_result_before_merge,
        }
        values.update(changes)
        return type(self)(**values)


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


def decode_json_object(text: str) -> dict[str, Any]:
    """Decode the first JSON object from a VLM response.

    The prompt asks for JSON only, but accepting a fenced object or a short
    explanatory prefix makes the retry path more useful.  Invalid content is
    rejected so the caller can issue a correction request.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty VLM response")
    candidate = text.strip()
    fenced = _CODE_FENCE_RE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Locate an object even if the model added one sentence before/after it.
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("VLM response did not contain a valid JSON object")


__all__ = [
    "ClassificationLabel",
    "ClassificationResult",
    "CropBox",
    "Decision",
    "ResultLabel",
    "RegionKind",
    "RegionCheckResult",
    "ScoreSet",
    "decode_json_object",
]
