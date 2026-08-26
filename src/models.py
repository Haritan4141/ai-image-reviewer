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
        if normalized_result not in {item.value for item in ResultLabel}:
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
        )

    # Common aliases used by callers that prefer a JSON-oriented name.
    from_dict = from_mapping

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
        }

    as_dict = to_dict

    def copy_with(self, **changes: Any) -> "ClassificationResult":
        """Return a normalised copy without exposing dataclasses.replace."""

        values: dict[str, Any] = {
            "result": self.result,
            "confidence": self.confidence,
            "scores": self.scores,
            "problems": list(self.problems),
            "summary": self.summary,
            "raw": self.raw,
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
    "Decision",
    "ResultLabel",
    "ScoreSet",
    "decode_json_object",
]
