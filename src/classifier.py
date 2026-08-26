"""VLM invocation and conservative local safety rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .models import ClassificationResult, ResultLabel, ScoreSet


DEFAULT_REVIEW_KEYWORDS: tuple[str, ...] = (
    "extra finger",
    "missing finger",
    "fused finger",
    "deformed hand",
    "extra limb",
    "missing limb",
    "fused body",
    "fused object",
    "malformed anatomy",
    "wrong number of fingers",
    "misaligned eyes",
    "eye position",
    "duplicate body",
    "generation artifact",
    "obvious artifact",
    "anatomical issue",
)

DEFAULT_FAIL_KEYWORDS: tuple[str, ...] = (
    "severe deformation",
    "major anatomy failure",
    "multiple extra limbs",
    "unrecognizable face",
    "heavy generation noise",
    "missing body",
    "body is fused",
)


@dataclass(frozen=True, slots=True)
class LocalRulesConfig:
    """Adjustable fail-safe rules applied after the VLM response.

    The field names mirror ``src.config.RulesConfig``.  Rules are intentionally
    data-only so they can be loaded from YAML and tuned without changing code.
    """

    threshold_pass: float = 0.85
    threshold_review: float = 0.50
    score_review_below: int = 6
    score_fail_below: int = 3
    fail_score_count: int = 2
    review_problem_keywords: tuple[str, ...] = DEFAULT_REVIEW_KEYWORDS
    fail_problem_keywords: tuple[str, ...] = DEFAULT_FAIL_KEYWORDS

    @classmethod
    def from_settings(cls, settings: object | None = None) -> "LocalRulesConfig":
        """Adapt ``AppConfig.rules``, a direct rules object, or a mapping."""

        source = settings
        nested = _get(source, "rules", None) if source is not None else None
        if nested is not None:
            source = nested
        defaults = cls()

        def value(name: str, default: Any) -> Any:
            found = _get(source, name, None)
            return default if found is None else found

        def keywords(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = value(name, default)
            if isinstance(raw, str):
                raw = (raw,)
            if not isinstance(raw, Sequence) or isinstance(raw, (bytes, str)):
                return default
            return tuple(str(item).strip() for item in raw if str(item).strip())

        return cls(
            threshold_pass=float(value("threshold_pass", defaults.threshold_pass)),
            threshold_review=float(value("threshold_review", defaults.threshold_review)),
            score_review_below=int(value("score_review_below", defaults.score_review_below)),
            score_fail_below=int(value("score_fail_below", defaults.score_fail_below)),
            fail_score_count=max(1, int(value("fail_score_count", defaults.fail_score_count))),
            review_problem_keywords=keywords("review_problem_keywords", defaults.review_problem_keywords),
            fail_problem_keywords=keywords("fail_problem_keywords", defaults.fail_problem_keywords),
        )


# Singular spelling is convenient for call sites; both are public.
LocalRuleConfig = LocalRulesConfig


def _get(value: object | None, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _keyword_hits(problems: Sequence[str], keywords: Sequence[str]) -> list[str]:
    haystack = "\n".join(str(problem).casefold() for problem in problems)
    return [keyword for keyword in keywords if keyword and str(keyword).casefold() in haystack]


def _low_scores(scores: ScoreSet, threshold: int) -> list[str]:
    return [name for name, score in scores.to_dict().items() if score < threshold]


def apply_local_rules(
    analysis: ClassificationResult,
    rules: LocalRulesConfig | Mapping[str, Any] | object | None = None,
) -> ClassificationResult:
    """Apply conservative local rules to a normalised VLM result.

    The VLM's FAIL is never relaxed.  A VLM PASS is promoted to REVIEW for low
    confidence, suspicious problem text, or low scores; severe configured
    keywords and multiple very low scores promote it to FAIL.  This keeps the
    automatic sorter biased toward human review instead of accidental loss.
    """

    if not isinstance(analysis, ClassificationResult):
        analysis = ClassificationResult.from_mapping(analysis)
    config = rules if isinstance(rules, LocalRulesConfig) else LocalRulesConfig.from_settings(rules)

    model_result = analysis.result
    result = model_result
    fail_hits = _keyword_hits(analysis.problems, config.fail_problem_keywords)
    review_hits = _keyword_hits(analysis.problems, config.review_problem_keywords)
    low_fail = _low_scores(analysis.scores, config.score_fail_below)
    low_review = _low_scores(analysis.scores, config.score_review_below)

    # Explicit model FAIL and explicit severe local evidence always win.
    if model_result is ResultLabel.FAIL:
        result = ResultLabel.FAIL
    elif fail_hits or len(low_fail) >= config.fail_score_count:
        result = ResultLabel.FAIL
    elif model_result is ResultLabel.REVIEW:
        result = ResultLabel.REVIEW
    elif (
        analysis.confidence < config.threshold_pass
        or analysis.confidence < config.threshold_review
        or review_hits
        or low_review
    ):
        result = ResultLabel.REVIEW
    else:
        result = ResultLabel.PASS

    return analysis.copy_with(result=result)


# A short alias is useful in tests and for future rule stages.
apply_rules = apply_local_rules


class ImageAnalysisClient(Protocol):
    def classify_image(self, image: str | Path, *, image_name: str | None = None) -> ClassificationResult: ...


@dataclass(slots=True)
class ImageClassifier:
    """Classify one file through the selected backend, then apply local rules."""

    client: ImageAnalysisClient
    rules: LocalRulesConfig = field(default_factory=LocalRulesConfig)
    fail_safe: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.rules, LocalRulesConfig):
            self.rules = LocalRulesConfig.from_settings(self.rules)

    def classify(self, image: str | Path, *, image_name: str | None = None) -> ClassificationResult:
        """Return the final PASS/REVIEW/FAIL decision for one image.

        API/decoding errors become REVIEW by default.  Set ``fail_safe=False``
        when a caller wants transport errors to propagate to its own retry or
        queue handling.
        """

        path = Path(image)
        try:
            raw = self.client.classify_image(path, image_name=image_name or path.name)
            return apply_local_rules(raw, self.rules)
        except Exception as exc:
            if not self.fail_safe:
                raise
            return ClassificationResult(
                result=ResultLabel.REVIEW,
                confidence=0.0,
                scores=ScoreSet(),
                problems=[f"classifier error: {type(exc).__name__}"],
                summary="The image could not be classified; manual review is required.",
            )

    classify_file = classify

    def __call__(self, image: str | Path, *, image_name: str | None = None) -> ClassificationResult:
        return self.classify(image, image_name=image_name)


Classifier = ImageClassifier


__all__ = [
    "Classifier",
    "DEFAULT_FAIL_KEYWORDS",
    "DEFAULT_REVIEW_KEYWORDS",
    "ImageClassifier",
    "LocalRuleConfig",
    "LocalRulesConfig",
    "apply_local_rules",
    "apply_rules",
]
