"""VLM invocation and conservative local safety rules."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

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

    mode: str = "standard"
    threshold_pass: float = 0.80
    threshold_review: float = 0.50
    score_review_below: int = 5
    score_fail_below: int = 2
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
            mode=str(value("mode", defaults.mode)).strip().lower(),
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
    """Apply the selected review profile and retain auditable decision evidence."""

    if not isinstance(analysis, ClassificationResult):
        analysis = ClassificationResult.from_mapping(analysis)
    if analysis.local_rules_applied:
        return analysis
    config = rules if isinstance(rules, LocalRulesConfig) else LocalRulesConfig.from_settings(rules)

    mode = config.mode if config.mode in {"lenient", "standard", "strict"} else "standard"
    model_result = analysis.model_result or analysis.result
    result = analysis.result
    fail_hits = _keyword_hits(analysis.problems, config.fail_problem_keywords)
    review_hits = _keyword_hits(analysis.problems, config.review_problem_keywords)
    score_values = analysis.scores.to_dict()
    low_fail = {name: score for name, score in score_values.items() if score < config.score_fail_below}
    low_review = {name: score for name, score in score_values.items() if score < config.score_review_below}
    corroborated_fail = bool(fail_hits) and len(low_fail) >= config.fail_score_count
    decision_source = analysis.decision_source if analysis.decision_source == "validation" else "model"
    reasons = list(analysis.rule_reasons)

    if analysis.decision_source == "validation":
        result = ResultLabel.REVIEW
    elif analysis.pipeline_version and analysis.result is ResultLabel.FAIL:
        # Re-normalizing a stored merged verdict must not erase crop evidence.
        result = ResultLabel.FAIL
    elif mode == "strict" and (
        model_result is ResultLabel.FAIL
        or fail_hits
        or len(low_fail) >= config.fail_score_count
    ):
        result = ResultLabel.FAIL
        if model_result is not ResultLabel.FAIL:
            decision_source = "local_rules"
            reasons.append("strict mode promoted the result to FAIL")
    elif mode != "strict" and corroborated_fail:
        result = ResultLabel.FAIL
        decision_source = "local_rules"
        reasons.append("independent fail keyword and low-score evidence agreed")
    elif model_result is ResultLabel.FAIL and mode != "strict":
        result = ResultLabel.REVIEW
        decision_source = "local_rules"
        reasons.append(f"uncorroborated model FAIL downgraded to REVIEW in {mode} mode")
    elif analysis.result is ResultLabel.REVIEW or model_result is ResultLabel.REVIEW:
        result = ResultLabel.REVIEW
    elif (
        analysis.confidence < config.threshold_pass
        or review_hits
        or low_review
    ):
        result = ResultLabel.REVIEW
        decision_source = "local_rules"
        if analysis.confidence < config.threshold_pass:
            reasons.append(
                f"confidence {analysis.confidence:.2f} below PASS threshold {config.threshold_pass:.2f}"
            )
        if review_hits:
            reasons.append("review keywords matched: " + ", ".join(review_hits))
        if low_review:
            reasons.append(
                "review-level low scores: "
                + ", ".join(f"{name}={score}" for name, score in low_review.items())
            )
    else:
        result = ResultLabel.PASS

    return analysis.copy_with(
        result=result,
        model_result=model_result,
        decision_source=decision_source,
        low_scores={"review": low_review, "fail": low_fail},
        keyword_hits={"review": review_hits, "fail": fail_hits},
        rule_reasons=reasons,
        review_mode=mode,
        local_rules_applied=True,
    )


# A short alias is useful in tests and for future rule stages.
apply_rules = apply_local_rules


class ImageAnalysisClient(Protocol):
    def classify_image(self, image: str | Path, *, image_name: str | None = None,
                       target: str = "full", region_index: int | None = None) -> ClassificationResult: ...


@dataclass(slots=True)
class ImageClassifier:
    """Classify one file through the selected backend, then apply local rules."""

    client: ImageAnalysisClient
    rules: LocalRulesConfig = field(default_factory=LocalRulesConfig)
    fail_safe: bool = True
    crop_config: object | None = None
    detector: object | None = None
    stop_requested: Callable[[], bool] | None = None
    logger: logging.Logger | None = None

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
            full = apply_local_rules(raw, self.rules)
        except Exception as exc:
            if not self.fail_safe:
                raise
            full = ClassificationResult(
                result=ResultLabel.REVIEW,
                confidence=0.0,
                scores=ScoreSet(),
                problems=[f"classifier error: {type(exc).__name__}"],
                summary="The image could not be classified; manual review is required.",
                model_result=None,
                decision_source="fail_safe",
                rule_reasons=[f"classifier error: {type(exc).__name__}"],
                review_mode=self.rules.mode,
                local_rules_applied=True,
            )
        if self.crop_config is not None and getattr(self.crop_config, "enabled", False):
            from .crop_pipeline import CropRecheckPipeline

            return CropRecheckPipeline(
                self.client, self.crop_config, self.rules,
                detector=self.detector, stop_requested=self.stop_requested, logger=self.logger,
            ).inspect(path, full)
        return full

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
