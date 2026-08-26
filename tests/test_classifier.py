from __future__ import annotations

from pathlib import Path

from src.classifier import ImageClassifier, LocalRulesConfig, apply_local_rules
from src.models import ClassificationResult, ResultLabel, ScoreSet


def _result(
    label: ResultLabel = ResultLabel.PASS,
    *,
    confidence: float = 0.95,
    problems: list[str] | None = None,
    scores: ScoreSet | None = None,
) -> ClassificationResult:
    return ClassificationResult(
        result=label,
        confidence=confidence,
        scores=scores or ScoreSet(9, 9, 9, 9, 9),
        problems=problems or [],
        summary="test",
    )


def test_low_confidence_pass_is_promoted_to_review() -> None:
    result = apply_local_rules(_result(confidence=0.84))

    assert result.result is ResultLabel.REVIEW


def test_explicit_fail_is_never_relaxed() -> None:
    result = apply_local_rules(_result(ResultLabel.FAIL, confidence=1.0))

    assert result.result is ResultLabel.FAIL


def test_severe_keyword_or_multiple_low_scores_promotes_to_fail() -> None:
    keyword_result = apply_local_rules(_result(problems=["severe deformation detected"]))
    score_result = apply_local_rules(_result(scores=ScoreSet(2, 2, 8, 8, 8)))

    assert keyword_result.result is ResultLabel.FAIL
    assert score_result.result is ResultLabel.FAIL


def test_review_keyword_promotes_high_confidence_pass_to_review() -> None:
    result = apply_local_rules(_result(problems=["possible deformed hand"]))

    assert result.result is ResultLabel.REVIEW


def test_custom_rules_are_applied_without_mutating_input() -> None:
    source = _result(confidence=0.75)
    rules = LocalRulesConfig(threshold_pass=0.70, threshold_review=0.50)

    result = apply_local_rules(source, rules)

    assert source.result is ResultLabel.PASS
    assert result.result is ResultLabel.PASS


class _Client:
    def __init__(self, result: ClassificationResult | None = None, error: Exception | None = None) -> None:
        self.result = result or _result()
        self.error = error
        self.calls: list[tuple[Path, str | None]] = []

    def classify_image(self, image: str | Path, *, image_name: str | None = None) -> ClassificationResult:
        self.calls.append((Path(image), image_name))
        if self.error:
            raise self.error
        return self.result


def test_image_classifier_delegates_and_applies_rules(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"not decoded by the fake client")
    client = _Client(_result(confidence=0.2))

    result = ImageClassifier(client).classify(image)

    assert result.result is ResultLabel.REVIEW
    assert client.calls == [(image, "sample.png")]


def test_image_classifier_is_fail_safe_by_default(tmp_path: Path) -> None:
    image = tmp_path / "broken.png"
    client = _Client(error=RuntimeError("API unavailable"))

    result = ImageClassifier(client).classify(image)

    assert result.result is ResultLabel.REVIEW
    assert result.confidence == 0.0
    assert result.problems == ["classifier error: RuntimeError"]
