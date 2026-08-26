from __future__ import annotations

import pytest

from src.models import (
    ClassificationResult,
    ResultLabel,
    ScoreSet,
    decode_json_object,
)


def test_result_label_coerce_is_case_insensitive_and_safe() -> None:
    assert ResultLabel.coerce(" pass ") is ResultLabel.PASS
    assert ResultLabel.coerce("review") is ResultLabel.REVIEW
    assert ResultLabel.coerce("not-a-decision") is ResultLabel.REVIEW


def test_score_set_clamps_and_normalises_values() -> None:
    scores = ScoreSet(anatomy=0, hands=12.6, face="8.0", artifacts=float("nan"), composition=True)

    assert scores.to_dict() == {
        "anatomy": 1,
        "hands": 10,
        "face": 8,
        "artifacts": 5,
        "composition": 5,
    }


def test_classification_result_requires_complete_shape_for_pass() -> None:
    result = ClassificationResult.from_mapping(
        {
            "result": "PASS",
            "confidence": 0.93,
            "scores": {"anatomy": 9, "hands": 8, "face": 9, "artifacts": 9, "composition": 8},
            "problems": [],
            "summary": "No obvious issues found.",
        }
    )

    assert result.result is ResultLabel.PASS
    assert result.confidence == pytest.approx(0.93)
    assert result.scores.hands == 8
    assert result.to_dict()["result"] == "PASS"


def test_classification_result_missing_fields_falls_back_to_review() -> None:
    result = ClassificationResult.from_mapping({"result": "PASS"})

    assert result.result is ResultLabel.REVIEW
    assert result.confidence == 0.0
    assert "missing confidence field" in result.problems
    assert "missing or invalid scores object" in result.problems


def test_pass_requires_problems_array_and_summary_string() -> None:
    result = ClassificationResult.from_mapping(
        {
            "result": "PASS",
            "confidence": 0.99,
            "scores": {"anatomy": 9, "hands": 9, "face": 9, "artifacts": 9, "composition": 9},
        }
    )

    assert result.result is ResultLabel.REVIEW
    assert "missing or invalid problems array" in result.problems
    assert "missing or invalid summary field" in result.problems


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"result":"PASS"}', {"result": "PASS"}),
        ('```json\n{"result":"REVIEW"}\n```', {"result": "REVIEW"}),
        ('The answer is: {"result":"FAIL"}', {"result": "FAIL"}),
    ],
)
def test_decode_json_object_accepts_common_vlm_wrappers(text: str, expected: dict[str, str]) -> None:
    assert decode_json_object(text) == expected


def test_decode_json_object_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        decode_json_object("I cannot inspect this image")
