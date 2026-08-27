from __future__ import annotations

import json

import pytest

from src.classifier import apply_local_rules
from src.models import ClassificationResult, CropBox, RegionCheckResult, RegionKind, ResultLabel, ScoreSet


def test_crop_box_clamps_and_measures() -> None:
    box = CropBox(-0.1, 0.2, 1.2, 0.8)
    assert box.to_list() == [0, 0.2, 1, 0.8]
    assert box.width == 1
    assert box.height == pytest.approx(0.6)
    assert box.area == pytest.approx(0.6)
    assert box.iou(box) == pytest.approx(1)
    assert box.padded(0.2).x1 == 0


@pytest.mark.parametrize("value", [
    [0, 0, 0, 1], [0.8, 0, 0.2, 1], [2, 0, 3, 1],
    [0, 0, float("nan"), 1], [0, 0, float("inf"), 1], [0, 0, True, 1], [0, 1],
])
def test_crop_box_rejects_bad_coordinates(value) -> None:
    with pytest.raises(ValueError):
        CropBox.from_mapping(value)


def test_region_result_roundtrip_and_safe_normalization() -> None:
    original = RegionCheckResult(
        kind="hand", index="2", box=[0.1, 0.2, 0.5, 0.6], result="review",
        confidence=.8, score=4, problems=["fused finger"], summary="Visible hand ambiguity",
        detector_name="fake", detector_confidence=.93, crop_path="hand-2.png",
        scores=ScoreSet(4, 4, 10, 9, 10), raw={"secret": "not persisted"},
    )
    data = original.to_dict()
    assert "raw" not in data
    restored = RegionCheckResult.from_mapping(json.loads(json.dumps(data)))
    assert restored.to_dict() == data
    assert restored.kind is RegionKind.HAND
    assert restored.index == 2
    malformed = RegionCheckResult.from_mapping({"kind": "face", "result": "PASS"})
    assert malformed.result is ResultLabel.REVIEW
    assert malformed.confidence == 0


def test_classification_extensions_roundtrip_copy_and_old_format() -> None:
    region = RegionCheckResult(kind="foot", box=[0, 0, .5, .5], score=4, confidence=.9, detector_confidence=.9)
    source = ClassificationResult(
        result="REVIEW", model_result="PASS", confidence=.9, scores=ScoreSet(9, 9, 9, 9, 9),
        summary="full pass, foot needs review", crop_checks=[region],
        full_result_before_merge="PASS", crop_mode="balanced", pipeline_stage="crop_merge",
        pipeline_version="crop-recheck-v1", decision_source="crop_merge",
    )
    restored = ClassificationResult.from_mapping(source.to_dict())
    assert restored.crop_checks[0].to_dict() == region.to_dict()
    assert restored.pipeline_version == "crop-recheck-v1"
    assert restored.full_result_before_merge == "PASS"
    assert apply_local_rules(restored).result is ResultLabel.REVIEW
    copied = source.copy_with(summary="new")
    copied.crop_checks[0].problems.append("new problem")
    assert source.crop_checks[0].problems == []
    old = ClassificationResult.from_mapping({
        "result": "PASS", "confidence": .99, "scores": ScoreSet(9, 9, 9, 9, 9).to_dict(),
        "problems": [], "summary": "ok",
    })
    assert old.crop_checks == []
    assert old.pipeline_version is None
    assert old.result is ResultLabel.PASS


def test_invalid_crops_and_invalid_full_json_cannot_revert_to_pass() -> None:
    valid = {"result": "PASS", "confidence": .99, "scores": ScoreSet(9, 9, 9, 9, 9).to_dict(),
             "problems": [], "summary": "ok"}
    for crop_data in ([{"kind": "invalid"}], "not a list"):
        result = ClassificationResult.from_mapping({**valid, "crop_checks": crop_data})
        assert apply_local_rules(result).result is ResultLabel.REVIEW
    del valid["summary"]
    assert apply_local_rules(ClassificationResult.from_mapping(valid)).result is ResultLabel.REVIEW


def test_contradictory_crop_verdict_and_model_supplied_metadata_are_not_trusted() -> None:
    check = RegionCheckResult(kind="hand", box=[0, 0, 1, 1], result="PASS", model_result="FAIL",
                              score=9, confidence=.99, detector_confidence=.99)
    assert check.result is ResultLabel.REVIEW
    model = ClassificationResult.from_model_mapping({
        "result": "FAIL", "confidence": .99, "scores": ScoreSet(9, 9, 9, 9, 9).to_dict(),
        "problems": [], "summary": "Uncorroborated fail",
        "pipeline_version": "forged", "model_result": "PASS", "local_rules_applied": True,
    })
    assert model.pipeline_version is None
    assert model.model_result is ResultLabel.FAIL
    assert not model.local_rules_applied
    assert apply_local_rules(model).result is ResultLabel.REVIEW
