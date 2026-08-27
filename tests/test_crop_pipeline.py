from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

import pytest
from PIL import Image

from src.classifier import ImageClassifier, LocalRulesConfig
from src.config import CropRecheckConfig, CropPlannerConfig
from src.crop_pipeline import merge_crop_verdicts, needs_crop_recheck
from src.models import ClassificationResult, CropBox, RegionCheckResult, RegionKind, ResultLabel, ScoreSet
from src.region_detection import DetectedRegion, DetectionResult


def verdict(label="PASS", confidence=.98, problems=None, scores=None):
    return ClassificationResult(result=label, confidence=confidence,
                                scores=scores or ScoreSet(9, 9, 9, 9, 9),
                                problems=problems or [], summary="A person is visible.")


def crop(label="PASS", **kwargs):
    values = dict(kind="hand", index=0, box=[.1, .2, .5, .6], result=label, score=9,
                  confidence=.95, detector_confidence=.95, scores=ScoreSet(9, 9, 9, 9, 9))
    values.update(kwargs)
    return RegionCheckResult(**values)


@pytest.mark.parametrize(("full", "region", "expected"), [
    ("PASS", "REVIEW", "REVIEW"), ("REVIEW", "PASS", "REVIEW"), ("FAIL", "PASS", "FAIL"),
    ("PASS", "PASS", "PASS"), ("PASS", "FAIL", "REVIEW"),
])
def test_monotone_merge(full, region, expected):
    merged = merge_crop_verdicts(verdict(full), [crop(region)])
    assert merged.result.value == expected
    assert merged.full_result_before_merge == full
    assert merged.crop_checks[0].kind is RegionKind.HAND
    assert merged.pipeline_version


def test_only_confident_corroborated_crop_failure_can_fail():
    severe = crop("FAIL", score=1, problems=["severe deformation of visible fingers"],
                  scores=ScoreSet(1, 1, 10, 9, 10))
    assert merge_crop_verdicts(verdict(), [severe]).result is ResultLabel.FAIL
    assert merge_crop_verdicts(verdict(), [severe.copy_with(confidence=.2)]).result is ResultLabel.REVIEW
    assert merge_crop_verdicts(verdict(), [severe.copy_with(detector_confidence=.1)]).result is ResultLabel.REVIEW


@pytest.mark.parametrize("change", [dict(confidence=.4), dict(score=2), dict(detector_confidence=.2),
                                    dict(problems=["fused finger"]), dict(box=None)])
def test_crop_pass_with_uncertain_evidence_is_review(change):
    assert merge_crop_verdicts(verdict(), [crop(**change)]).result is ResultLabel.REVIEW


class Client:
    def __init__(self, full=None, crop_result=None, detection=None):
        self.full = full or verdict()
        self.crop_result = crop_result or verdict()
        self.detection = detection or {
            "person_present": True, "confidence": .97, "not_visible": [],
            "regions": [
                {"kind": "face", "box": [.1, .1, .3, .3], "confidence": .95},
                {"kind": "hand", "box": [.5, .3, .7, .5], "confidence": .95},
                {"kind": "foot", "box": [.4, .8, .55, .95], "confidence": .95},
            ],
        }
        self.calls = []

    def classify_image(self, image, *, image_name=None, target="full", region_index=None):
        assert Path(image).is_file()
        self.calls.append(target)
        if target == "full":
            return self.full
        if isinstance(self.crop_result, Exception):
            raise self.crop_result
        return self.crop_result

    def locate_regions(self, image, *, image_name=None):
        self.calls.append("detect")
        if isinstance(self.detection, Exception):
            raise self.detection
        return self.detection


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (1000, 1000), "blue").save(path)
    return path


def config(image, mode="balanced", **changes):
    return CropRecheckConfig(enabled=True, mode=mode, crop_cache_dir=image.parent / "crops", **changes)


@pytest.mark.parametrize(("mode", "expected"), [
    ("fast", ["full"]),
    ("balanced", ["full", "detect", "face", "hand"]),
    ("strict", ["full", "detect", "face", "hand", "foot"]),
])
def test_modes_share_backend_and_keep_scanner_contract(image, mode, expected):
    client = Client()
    result = ImageClassifier(client, crop_config=config(image, mode)).classify(image)
    assert result.result is ResultLabel.PASS
    assert client.calls == expected
    assert result.crop_mode == mode
    assert len(result.crop_checks) == max(0, len(expected) - 2)
    assert all(Path(check.crop_path).exists() for check in result.crop_checks)


def test_disabled_pipeline_preserves_full_only(image):
    client = Client()
    result = ImageClassifier(client, crop_config=CropRecheckConfig()).classify(image)
    assert client.calls == ["full"]
    assert result.pipeline_version is None


@pytest.mark.parametrize("full", [
    verdict("REVIEW"), verdict(confidence=.85), verdict(scores=ScoreSet(4, 9, 9, 9, 9)),
    verdict(problems=["finger concern"]), verdict().copy_with(summary="A distant person with small hands."),
])
def test_fast_triggers_on_review_uncertainty_scores_words_or_small_parts(image, full):
    client = Client(full=full)
    ImageClassifier(client, crop_config=config(image, "fast")).classify(image)
    assert "detect" in client.calls


@pytest.mark.parametrize("full,large", [(verdict(scores=ScoreSet(3, 9, 9, 9, 9)), False),
                                      (verdict(problems=["fused toes"]), False), (verdict(), True)])
def test_balanced_foot_is_conditional(image, full, large):
    client = Client(full=full)
    if large:
        client.detection["regions"][2]["box"] = [.2, .6, .6, .9]
    ImageClassifier(client, crop_config=config(image)).classify(image)
    assert "foot" in client.calls


def test_no_person_or_confidently_occluded_parts_do_not_become_defects(image):
    for detection in (
        {"person_present": False, "confidence": .98, "not_visible": [], "regions": []},
        {"person_present": True, "confidence": .98, "not_visible": ["face", "hand", "foot"], "regions": []},
    ):
        client = Client(detection=detection)
        result = ImageClassifier(client, crop_config=config(image, "strict")).classify(image)
        assert result.result is ResultLabel.PASS
        assert client.calls == ["full", "detect"]


@pytest.mark.parametrize("detection", [
    RuntimeError("offline"), {}, {"person_present": True, "confidence": .99, "regions": [], "not_visible": []},
    {"person_present": False, "confidence": .2, "regions": [], "not_visible": []},
])
def test_detection_failure_is_review(image, detection):
    client = Client()
    client.detection = detection
    result = ImageClassifier(client, crop_config=config(image)).classify(image)
    assert result.result is ResultLabel.REVIEW
    assert result.full_result_before_merge == "PASS"
    assert result.decision_source == "crop_merge"


@pytest.mark.parametrize("crop_result", [ValueError("invalid JSON"), {"result": "PASS"}, verdict("REVIEW")])
def test_crop_failure_or_invalid_json_is_recorded(image, crop_result):
    client = Client(crop_result=crop_result)
    result = ImageClassifier(client, crop_config=config(image)).classify(image)
    assert result.result is ResultLabel.REVIEW
    assert len(result.crop_checks) == 2
    assert all(check.result is ResultLabel.REVIEW for check in result.crop_checks)


def test_duplicate_regions_are_suppressed_but_crop_limit_and_small_regions_are_review(image):
    client = Client()
    hand = client.detection["regions"][1]
    client.detection["regions"].append(dict(hand))
    result = ImageClassifier(client, crop_config=config(image)).classify(image)
    assert client.calls.count("hand") == 1
    assert result.result is ResultLabel.PASS
    client.calls.clear()
    client.detection["regions"].append({"kind": "hand", "box": [.8, .1, .95, .3], "confidence": .96})
    cfg = config(image, planner=CropPlannerConfig(max_hand_crops=1))
    result = ImageClassifier(client, crop_config=cfg).classify(image)
    assert result.result is ResultLabel.REVIEW
    assert any("limit reached" in reason for reason in result.rule_reasons)
    client.detection["regions"][0]["box"] = [.1, .1, .11, .11]
    result = ImageClassifier(client, crop_config=config(image)).classify(image)
    assert result.result is ResultLabel.REVIEW
    assert result.crop_checks[0].decision_source == "crop_error"


def test_low_detector_confidence_is_not_silently_dropped(image):
    client = Client()
    client.detection["regions"][1]["confidence"] = .2
    result = ImageClassifier(client, crop_config=config(image)).classify(image)
    assert result.result is ResultLabel.REVIEW
    assert "hand" not in client.calls


def test_keep_false_and_stop_between_crops(image):
    client = Client()
    stop = lambda: "face" in client.calls
    result = ImageClassifier(client, crop_config=config(image, keep_crop_files=False), stop_requested=stop).classify(image)
    assert client.calls == ["full", "detect", "face"]
    assert result.result is ResultLabel.REVIEW
    assert result.pipeline_stage == "cancelled"
    assert all(check.crop_path is None for check in result.crop_checks)
    assert not list((image.parent / "crops").rglob("*.png"))


def test_full_fail_short_circuits_and_review_is_not_raised_to_pass(image):
    client = Client(full=verdict("FAIL", problems=["severe deformation"], scores=ScoreSet(1, 1, 9, 9, 9)))
    result = ImageClassifier(client, crop_config=config(image)).classify(image)
    assert result.result is ResultLabel.FAIL
    assert client.calls == ["full"]
    client = Client(full=verdict("REVIEW"))
    result = ImageClassifier(client, crop_config=config(image)).classify(image)
    assert result.result is ResultLabel.REVIEW
    assert all(check.result is ResultLabel.PASS for check in result.crop_checks)


def test_review_only_override_and_custom_detector_interface(image):
    cfg = config(image, planner=CropPlannerConfig(run_on_review_only=True))
    assert not needs_crop_recheck(verdict(), cfg, LocalRulesConfig())
    class Detector:
        def detect_regions(self, path):
            return DetectionResult(
                regions=[DetectedRegion(RegionKind.HAND, CropBox(.1, .1, .5, .5), .95)],
                person_present=True, confidence=.95, not_visible={RegionKind.FACE},
            )
    client = Client()
    result = ImageClassifier(client, crop_config=config(image), detector=Detector()).classify(image)
    assert result.result is ResultLabel.PASS
    assert client.calls == ["full", "hand"]


def test_none_detector_falls_back_without_crashing(image):
    cfg = config(image)
    cfg = replace(cfg, detectors=replace(cfg.detectors, provider="none"))
    client = Client()
    result = ImageClassifier(client, crop_config=cfg).classify(image)
    assert result.result is ResultLabel.REVIEW
    assert client.calls == ["full"]


def test_unrelated_crop_scores_are_neutralized_before_local_rules(image):
    # A crop-only composition complaint cannot manufacture severe anatomy proof.
    client = Client(crop_result=verdict("PASS", scores=ScoreSet(9, 9, 9, 9, 1)))
    result = ImageClassifier(client, crop_config=config(image)).classify(image)
    assert result.result is ResultLabel.PASS
    assert all(check.scores.composition == 10 for check in result.crop_checks)


def test_crop_progress_uses_the_application_logger(image, caplog):
    logger = logging.getLogger("test.crop-progress")
    with caplog.at_level(logging.INFO, logger=logger.name):
        ImageClassifier(Client(), crop_config=config(image), logger=logger).classify(image)
    assert "クロップ領域検出" in caplog.text
    assert "クロップ計画" in caplog.text
    assert "hand[0]" in caplog.text
