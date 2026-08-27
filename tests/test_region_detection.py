from pathlib import Path

from src.models import RegionKind
from src.region_detection import DetectionResult, VLMRegionDetector, UnavailableRegionDetector


def test_normalizes_multiple_people_and_clamps_boxes() -> None:
    detection = DetectionResult.from_mapping({
        "person_present": True, "confidence": .9, "not_visible": ["foot"],
        "regions": [
            {"kind": "hand", "box": [-.1, 0, .2, .3], "confidence": .9},
            {"kind": "hand", "box": [.6, .5, 1.1, .9], "confidence": .95},
        ],
    })
    assert not detection.problems
    assert len(detection.regions) == 2
    assert detection.not_visible == {RegionKind.FOOT}


def test_invalid_localization_is_not_silently_no_person() -> None:
    result = DetectionResult.from_mapping({"person_present": "false", "confidence": float("nan"),
                                         "regions": [{"kind": "hand", "box": [1, 0, 0, 1], "confidence": .9}]})
    assert result.person_present is None
    assert result.confidence == 0
    assert result.problems
    assert not result.regions


def test_contradictory_detection_records_uncertainty() -> None:
    result = DetectionResult.from_mapping({
        "person_present": False, "confidence": .99, "not_visible": ["face"],
        "regions": [{"kind": "face", "box": [0, 0, .5, .5], "confidence": .95}],
    })
    assert result.person_present is None
    assert result.problems
    assert not result.not_visible


def test_unavailable_detector_and_transport_failures_are_safe() -> None:
    class Broken:
        def locate_regions(self, *args, **kwargs):
            raise RuntimeError("not logged")

    assert VLMRegionDetector(Broken()).detect_regions(Path("image.png")).problems
    assert UnavailableRegionDetector().detect_regions(Path("image.png")).person_present is None
