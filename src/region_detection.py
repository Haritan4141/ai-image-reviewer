"""Replaceable region localization, with an explicit unknown/failure outcome.

The initial provider uses the selected VLM, not a separate detector model. Its
coordinates and confidence are approximate evidence, never proof of correctness.
No fixed grid is silently substituted for a missing face, hand, or foot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Mapping, Protocol

from .models import CropBox, RegionKind


def _confidence(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("confidence must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("confidence must be finite and between zero and one")
    return number


@dataclass(frozen=True, slots=True)
class DetectedRegion:
    kind: RegionKind
    box: CropBox
    confidence: float
    detector_name: str = "custom"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RegionKind.coerce(self.kind))
        object.__setattr__(self, "box", CropBox.from_mapping(self.box))
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(slots=True)
class DetectionResult:
    regions: list[DetectedRegion] = field(default_factory=list)
    person_present: bool | None = None
    confidence: float = 0.0
    not_visible: set[RegionKind] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)
    detector_name: str = "unknown"

    @classmethod
    def from_mapping(cls, payload: object, *, detector_name: str = "vlm") -> "DetectionResult":
        if not isinstance(payload, Mapping):
            raise ValueError("localization response must be a JSON object")
        result = cls(detector_name=detector_name)
        person = payload.get("person_present")
        if "person_present" not in payload or (person is not None and not isinstance(person, bool)):
            result.problems.append("invalid or missing person_present")
        else:
            result.person_present = person
        try:
            result.confidence = _confidence(payload.get("confidence"))
        except ValueError:
            result.problems.append("invalid or missing localization confidence")
        regions = payload.get("regions")
        if not isinstance(regions, list):
            result.problems.append("invalid or missing regions list")
            regions = []
        if len(regions) > 32:
            result.problems.append("localization exceeded the 32-region safety limit")
        for index, entry in enumerate(regions[:32]):
            try:
                if not isinstance(entry, Mapping):
                    raise ValueError("region must be an object")
                result.regions.append(DetectedRegion(
                    kind=RegionKind.coerce(entry.get("kind")),
                    box=CropBox.from_mapping(entry.get("box")),
                    confidence=_confidence(entry.get("confidence")), detector_name=detector_name,
                ))
            except (TypeError, ValueError):
                result.problems.append(f"invalid detected region {index}")
        absent = payload.get("not_visible")
        if not isinstance(absent, list):
            result.problems.append("invalid or missing not_visible list")
        else:
            for value in absent:
                try:
                    result.not_visible.add(RegionKind.coerce(value))
                except ValueError:
                    result.problems.append("unknown not_visible region kind")
        visible_kinds = {region.kind for region in result.regions}
        if result.not_visible & visible_kinds:
            result.problems.append("contradictory visible and not_visible region evidence")
            result.not_visible -= visible_kinds
        if result.person_present is False and result.regions:
            result.problems.append("regions returned despite person_present=false")
            result.person_present = None
        if result.person_present is None and isinstance(payload.get("summary"), str) and payload["summary"].strip():
            result.problems.append("localization incomplete: " + payload["summary"].strip()[:300])
        return result


class RegionDetector(Protocol):
    def detect_regions(self, image: str | Path) -> DetectionResult: ...


class VLMRegionDetector:
    """Locate regions through either backend's common localization contract."""

    def __init__(self, client: object) -> None:
        self.client = client
        self.name = "vlm:" + type(client).__name__

    def detect_regions(self, image: str | Path) -> DetectionResult:
        try:
            locate = getattr(self.client, "locate_regions")
            payload = locate(image, image_name=Path(image).name)
            return DetectionResult.from_mapping(payload, detector_name=self.name)
        except Exception as exc:
            return DetectionResult(
                detector_name=self.name,
                problems=[f"region localization failed: {type(exc).__name__}"],
            )


class UnavailableRegionDetector:
    def __init__(self, reason: str = "region detector is disabled or unavailable") -> None:
        self.reason = reason

    def detect_regions(self, image: str | Path) -> DetectionResult:
        return DetectionResult(detector_name="none", problems=[self.reason])


def create_region_detector(client: object, settings: object) -> RegionDetector:
    provider = getattr(settings, "provider", "auto")
    if provider in {"auto", "vlm"} and callable(getattr(client, "locate_regions", None)):
        return VLMRegionDetector(client)
    if provider == "none":
        return UnavailableRegionDetector("detector provider is none; required regions could not be checked")
    fallback = bool(getattr(settings, "allow_fallback", True))
    return UnavailableRegionDetector(
        "VLM localization unavailable; conservative no-region fallback" if fallback
        else "requested detector unavailable and fallback disabled"
    )
