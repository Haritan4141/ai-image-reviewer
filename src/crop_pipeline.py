"""Optional full -> localization -> selected crops -> conservative merge pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from .crop_utils import CropTooSmallError, CropWorkspace
from .models import ClassificationResult, RegionCheckResult, RegionKind, ResultLabel, ScoreSet
from .region_detection import DetectionResult, DetectedRegion, RegionDetector, create_region_detector

if TYPE_CHECKING:
    from .classifier import LocalRulesConfig
    from .config import CropRecheckConfig


PIPELINE_VERSION = "crop-recheck-v1"
SUPPORTED_TARGETS = (RegionKind.FACE, RegionKind.HAND, RegionKind.FOOT)
_SEVERITY = {ResultLabel.PASS: 0, ResultLabel.REVIEW: 1, ResultLabel.FAIL: 2}
_FOOT_WORDS = ("foot", "feet", "toe", "足", "つま先")
_SMALL_REGION_WORDS = (
    "small face", "small hand", "tiny face", "tiny hand", "distant person", "distant people",
    "small figures", "顔が小", "手が小", "小さな顔", "小さな手", "遠くの人物",
)


@dataclass(slots=True)
class CropPlan:
    regions: list[DetectedRegion] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _contains(text: str, keywords: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(word.casefold() in folded for word in keywords if word)


def needs_crop_recheck(full: ClassificationResult, config: "CropRecheckConfig", rules: "LocalRulesConfig") -> bool:
    if not config.enabled or full.result is ResultLabel.FAIL:
        return False
    if not any(config.targets[kind.value].enabled for kind in SUPPORTED_TARGETS):
        return False
    if config.planner.run_on_review_only and full.result is not ResultLabel.REVIEW:
        return False
    if config.mode != "fast":
        return True
    planner = config.planner
    return (
        full.result is ResultLabel.REVIEW
        or full.confidence < planner.min_confidence_for_skip
        or any(full.scores[name] < rules.score_review_below for name in planner.trigger_low_scores)
        or _contains("\n".join(full.problems), planner.review_problem_keywords)
        or _contains(full.summary, _SMALL_REGION_WORDS)
    )


def plan_regions(
    full: ClassificationResult, detection: DetectionResult,
    config: "CropRecheckConfig", rules: "LocalRulesConfig",
) -> CropPlan:
    plan = CropPlan(issues=list(detection.problems))
    planner = config.planner
    confident = detection.confidence >= planner.min_detector_confidence
    if not confident:
        plan.issues.append("localization confidence is below the detector threshold")
    if detection.person_present is None:
        plan.issues.append("person presence is uncertain")
    if detection.person_present is False and confident and not detection.problems:
        if config.mode != "balanced" or config.detectors.person_required_for_balanced:
            plan.notes.append("no visible person detected; anatomy crops not required")
            return plan

    kinds = [kind for kind in SUPPORTED_TARGETS if config.targets[kind.value].enabled]
    foot_needed = (
        config.mode == "strict"
        or full.scores.anatomy < rules.score_review_below
        or _contains("\n".join(full.problems), _FOOT_WORDS)
        or any(region.kind is RegionKind.FOOT and region.box.area >= planner.large_foot_area
               for region in detection.regions)
    )
    if not foot_needed and RegionKind.FOOT in kinds:
        kinds.remove(RegionKind.FOOT)
        plan.notes.append("foot recheck not triggered in this mode")

    for kind in kinds:
        candidates = sorted(
            (region for region in detection.regions if region.kind is kind),
            key=lambda region: (-region.confidence, region.box.x1, region.box.y1),
        )
        unique: list[DetectedRegion] = []
        for region in candidates:
            if any(region.box.iou(other.box) >= planner.dedup_iou for other in unique):
                continue
            unique.append(region)
        if not unique:
            if kind in detection.not_visible and confident and not detection.problems:
                plan.notes.append(f"{kind.value} not visible; occlusion/cropping is not a defect")
            else:
                plan.issues.append(f"{kind.value} region missing or localization uncertain")
            continue
        valid = []
        for region in unique:
            if region.confidence < planner.min_detector_confidence:
                plan.issues.append(f"{kind.value} candidate has low detector confidence")
            else:
                valid.append(region)
        limit = getattr(planner, f"max_{kind.value}_crops")
        if len(valid) > limit:
            plan.issues.append(f"{kind.value} crop limit reached; {len(valid) - limit} regions uninspected")
        # Confidence selects the bounded subset; spatial order supplies stable indices.
        plan.regions.extend(sorted(valid[:limit], key=lambda region: (region.box.x1, region.box.y1)))
    if not kinds:
        plan.notes.append("all supported targets disabled or not triggered")
    return plan


def merge_crop_verdicts(
    full: ClassificationResult, crops: list[RegionCheckResult], *, mode: str = "balanced",
    issues: list[str] | None = None, notes: list[str] | None = None,
    rules: "LocalRulesConfig | None" = None, min_detector_confidence: float = 0.8,
    stage: str = "crop_merge",
) -> ClassificationResult:
    """Monotone merge: crop PASS cannot erase any full-image concern.

    FAIL requires high-confidence, corroborated severe evidence in a crop.
    Otherwise a crop FAIL is a REVIEW flag, not proof that the whole image fails.
    """
    from .classifier import LocalRulesConfig

    rule_config = rules or LocalRulesConfig()
    final = full.result
    problems = list(full.problems)
    reasons = list(full.rule_reasons)
    checks: list[RegionCheckResult] = []
    uncertain = list(issues or [])
    for crop in crops:
        label = crop.result
        crop_reasons = list(crop.rule_reasons)
        low_confidence = crop.confidence < rule_config.threshold_pass
        low_detector = crop.detector_confidence < min_detector_confidence
        fail_hit = _contains("\n".join(crop.problems), rule_config.fail_problem_keywords)
        low_count = sum(score < rule_config.score_fail_below for score in crop.scores.to_dict().values()) if crop.scores else int(crop.score < rule_config.score_fail_below)
        severe = fail_hit and low_count >= rule_config.fail_score_count
        if label is ResultLabel.FAIL and (low_confidence or low_detector or not severe):
            label = ResultLabel.REVIEW
            crop_reasons.append("crop FAIL lacks confident corroborated severe evidence; manual review required")
        if label is ResultLabel.PASS and (
            low_confidence or low_detector or crop.box is None
            or crop.score < rule_config.score_review_below
            or _contains("\n".join(crop.problems), rule_config.review_problem_keywords)
            or fail_hit
        ):
            label = ResultLabel.REVIEW
            crop_reasons.append("crop has low confidence, low score, missing location or problem-keyword evidence")
        check = crop.copy_with(
            result=label, rule_reasons=crop_reasons,
            decision_source="crop_rules" if label is not crop.result else crop.decision_source,
        )
        checks.append(check)
        if _SEVERITY[label] > _SEVERITY[final]:
            final = label
        if label is not ResultLabel.PASS:
            prefix = f"{crop.kind.value}[{crop.index}]"
            reasons.append(f"{prefix}: {label.value} ({check.decision_source})")
            problems.extend(f"{prefix}: {problem}" for problem in (crop.problems or [crop.summary or label.value]))
    if uncertain:
        if final is ResultLabel.PASS:
            final = ResultLabel.REVIEW
        reasons.extend(uncertain)
        problems.extend(f"crop recheck: {issue}" for issue in uncertain)
    reasons.extend(notes or [])
    changed = final is not full.result or bool(uncertain)
    confidence = min([full.confidence] + [check.confidence for check in checks if check.result is not ResultLabel.PASS])
    if uncertain:
        confidence = min(confidence, rule_config.threshold_review)
    return full.copy_with(
        result=final, confidence=confidence, crop_checks=checks,
        full_result_before_merge=full.result.value, crop_mode=mode,
        pipeline_stage=stage, pipeline_version=PIPELINE_VERSION,
        decision_source="crop_merge" if changed and full.result is not ResultLabel.FAIL else full.decision_source,
        rule_reasons=list(dict.fromkeys(reasons)), problems=list(dict.fromkeys(problems)),
        local_rules_applied=True,
    )


class CropRecheckPipeline:
    def __init__(
        self, client: object, config: "CropRecheckConfig", rules: "LocalRulesConfig", *,
        detector: RegionDetector | None = None, stop_requested: Callable[[], bool] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.client, self.config, self.rules = client, config, rules
        self.detector = detector or create_region_detector(client, config.detectors)
        self.stop_requested = stop_requested or (lambda: False)
        self.logger = logger or logging.getLogger(__name__)

    def inspect(self, image: Path, full: ClassificationResult) -> ClassificationResult:
        from .classifier import apply_local_rules

        merge_args = dict(mode=self.config.mode, rules=self.rules,
                          min_detector_confidence=self.config.planner.min_detector_confidence)
        if not needs_crop_recheck(full, self.config, self.rules):
            return merge_crop_verdicts(full, [], notes=["crop recheck not triggered"], stage="full_only", **merge_args)
        if self.stop_requested():
            return merge_crop_verdicts(full, [], issues=["crop recheck cancelled before localization"], stage="cancelled", **merge_args)
        self.logger.info("クロップ領域検出: %s / mode=%s", image.name, self.config.mode)
        try:
            detection = self.detector.detect_regions(image)
            if not isinstance(detection, DetectionResult):
                raise ValueError("detector must return DetectionResult")
            plan = plan_regions(full, detection, self.config, self.rules)
            self.logger.info("クロップ計画: %s / %d領域 / 確認上の注意=%d",
                             image.name, len(plan.regions), len(plan.issues))
        except Exception as exc:
            return merge_crop_verdicts(full, [], issues=[f"detector failure: {type(exc).__name__}"], stage="detection_failed", **merge_args)
        if not plan.regions:
            return merge_crop_verdicts(full, [], issues=plan.issues, notes=plan.notes,
                                       stage="detection_failed" if plan.issues else "detection_complete", **merge_args)
        checks = []
        cancelled = False
        try:
            with CropWorkspace(image, self.config.crop_cache_dir, keep=self.config.keep_crop_files) as workspace:
                indices: dict[RegionKind, int] = {}
                for region in plan.regions:
                    if self.stop_requested():
                        plan.issues.append("crop recheck cancelled; remaining regions uninspected")
                        cancelled = True
                        break
                    index = indices.get(region.kind, 0)
                    indices[region.kind] = index + 1
                    crop_path = None
                    box = region.box
                    try:
                        path, box = workspace.generate(
                            region.box, region.kind, index, padding=self.config.planner.crop_padding_ratio,
                            min_size=self.config.planner.min_crop_size,
                        )
                        if self.config.keep_crop_files:
                            crop_path = str(path)
                        self.logger.info("クロップ再判定: %s / %s[%d]", image.name, region.kind.value, index)
                        raw = self.client.classify_image(path, target=region.kind.value, region_index=index, image_name=path.name)
                        if not isinstance(raw, ClassificationResult):
                            raw = ClassificationResult.from_model_mapping(raw)
                        # Crop composition and unrelated anatomy are out of scope.
                        # Neutralize them even if a model ignores the scope prompt.
                        scoped_scores = raw.scores.to_dict()
                        relevant = {"anatomy", "artifacts", {RegionKind.FACE: "face", RegionKind.HAND: "hands", RegionKind.FOOT: "anatomy"}[region.kind]}
                        for name in scoped_scores:
                            if name not in relevant:
                                scoped_scores[name] = 10
                        raw = raw.copy_with(scores=ScoreSet.from_mapping(scoped_scores))
                        checked = apply_local_rules(raw, self.rules)
                        score_name = {RegionKind.FACE: "face", RegionKind.HAND: "hands", RegionKind.FOOT: "anatomy"}[region.kind]
                        label = checked.result
                        reasons = list(checked.rule_reasons)
                        if label is ResultLabel.PASS and _contains("\n".join(checked.problems), self.config.planner.review_problem_keywords):
                            label = ResultLabel.REVIEW
                            reasons.append("crop problem words require manual review")
                        check = RegionCheckResult(
                            kind=region.kind, index=index, box=box, result=label,
                            model_result=checked.model_result, confidence=checked.confidence,
                            score=checked.scores[score_name], scores=checked.scores,
                            problems=checked.problems, summary=checked.summary,
                            decision_source=checked.decision_source if label is checked.result else "crop_rules",
                            rule_reasons=reasons, detector_name=region.detector_name,
                            detector_confidence=region.confidence, crop_path=crop_path,
                        )
                    except Exception as exc:
                        reason = (
                            f"crop too small (minimum width/height {self.config.planner.min_crop_size}px)"
                            if isinstance(exc, CropTooSmallError)
                            else f"crop inspection failed: {type(exc).__name__}"
                        )
                        check = RegionCheckResult(
                            kind=region.kind, index=index, box=box, result=ResultLabel.REVIEW,
                            confidence=0, score=5, problems=[reason], summary="Region needs manual review.",
                            decision_source="crop_error", rule_reasons=[reason], detector_name=region.detector_name,
                            detector_confidence=region.confidence, crop_path=crop_path,
                        )
                    checks.append(check)
                    self.logger.info("クロップ結果: %s / %s[%d] -> %s (%.2f)",
                                     image.name, region.kind.value, index, check.result.value, check.confidence)
        except Exception as exc:
            plan.issues.append(f"crop workspace failure: {type(exc).__name__}")
        return merge_crop_verdicts(
            full, checks, issues=plan.issues, notes=plan.notes,
            stage="cancelled" if cancelled else "crop_merge", **merge_args,
        )
