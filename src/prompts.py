"""Profile-aware prompts for full-frame and target-specific VLM inspection.

The prompt module owns the wire contracts shared by both model backends. A
crop still returns the original five-score classification object; the target
instruction changes what is considered observable, not the result shape.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


JSON_SCHEMA = """{
  \"result\": \"PASS\" | \"REVIEW\" | \"FAIL\",
  \"confidence\": 0.0,
  \"scores\": {
    \"anatomy\": 1,
    \"hands\": 1,
    \"face\": 1,
    \"artifacts\": 1,
    \"composition\": 1
  },
  \"problems\": [],
  \"summary\": \"brief explanation\"
}"""


PROMPT_VERSION = "2026-08-27-v3-crops"
REVIEW_MODES = ("lenient", "standard", "strict")
TARGETS = ("full", "face", "hand", "foot", "upper_body", "lower_body")
REGION_KINDS = ("face", "hand", "foot", "upper_body", "lower_body")


SYSTEM_PROMPT = """You are a visual quality inspector for AI-generated illustrations.
Judge unintended generation defects only. Do not judge the subject, adult content,
art style, artistic taste, or whether stylized proportions are realistic.

The following are not defects by themselves: anime or cartoon rendering,
intentional anatomical exaggeration, unusual but plausible poses, foreshortening,
close crops, partially hidden body parts, multiple people overlapping, foreground
or background figures, bodily fluids, text, motion lines, decorative spikes,
glow, highlights, and simplified details. Trace which person or object each visible
part belongs to before calling it duplicated, fused, missing, or malformed. A
cropped or occluded hand, limb, or object is not evidence that it is defective.

Inspect the entire image, paying particular attention to:
- hands and every visible finger (wrong count, fused fingers, malformed palms,
  twisted wrists, impossible poses, and hand-object fusion);
- face structure and eye alignment, pupils, teeth, ears, and facial symmetry;
- anatomy, limbs, joints, feet, extra/duplicated or missing body parts, and
  fused body parts;
- object fusion, objects that melt together, duplicate, or have impossible
  geometry;
- obvious AI noise, smearing, texture repetition, broken edges, and large
  global composition failures.

Only report a problem when a specific unintended defect is actually visible.
For every reported defect, identify the visible region and concrete evidence.
Do not invent hidden anatomy. If a possible defect cannot be distinguished from
intentional stylization, perspective, overlap, or occlusion, do not use FAIL.

{profile_instructions}
"""


CROP_SYSTEM_PROMPT = """You are a visual quality inspector for one target crop from
an AI-generated illustration. Judge unintended generation defects only. Do not
judge the subject, adult content, art style, artistic taste, or whether
stylized proportions are realistic.

Anime or cartoon rendering, intentional anatomical exaggeration, unusual but
plausible poses, foreshortening, partial visibility, overlap, bodily fluids,
text, motion lines, decorative effects, glow, highlights, and simplified detail
are not defects by themselves. Do not invent hidden anatomy or treat pixels
outside this crop as evidence. If a possible defect is ambiguous, choose REVIEW
rather than FAIL.

{profile_instructions}
"""


PROFILE_INSTRUCTIONS = {
    "lenient": """Inspection mode: LENIENT.
Use PASS when the image is usable and no unmistakable major defect is visible.
Accept minor simplification and ambiguous details. Use REVIEW for a clearly visible
but non-critical concern. Use FAIL only for multiple unmistakable major structural
failures that materially break the image.""",
    "standard": """Inspection mode: STANDARD.
Use PASS when no clear unintended defect materially harms the image. Minor
simplification, intentional exaggeration, complex overlap, and uncertain hidden
details are acceptable. Use REVIEW for a concrete possible defect that is visible
but not certain. Use FAIL only for an unambiguous, substantial structural failure
supported by precise visible evidence; uncertainty must be REVIEW, not FAIL.""",
    "strict": """Inspection mode: STRICT.
Inspect small visible details carefully. Use REVIEW for suspicious or uncertain
visible defects. Use FAIL for a clear, substantial generation failure. Use PASS
only when no obvious defect is visible, while still respecting intentional style,
perspective, crop, and overlap.""",
}


USER_PROMPT = """Inspect this AI-generated image using the selected quality mode.
Check the full frame, but do not infer defects in hidden or cropped regions. Distinguish
intentional stylization, overlap, perspective, fluids, text, and decorative effects
from actual generation errors. If a possible defect is visible but uncertain, choose
REVIEW. Choose FAIL only when the defect is clearly and directly visible.

Return JSON only using exactly these keys and value constraints:
{schema}

`confidence` must be a number from 0.0 to 1.0. Every score must be an integer
from 1 (severely broken) to 10 (clean). `problems` must be a short string
array; use an empty array only when no issue is visible. Do not add keys."""


# The localization response deliberately has a separate contract from the
# five-score classification response. In particular, a model must be able to
# say that a kind is confidently absent (``person_present: false`` and
# ``not_visible``) rather than conflating that with detector failure
# (``person_present: null`` and an empty ``not_visible`` list).
LOCALIZATION_JSON_SCHEMA = """{
  \"person_present\": true | false | null,
  \"confidence\": 0.0,
  \"regions\": [
    {
      \"kind\": \"face\" | \"hand\" | \"foot\" | \"upper_body\" | \"lower_body\",
      \"box\": [0.0, 0.0, 1.0, 1.0],
      \"confidence\": 0.0
    }
  ],
  \"not_visible\": [],
  \"summary\": \"brief explanation\"
}"""


# Shared machine-readable schema used by the Codex CLI output constraint and
# by callers that want to inspect the LM Studio contract before sending it.
LOCALIZATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "person_present": {"type": ["boolean", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "regions": {
            "type": "array",
            "maxItems": 32,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(REGION_KINDS)},
                    "box": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["kind", "box", "confidence"],
                "additionalProperties": False,
            },
        },
        "not_visible": {
            "type": "array",
            "items": {"type": "string", "enum": list(REGION_KINDS)},
        },
        "summary": {"type": "string"},
    },
    "required": ["person_present", "confidence", "regions", "not_visible", "summary"],
    "additionalProperties": False,
}

LOCALIZATION_CORRECTION_PROMPT = """Your previous response was not valid JSON in the
required localization schema. Re-inspect the same image and return exactly one
parseable JSON object, with no Markdown or prose. Do not invent boxes. Use
`person_present: false` and list kinds in `not_visible` only when their absence
is confidently visible; use `person_present: null`, an empty `regions` list,
and an empty `not_visible` list when localization is unknown or failed.

Required shape:
""" + LOCALIZATION_JSON_SCHEMA


TARGET_PROMPT_INSTRUCTIONS: dict[str, str] = {
    "full": """Target: FULL FRAME.
Inspect the complete image and all visible people and objects. Score each of
the five dimensions using only evidence visible in the frame. Do not infer
defects in cropped, hidden, or occluded regions. When people are visible, note
in the summary whether faces, hands, or feet are actually visible and large
enough to assess; do not invent a small-part condition when it cannot be seen.""",
    "face": """Target: FACE CROP.
Inspect ONLY the visible face region represented by this image crop. Focus on
eyes and eye alignment, pupils, mouth, teeth, ears, facial symmetry, duplicate
or fused facial parts, and clearly malformed visible facial structure. Do not
infer missing body parts outside the crop or behind occlusion. A partial face
at an image edge is not defective merely because the rest is out of frame.
For unrelated score dimensions (hands, artifacts, and composition when not
directly relevant to the visible face), use the neutral score 10. Use the
anatomy score for visible local face structure only.""",
    "hand": """Target: HAND CROP.
Inspect ONLY the visible hand region represented by this image crop. Focus on
clearly visible finger count, fused or duplicated fingers, palm and joints,
thumb placement, wrist connection, and hand-object fusion. Do not infer a
missing arm or body part outside the crop or behind occlusion. A hand cut by
the crop boundary is not defective merely because the unseen part is absent.
For unrelated score dimensions (face, artifacts, and composition when not
directly relevant to the visible hand), use the neutral score 10. Use the
anatomy score for visible local hand structure only.""",
    "foot": """Target: FOOT CROP.
Inspect ONLY the visible foot or toes represented by this image crop. Focus on
clearly visible toe count, fused or duplicated toes, foot shape, ankle or shoe
connection when visible, and obvious local deformation. Do not infer missing
leg or body parts outside the crop or behind occlusion. A foot cut by the crop
boundary is not defective merely because the unseen part is absent.
For unrelated score dimensions (hands, face, artifacts, and composition when
not directly relevant to the visible foot), use the neutral score 10. Use the
anatomy score for visible local foot structure only.""",
    "upper_body": """Target: UPPER-BODY CROP.
Inspect ONLY the visible upper-body region in this crop. Focus on clearly
visible local anatomy, joints, limb connections, object fusion, and artifacts.
Do not infer hidden or out-of-frame anatomy. Use neutral score 10 for unrelated
dimensions and use REVIEW when a visible issue is ambiguous.""",
    "lower_body": """Target: LOWER-BODY CROP.
Inspect ONLY the visible lower-body region in this crop. Focus on clearly
visible local anatomy, joints, feet connections, object fusion, and artifacts.
Do not infer hidden or out-of-frame anatomy. Use neutral score 10 for unrelated
dimensions and use REVIEW when a visible issue is ambiguous.""",
}


LOCALIZATION_SYSTEM_PROMPT = """You are a conservative region localizer for an
AI-generated image. Locate only regions that are actually visible in the
upright image. Return all visible faces, hands, and feet for every person, plus
upper_body/lower_body only when clearly visible. Use normalized upright
coordinates [x1, y1, x2, y2] in the range 0..1, with x1 < x2 and y1 < y2.
Never fabricate a box for a hidden, cropped, or uncertain part. Do not use
left/right names; identify repeated regions by array order. A confidently
absent kind belongs in `not_visible`. If the image cannot be localized or
presence is unknown, use `person_present: null`, confidence 0, no regions, and
an empty `not_visible` list. Return no more than 32 regions.

Return exactly one JSON object and nothing else:
""" + LOCALIZATION_JSON_SCHEMA


CORRECTION_PROMPT = """Your previous response was not valid JSON in the required
schema. Re-inspect the same image and respond with exactly one parseable JSON
object, with no Markdown or prose. Use only the keys `result`, `confidence`,
`scores`, `problems`, and `summary`; use PASS, REVIEW, or FAIL for `result` and
integer scores from 1 to 10. When uncertain, use REVIEW.

Required shape:
""" + JSON_SCHEMA


def normalize_review_mode(mode: str | None) -> str:
    value = str(mode or "standard").strip().lower()
    return value if value in REVIEW_MODES else "standard"


def normalize_target(target: str | None) -> str:
    """Normalize a target name, rejecting unknown targets explicitly."""

    value = str(target or "full").strip().lower()
    if value not in TARGETS:
        raise ValueError(f"unknown inspection target: {target!r}")
    return value


def build_system_prompt(mode: str = "standard") -> str:
    """Return the backwards-compatible full-frame system instruction."""

    selected = normalize_review_mode(mode)
    return (
        SYSTEM_PROMPT.format(profile_instructions=PROFILE_INSTRUCTIONS[selected])
        + "\n\nReturn exactly one JSON object and nothing else: no Markdown fences, comments, "
        "explanation, or leading/trailing prose. The object must have this shape:\n"
        + JSON_SCHEMA
    )


def build_target_system_prompt(target: str = "full", mode: str = "standard") -> str:
    """Return a full or target-specific classification system prompt."""

    selected_target = normalize_target(target)
    selected_mode = normalize_review_mode(mode)
    if selected_target == "full":
        return build_system_prompt(selected_mode)
    return (
        CROP_SYSTEM_PROMPT.format(profile_instructions=PROFILE_INSTRUCTIONS[selected_mode])
        + "\n\n"
        + TARGET_PROMPT_INSTRUCTIONS[selected_target]
        + "\nThe crop target is the only visual evidence to judge. Return exactly one "
        "JSON object and nothing else; use this five-score shape:\n"
        + JSON_SCHEMA
    )


def _safe_image_name(image_name: str | None) -> str:
    if not image_name:
        return ""
    # Keep this helpful for logs while avoiding arbitrary long paths.
    return str(image_name).replace("\\", "/").rsplit("/", 1)[-1][:200]


def build_user_prompt(image_name: str | None = None, *, mode: str = "standard") -> str:
    """Build the backwards-compatible full-frame user text."""

    suffix = ""
    safe_name = _safe_image_name(image_name)
    if safe_name:
        suffix = f"\nImage filename (metadata only): {safe_name}"
    selected = normalize_review_mode(mode)
    return USER_PROMPT.format(schema=JSON_SCHEMA) + f"\nInspection mode: {selected.upper()}." + suffix


def build_target_user_prompt(
    image_name: str | None = None,
    *,
    target: str = "full",
    mode: str = "standard",
    region_index: int | None = None,
) -> str:
    """Build text for a full image or a target crop.

    ``region_index`` is metadata only. It is intentionally not described as a
    left/right location, because detection order is the only stable identity
    for repeated regions.
    """

    selected_target = normalize_target(target)
    if selected_target == "full":
        return build_user_prompt(image_name, mode=mode)

    suffix = ""
    safe_name = _safe_image_name(image_name)
    if safe_name:
        suffix = f"\nImage filename (metadata only): {safe_name}"
    selected_mode = normalize_review_mode(mode)
    text = (
        "Inspect ONLY the visible target region represented by this crop using the "
        "selected quality mode. Do not infer defects from missing, hidden, or "
        "out-of-frame body parts. If a possible issue is uncertain, choose REVIEW; "
        "choose FAIL only for clear, directly visible evidence.\n\n"
        "Return JSON only using exactly these keys and value constraints:\n"
        + JSON_SCHEMA
        + "\n\n`confidence` must be a number from 0.0 to 1.0. Every score must be an "
        "integer from 1 (severely broken) to 10 (clean). `problems` must be a short "
        "string array; use an empty array only when no issue is visible. Do not add keys."
        + f"\nInspection mode: {selected_mode.upper()}."
        + "\n\n"
        + TARGET_PROMPT_INSTRUCTIONS[selected_target]
    )
    if region_index is not None:
        try:
            index = max(0, int(region_index))
        except (TypeError, ValueError):
            index = 0
        text += f"\nCrop sequence index (metadata only): {index}."
    return text + suffix


def build_messages_for_target(
    image_data_url: str,
    image_name: str | None = None,
    *,
    target: str = "full",
    mode: str = "standard",
    region_index: int | None = None,
) -> list[dict[str, Any]]:
    """Create OpenAI-compatible multimodal messages for one target."""

    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:"):
        raise ValueError("image_data_url must be a data: URL")
    selected_target = normalize_target(target)
    return [
        {"role": "system", "content": build_target_system_prompt(selected_target, mode)},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_target_user_prompt(
                        image_name,
                        target=selected_target,
                        mode=mode,
                        region_index=region_index,
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages(
    image_data_url: str,
    image_name: str | None = None,
    *,
    mode: str = "standard",
) -> list[dict[str, Any]]:
    """Create the backwards-compatible full-frame messages."""

    return build_messages_for_target(image_data_url, image_name, target="full", mode=mode)


def build_full_messages(
    image_data_url: str,
    image_name: str | None = None,
    *,
    mode: str = "standard",
) -> list[dict[str, Any]]:
    return build_messages_for_target(image_data_url, image_name, target="full", mode=mode)


def build_face_crop_messages(
    image_data_url: str,
    image_name: str | None = None,
    *,
    mode: str = "standard",
    region_index: int | None = None,
) -> list[dict[str, Any]]:
    return build_messages_for_target(
        image_data_url, image_name, target="face", mode=mode, region_index=region_index
    )


def build_hand_crop_messages(
    image_data_url: str,
    image_name: str | None = None,
    *,
    mode: str = "standard",
    region_index: int | None = None,
) -> list[dict[str, Any]]:
    return build_messages_for_target(
        image_data_url, image_name, target="hand", mode=mode, region_index=region_index
    )


def build_foot_crop_messages(
    image_data_url: str,
    image_name: str | None = None,
    *,
    mode: str = "standard",
    region_index: int | None = None,
) -> list[dict[str, Any]]:
    return build_messages_for_target(
        image_data_url, image_name, target="foot", mode=mode, region_index=region_index
    )


def build_localization_messages(
    image_data_url: str,
    image_name: str | None = None,
) -> list[dict[str, Any]]:
    """Create messages for the separate region-localization contract."""

    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:"):
        raise ValueError("image_data_url must be a data: URL")
    suffix = ""
    safe_name = _safe_image_name(image_name)
    if safe_name:
        suffix = f"\nImage filename (metadata only): {safe_name}"
    user_text = (
        "Localize visible people and face/hand/foot regions in this upright image. "
        "Return JSON only using exactly this schema:\n"
        + LOCALIZATION_JSON_SCHEMA
        + suffix
    )
    return [
        {"role": "system", "content": LOCALIZATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_correction_messages(
    original_messages: list[dict[str, Any]],
    invalid_response: str | None = None,
    *,
    correction_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Append a bounded JSON-correction turn for a retry.

    ``correction_prompt`` lets localization use its own schema. The default
    remains the five-score classification correction for compatibility.
    """

    excerpt = (invalid_response or "").strip()[:2000]
    text = correction_prompt or CORRECTION_PROMPT
    if excerpt:
        text += f"\nFor context, the invalid response began with:\n{excerpt}"
    messages = list(original_messages)
    messages.append({"role": "user", "content": text})
    return messages


def build_localization_correction_messages(
    original_messages: list[dict[str, Any]],
    invalid_response: str | None = None,
) -> list[dict[str, Any]]:
    return build_correction_messages(
        original_messages,
        invalid_response,
        correction_prompt=LOCALIZATION_CORRECTION_PROMPT,
    )


def _finite(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def normalize_localization_payload(value: object) -> dict[str, Any]:
    """Validate and normalize the shared localization response contract.

    Invalid structures raise ``ValueError`` so the LM Studio repair path can
    ask the model for the localization schema again. Bad individual boxes are
    rejected rather than repaired into fabricated coordinates.
    """

    if not isinstance(value, Mapping):
        raise ValueError("localization payload must be a JSON object")
    required = ("person_present", "confidence", "regions", "not_visible", "summary")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError("localization payload missing fields: " + ", ".join(missing))

    person_present = value.get("person_present")
    if person_present is not None and not isinstance(person_present, bool):
        raise ValueError("person_present must be true, false, or null")
    confidence = _finite(value.get("confidence"), default=-1.0)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("localization confidence must be between 0 and 1")

    regions_value = value.get("regions")
    if not isinstance(regions_value, Sequence) or isinstance(regions_value, (str, bytes)):
        raise ValueError("localization regions must be an array")
    if len(regions_value) > 32:
        raise ValueError("localization regions exceeded the 32-region safety limit")
    regions: list[dict[str, Any]] = []
    for item in regions_value:
        if not isinstance(item, Mapping):
            raise ValueError("localization region must be an object")
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in REGION_KINDS:
            raise ValueError(f"unknown localization region kind: {kind!r}")
        box_value = item.get("box")
        if not isinstance(box_value, Sequence) or isinstance(box_value, (str, bytes)) or len(box_value) != 4:
            raise ValueError("localization region box must contain four coordinates")
        box: list[float] = []
        for coordinate in box_value:
            number = _finite(coordinate, default=-1.0)
            if not 0.0 <= number <= 1.0:
                raise ValueError("localization box coordinates must be between 0 and 1")
            box.append(round(number, 6))
        if not box[0] < box[2] or not box[1] < box[3]:
            raise ValueError("localization region box must have positive width and height")
        region_confidence = _finite(item.get("confidence"), default=-1.0)
        if not 0.0 <= region_confidence <= 1.0:
            raise ValueError("localization region confidence must be between 0 and 1")
        regions.append({"kind": kind, "box": box, "confidence": round(region_confidence, 6)})

    not_visible_value = value.get("not_visible")
    if not isinstance(not_visible_value, Sequence) or isinstance(not_visible_value, (str, bytes)):
        raise ValueError("localization not_visible must be an array")
    not_visible: list[str] = []
    for item in not_visible_value:
        kind = str(item).strip().lower()
        if kind not in REGION_KINDS:
            raise ValueError(f"unknown not_visible region kind: {kind!r}")
        if kind not in not_visible:
            not_visible.append(kind)

    summary = value.get("summary")
    if not isinstance(summary, str):
        raise ValueError("localization summary must be a string")

    return {
        "person_present": person_present,
        "confidence": round(confidence, 6),
        "regions": regions,
        "not_visible": not_visible,
        "summary": summary.strip(),
    }


def unknown_localization(reason: str) -> dict[str, Any]:
    """Return an explicit unknown/failed localization result."""

    clean = str(reason or "localization failed").strip()[:500]
    return {
        "person_present": None,
        "confidence": 0.0,
        "regions": [],
        "not_visible": [],
        "summary": clean or "localization failed",
    }


__all__ = [
    "CROP_SYSTEM_PROMPT",
    "CORRECTION_PROMPT",
    "JSON_SCHEMA",
    "LOCALIZATION_CORRECTION_PROMPT",
    "LOCALIZATION_JSON_SCHEMA",
    "LOCALIZATION_OUTPUT_SCHEMA",
    "LOCALIZATION_SYSTEM_PROMPT",
    "PROFILE_INSTRUCTIONS",
    "PROMPT_VERSION",
    "REGION_KINDS",
    "REVIEW_MODES",
    "SYSTEM_PROMPT",
    "TARGETS",
    "TARGET_PROMPT_INSTRUCTIONS",
    "USER_PROMPT",
    "build_correction_messages",
    "build_face_crop_messages",
    "build_foot_crop_messages",
    "build_full_messages",
    "build_hand_crop_messages",
    "build_localization_correction_messages",
    "build_localization_messages",
    "build_messages",
    "build_messages_for_target",
    "build_system_prompt",
    "build_target_system_prompt",
    "build_target_user_prompt",
    "build_user_prompt",
    "normalize_localization_payload",
    "normalize_review_mode",
    "normalize_target",
    "unknown_localization",
]
