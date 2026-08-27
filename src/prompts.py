"""Profile-aware prompt templates for image-quality inspection with a VLM."""

from __future__ import annotations

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


PROMPT_VERSION = "2026-08-27-v2"
REVIEW_MODES = ("lenient", "standard", "strict")


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

`confidence` must be a number from 0.0 to 1.0.  Every score must be an integer
from 1 (severely broken) to 10 (clean).  `problems` must be a short string
array; use an empty array only when no issue is visible.  Do not add keys."""


CORRECTION_PROMPT = """Your previous response was not valid JSON in the required
schema.  Re-inspect the same image and respond with exactly one parseable JSON
object, with no Markdown or prose.  Use only the keys `result`, `confidence`,
`scores`, `problems`, and `summary`; use PASS, REVIEW, or FAIL for `result` and
integer scores from 1 to 10.  When uncertain, use REVIEW.

Required shape:
""" + JSON_SCHEMA


def normalize_review_mode(mode: str | None) -> str:
    value = str(mode or "standard").strip().lower()
    return value if value in REVIEW_MODES else "standard"


def build_system_prompt(mode: str = "standard") -> str:
    """Return the system instruction for the selected inspection mode."""

    selected = normalize_review_mode(mode)
    return (
        SYSTEM_PROMPT.format(profile_instructions=PROFILE_INSTRUCTIONS[selected])
        + "\n\nReturn exactly one JSON object and nothing else: no Markdown fences, comments, "
        "explanation, or leading/trailing prose. The object must have this shape:\n"
        + JSON_SCHEMA
    )


def build_user_prompt(image_name: str | None = None, *, mode: str = "standard") -> str:
    """Build the user text for one image.

    ``image_name`` is metadata only; the image itself is sent as a data URL in
    a separate content part.  Passing only a basename avoids leaking a local
    Windows/UNC path into the model prompt.
    """

    suffix = ""
    if image_name:
        # Keep this helpful for logs while avoiding arbitrary long paths.
        safe_name = str(image_name).replace("\\", "/").rsplit("/", 1)[-1][:200]
        suffix = f"\nImage filename (metadata only): {safe_name}"
    selected = normalize_review_mode(mode)
    return USER_PROMPT.format(schema=JSON_SCHEMA) + f"\nInspection mode: {selected.upper()}." + suffix


def build_messages(
    image_data_url: str,
    image_name: str | None = None,
    *,
    mode: str = "standard",
) -> list[dict[str, Any]]:
    """Create OpenAI-compatible multimodal chat messages."""

    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:"):
        raise ValueError("image_data_url must be a data: URL")
    return [
        {"role": "system", "content": build_system_prompt(mode)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(image_name, mode=mode)},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_correction_messages(
    original_messages: list[dict[str, Any]],
    invalid_response: str | None = None,
) -> list[dict[str, Any]]:
    """Append a JSON-correction turn for a retry.

    The invalid response is included only in a bounded diagnostic excerpt.  It
    helps some models repair a fenced or prefixed answer without allowing a
    huge model response to grow the request indefinitely.
    """

    excerpt = (invalid_response or "").strip()[:2000]
    text = CORRECTION_PROMPT
    if excerpt:
        text += f"\nFor context, the invalid response began with:\n{excerpt}"
    messages = list(original_messages)
    messages.append({"role": "user", "content": text})
    return messages


__all__ = [
    "CORRECTION_PROMPT",
    "JSON_SCHEMA",
    "PROFILE_INSTRUCTIONS",
    "PROMPT_VERSION",
    "REVIEW_MODES",
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "build_correction_messages",
    "build_messages",
    "build_system_prompt",
    "build_user_prompt",
    "normalize_review_mode",
]
