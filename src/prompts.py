"""Prompt templates for strict image-quality inspection with a VLM."""

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


SYSTEM_PROMPT = """You are a strict visual quality inspector for AI-generated images.
Your task is to inspect the supplied image for generation defects, not to judge
the subject, style, or artistic taste.  Be conservative: an image should be
PASS only when it is a strong adoption candidate with no obvious defect.

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

Use FAIL for a clear, substantial failure.  Use REVIEW for any suspicious
defect, a minor but visible anatomical problem, an occluded area that cannot be
judged confidently, or uncertainty.  Use PASS only when the image is clean.

Return exactly one JSON object and nothing else: no Markdown fences, comments,
explanation, or leading/trailing prose.  The object must have this shape:
""" + JSON_SCHEMA


USER_PROMPT = """Inspect this AI-generated image using the strict quality rules.
Check the full frame and do not assume that an obscured hand, face, limb, or
object is correct.  If you cannot tell, choose REVIEW.  If there is a clear
major breakage, choose FAIL.

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


def build_system_prompt() -> str:
    """Return the immutable system instruction used for every image."""

    return SYSTEM_PROMPT


def build_user_prompt(image_name: str | None = None) -> str:
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
    return USER_PROMPT.format(schema=JSON_SCHEMA) + suffix


def build_messages(image_data_url: str, image_name: str | None = None) -> list[dict[str, Any]]:
    """Create OpenAI-compatible multimodal chat messages."""

    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:"):
        raise ValueError("image_data_url must be a data: URL")
    return [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(image_name)},
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
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "build_correction_messages",
    "build_messages",
    "build_system_prompt",
    "build_user_prompt",
]
