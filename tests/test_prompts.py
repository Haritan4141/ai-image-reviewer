from __future__ import annotations

import pytest

from src.prompts import (
    CORRECTION_PROMPT,
    LOCALIZATION_CORRECTION_PROMPT,
    LOCALIZATION_OUTPUT_SCHEMA,
    build_correction_messages,
    build_face_crop_messages,
    build_foot_crop_messages,
    build_hand_crop_messages,
    build_localization_messages,
    build_messages,
    build_messages_for_target,
    build_system_prompt,
    build_user_prompt,
    normalize_localization_payload,
)


def test_system_prompt_covers_strict_image_quality_checks_and_json_only_output() -> None:
    prompt = build_system_prompt().lower()

    for required in ("ai-generated", "hands", "finger", "face", "limb", "json", "review", "fail"):
        assert required in prompt


def test_standard_prompt_accepts_intentional_stylization_and_requires_visible_fail_evidence() -> None:
    prompt = build_system_prompt("standard").lower()

    assert "intentional anatomical exaggeration" in prompt
    assert "multiple people overlapping" in prompt
    assert "not evidence" in prompt
    assert "precise visible evidence" in prompt


def test_review_mode_is_included_in_multimodal_messages() -> None:
    messages = build_messages("data:image/png;base64,AAAA", mode="strict")

    assert "inspection mode: strict" in str(messages).lower()


def test_build_messages_keeps_local_paths_out_of_prompt() -> None:
    messages = build_messages("data:image/png;base64,AAAA", image_name=r"D:\private\batch\image.png")
    user_text = messages[1]["content"][0]["text"]

    assert "image.png" in user_text
    assert "D:\\private" not in user_text
    assert messages[1]["content"][1]["image_url"]["url"].startswith("data:image/png")


def test_build_messages_requires_data_url() -> None:
    with pytest.raises(ValueError):
        build_messages("C:/images/image.png")


def test_correction_messages_are_bounded_and_request_json_only() -> None:
    original = build_messages("data:image/png;base64,AAAA")
    messages = build_correction_messages(original, "x" * 5000)
    correction = messages[-1]["content"]

    assert CORRECTION_PROMPT.splitlines()[0] in correction
    assert "valid JSON" in correction
    assert len(correction) < 3000


@pytest.mark.parametrize(
    ("target", "required", "unrelated"),
    [
        ("face", "FACE CROP", "hands"),
        ("hand", "HAND CROP", "face"),
        ("foot", "FOOT CROP", "hands"),
    ],
)
def test_target_crop_prompts_scope_evidence_and_neutral_scores(
    target: str, required: str, unrelated: str
) -> None:
    messages = build_messages_for_target(
        "data:image/png;base64,AAAA", target=target, region_index=2
    )
    combined = str(messages).lower()

    assert required.lower() in combined
    assert "only the visible" in combined
    assert (
        "out-of-frame" in combined
        or "out of frame" in combined
        or "outside the crop" in combined
    )
    assert "neutral score 10" in combined
    assert unrelated in combined
    assert "crop sequence index" in combined
    assert '"result"' in str(messages)


def test_named_crop_helpers_use_the_same_target_contract() -> None:
    face = build_face_crop_messages("data:image/png;base64,AAAA")
    hand = build_hand_crop_messages("data:image/png;base64,AAAA")
    foot = build_foot_crop_messages("data:image/png;base64,AAAA")

    assert "FACE CROP" in str(face)
    assert "HAND CROP" in str(hand)
    assert "FOOT CROP" in str(foot)
    for messages in (face, hand, foot):
        assert '"composition": 1' in str(messages)


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown inspection target"):
        build_messages_for_target("data:image/png;base64,AAAA", target="torso")


def test_localization_does_not_turn_malformed_presence_or_overflow_into_confident_data() -> None:
    base = {
        "person_present": False,
        "confidence": 0.9,
        "regions": [],
        "not_visible": ["face", "hand", "foot"],
        "summary": "no visible person",
    }
    with pytest.raises(ValueError, match="person_present"):
        normalize_localization_payload({**base, "person_present": "false"})

    too_many = {
        **base,
        "person_present": True,
        "regions": [
            {"kind": "face", "box": [0.01, 0.01, 0.02, 0.02], "confidence": 0.9}
            for _ in range(33)
        ],
    }
    with pytest.raises(ValueError, match="32-region"):
        normalize_localization_payload(too_many)


def test_localization_prompt_and_payload_are_separate_from_classification() -> None:
    messages = build_localization_messages("data:image/png;base64,AAAA", "D:/private/a.png")
    text = str(messages).lower()

    assert "normalized upright" in text
    assert "left/right" in text
    assert "not_visible" in text
    assert "scores" not in text
    assert LOCALIZATION_CORRECTION_PROMPT != CORRECTION_PROMPT
    assert LOCALIZATION_OUTPUT_SCHEMA["required"] == [
        "person_present",
        "confidence",
        "regions",
        "not_visible",
        "summary",
    ]

    payload = normalize_localization_payload(
        {
            "person_present": True,
            "confidence": 0.91,
            "regions": [
                {"kind": "hand", "box": [0.0, 0.2, 0.4, 0.8], "confidence": 0.88},
            ],
            "not_visible": ["foot", "foot"],
            "summary": "one visible hand",
        }
    )
    assert payload["regions"][0]["box"] == [0.0, 0.2, 0.4, 0.8]
