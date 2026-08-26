from __future__ import annotations

import pytest

from src.prompts import (
    CORRECTION_PROMPT,
    build_correction_messages,
    build_messages,
    build_system_prompt,
    build_user_prompt,
)


def test_system_prompt_covers_strict_image_quality_checks_and_json_only_output() -> None:
    prompt = build_system_prompt().lower()

    for required in ("ai-generated", "hands", "finger", "face", "limb", "json", "review", "fail"):
        assert required in prompt


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
