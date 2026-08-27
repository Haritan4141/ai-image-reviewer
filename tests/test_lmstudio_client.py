from __future__ import annotations

import base64
import io
from typing import Any

import pytest

from src.lmstudio_client import (
    LOCALIZATION_OUTPUT_SCHEMA,
    LMStudioClient,
    LMStudioClientConfig,
    LMStudioHTTPError,
    LMStudioResponseError,
    image_to_data_url,
)


class _Response:
    def __init__(self, payload: Any, status_code: int = 200, text: str = "") -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class _Session:
    def __init__(self, posts: list[_Response], get_response: _Response | None = None) -> None:
        self.posts = list(posts)
        self.get_response = get_response or _Response({"data": []})
        self.post_calls: list[tuple[str, dict[str, Any]]] = []
        self.get_calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.post_calls.append((url, kwargs))
        return self.posts.pop(0)

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.get_calls.append((url, kwargs))
        return self.get_response


def _envelope(content: str) -> _Response:
    return _Response({"choices": [{"message": {"content": content}}]})


def _valid_payload() -> str:
    return (
        '{"result":"PASS","confidence":0.93,'
        '"scores":{"anatomy":9,"hands":8,"face":9,"artifacts":9,"composition":8},'
        '"problems":[],"summary":"ok"}'
    )


def _localization_payload() -> str:
    return (
        '{"person_present":true,"confidence":0.94,'
        '"regions":[{"kind":"face","box":[0.2,0.1,0.5,0.4],"confidence":0.92},'
        '{"kind":"hand","box":[0.1,0.5,0.25,0.75],"confidence":0.87}],'
        '"not_visible":["foot"],"summary":"visible face and hand"}'
    )


def test_image_to_data_url_encodes_bytes() -> None:
    raw = b"image bytes"

    value = image_to_data_url(raw, mime_type="image/jpeg")

    assert value.startswith("data:image/jpeg;base64,")
    encoded = value.split(",", 1)[1]
    assert base64.b64decode(encoded) == raw


def test_client_posts_multimodal_json_and_normalises_result() -> None:
    session = _Session([_envelope(_valid_payload())])
    config = LMStudioClientConfig(
        retries=0,
        retry_delay_seconds=0,
        max_image_dimension=0,
        review_mode="strict",
    )
    client = LMStudioClient(config, session=session)

    result = client.classify_image(b"image bytes", image_name="sample.png")

    assert result.result.value == "PASS"
    assert session.post_calls[0][0] == "http://127.0.0.1:1234/v1/chat/completions"
    body = session.post_calls[0][1]["json"]
    assert body["model"] == "qwen3-vl-8b"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][1]["content"][0]["type"] == "text"
    assert "Inspection mode: STRICT" in body["messages"][1]["content"][0]["text"]
    image_item = body["messages"][1]["content"][1]
    assert image_item["type"] == "image_url"
    assert image_item["image_url"]["url"].startswith("data:image/png;base64,")


def test_malformed_response_is_retried_with_correction_turn() -> None:
    session = _Session([_envelope("not json"), _envelope(_valid_payload())])
    config = LMStudioClientConfig(retries=1, retry_delay_seconds=0, max_image_dimension=0)
    client = LMStudioClient(config, session=session)

    result = client.classify_image(b"image bytes")

    assert result.result.value == "PASS"
    assert len(session.post_calls) == 2
    second_messages = session.post_calls[1][1]["json"]["messages"]
    assert any("valid JSON" in str(message.get("content", "")) for message in second_messages)


def test_exhausted_malformed_response_raises_response_error() -> None:
    session = _Session([_envelope("not json")])
    client = LMStudioClient(
        LMStudioClientConfig(retries=0, retry_delay_seconds=0, max_image_dimension=0),
        session=session,
    )

    with pytest.raises(LMStudioResponseError):
        client.classify_image(b"image bytes")


def test_http_error_is_reported() -> None:
    session = _Session([_Response({"error": "bad"}, status_code=500, text="server error")])
    client = LMStudioClient(
        LMStudioClientConfig(retries=0, retry_delay_seconds=0, max_image_dimension=0),
        session=session,
    )

    with pytest.raises(LMStudioHTTPError, match="HTTP 500"):
        client.request_json([])


def test_get_models_and_connection_check() -> None:
    response = _Response({"data": [{"id": "qwen3-vl-8b"}]})
    session = _Session([], get_response=response)
    client = LMStudioClient(session=session)

    assert client.get_models() == [{"id": "qwen3-vl-8b"}]
    assert client.check_connection() is True
    assert session.get_calls[0][0] == "http://127.0.0.1:1234/v1/models"


def test_target_crop_uses_target_specific_prompt_and_keeps_full_score_contract() -> None:
    session = _Session([_envelope(_valid_payload())])
    client = LMStudioClient(
        LMStudioClientConfig(retries=0, retry_delay_seconds=0, max_image_dimension=0),
        session=session,
    )

    result = client.classify_image(b"image bytes", target="hand", region_index=1)

    assert result.result.value == "PASS"
    body = session.post_calls[0][1]["json"]
    text = str(body["messages"]).lower()
    assert "hand crop" in text
    assert "only the visible hand" in text
    assert "neutral score 10" in text
    assert '"composition": 1' in text


def test_localization_uses_separate_schema_and_normalises_presence_regions() -> None:
    session = _Session([_envelope(_localization_payload())])
    client = LMStudioClient(
        LMStudioClientConfig(retries=0, retry_delay_seconds=0, max_image_dimension=0),
        session=session,
    )

    result = client.locate_regions(b"image bytes", image_name="sample.png")

    assert result["person_present"] is True
    assert result["not_visible"] == ["foot"]
    assert result["regions"][0]["kind"] == "face"
    body = session.post_calls[0][1]["json"]
    assert "person_present" in str(body["messages"])
    assert "scores" not in str(body["messages"])
    assert LOCALIZATION_OUTPUT_SCHEMA["properties"]["regions"]["maxItems"] == 32


def test_localization_malformed_response_retries_with_localization_correction() -> None:
    session = _Session([_envelope('{"person_present":true}'), _envelope(_localization_payload())])
    client = LMStudioClient(
        LMStudioClientConfig(retries=1, retry_delay_seconds=0, max_image_dimension=0),
        session=session,
    )

    result = client.locate_regions(b"image bytes")

    assert result["person_present"] is True
    assert len(session.post_calls) == 2
    second_messages = session.post_calls[1][1]["json"]["messages"]
    correction = str(second_messages[-1]["content"])
    assert "localization schema" in correction
    assert "person_present" in correction
    assert "`scores`" not in correction


def test_localization_failure_is_explicit_unknown_not_confident_absence() -> None:
    session = _Session([_envelope("not json")])
    client = LMStudioClient(
        LMStudioClientConfig(retries=0, retry_delay_seconds=0, max_image_dimension=0),
        session=session,
    )

    result = client.locate_regions(b"image bytes")

    assert result["person_present"] is None
    assert result["regions"] == []
    assert result["not_visible"] == []


def test_small_exif_rotated_image_is_transposed_without_changing_source(tmp_path: Path) -> None:
    from PIL import Image

    image = tmp_path / "rotated.jpg"
    original = Image.new("RGB", (2, 3), color=(10, 20, 30))
    exif = original.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    original.save(image, format="JPEG", exif=exif)
    source_bytes = image.read_bytes()

    value = image_to_data_url(image, max_dimension=0)
    encoded = base64.b64decode(value.split(",", 1)[1])
    with Image.open(io.BytesIO(encoded)) as prepared:
        assert prepared.size == (3, 2)
        assert prepared.getexif().get(274) is None
    assert image.read_bytes() == source_bytes
