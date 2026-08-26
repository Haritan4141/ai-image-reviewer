from __future__ import annotations

import base64
from typing import Any

import pytest

from src.lmstudio_client import (
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


def test_image_to_data_url_encodes_bytes() -> None:
    raw = b"image bytes"

    value = image_to_data_url(raw, mime_type="image/jpeg")

    assert value.startswith("data:image/jpeg;base64,")
    encoded = value.split(",", 1)[1]
    assert base64.b64decode(encoded) == raw


def test_client_posts_multimodal_json_and_normalises_result() -> None:
    session = _Session([_envelope(_valid_payload())])
    config = LMStudioClientConfig(retries=0, retry_delay_seconds=0, max_image_dimension=0)
    client = LMStudioClient(config, session=session)

    result = client.classify_image(b"image bytes", image_name="sample.png")

    assert result.result.value == "PASS"
    assert session.post_calls[0][0] == "http://127.0.0.1:1234/v1/chat/completions"
    body = session.post_calls[0][1]["json"]
    assert body["model"] == "qwen3-vl-8b"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][1]["content"][0]["type"] == "text"
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
