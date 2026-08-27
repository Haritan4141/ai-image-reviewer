"""Small, dependency-light client for LM Studio's OpenAI-compatible API."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import io
import mimetypes
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence

import requests

from .models import ClassificationResult, decode_json_object
from .prompts import (
    LOCALIZATION_CORRECTION_PROMPT,
    LOCALIZATION_OUTPUT_SCHEMA,
    build_correction_messages,
    build_localization_messages,
    build_messages_for_target,
    normalize_localization_payload,
    normalize_target,
    unknown_localization,
)


class LMStudioError(RuntimeError):
    """Base class for local LM Studio failures."""


class LMStudioHTTPError(LMStudioError):
    """The server responded with an HTTP error."""


class LMStudioResponseError(LMStudioError):
    """The server response was not usable as the required JSON object."""


def _classification_from_model_payload(value: object) -> ClassificationResult:
    """Do not accept application-owned audit metadata from the model."""

    return ClassificationResult.from_model_mapping(value)


@dataclass(frozen=True, slots=True)
class LMStudioClientConfig:
    """Settings required by :class:`LMStudioClient`.

    This mirrors ``src.config.LMStudioConfig``.  The client accepts that
    project config directly as well, so callers normally do not need to create
    this adapter explicitly.
    """

    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "qwen3-vl-8b"
    timeout_seconds: float = 120.0
    retries: int = 2
    retry_delay_seconds: float = 2.0
    max_image_dimension: int = 2048
    jpeg_quality: int = 90
    max_tokens: int = 800
    temperature: float = 0.0
    review_mode: str = "standard"


class _Session(Protocol):
    def post(self, url: str, **kwargs: Any) -> Any: ...

    def get(self, url: str, **kwargs: Any) -> Any: ...


def _setting(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def config_from_settings(settings: object | None = None, **overrides: Any) -> LMStudioClientConfig:
    """Adapt ``src.config.LMStudioConfig`` or a compatible mapping/object.

    ``AppConfig`` stores the LM Studio section under ``.lmstudio``.  A direct
    section, a mapping, or ``None`` are also accepted to keep this module useful
    in small scripts and tests.
    """

    source = settings
    rules = _setting(settings, "rules", None) if settings is not None else None
    review_mode = overrides.get(
        "review_mode",
        _setting(rules, "mode", _setting(source, "review_mode", "standard")),
    )
    if isinstance(source, (str, Path)) and "base_url" not in overrides:
        overrides = {"base_url": str(source), **overrides}
        source = None
    nested = _setting(source, "lmstudio", None) if source is not None else None
    if nested is not None:
        source = nested
    defaults = LMStudioClientConfig()
    values: dict[str, Any] = {}
    for name in (
        "base_url",
        "model",
        "timeout_seconds",
        "retries",
        "retry_delay_seconds",
        "max_image_dimension",
        "jpeg_quality",
        "max_tokens",
        "temperature",
    ):
        value = overrides.get(name, _setting(source, name, getattr(defaults, name)))
        values[name] = value
    # A few callers call this setting ``api_url`` or ``timeout``.  Accepting
    # these aliases does not change the documented config surface.
    if "base_url" not in overrides and source is not None:
        values["base_url"] = _setting(source, "base_url", _setting(source, "api_url", values["base_url"]))
    if "timeout_seconds" not in overrides and source is not None:
        values["timeout_seconds"] = _setting(source, "timeout_seconds", _setting(source, "timeout", values["timeout_seconds"]))
    return LMStudioClientConfig(
        base_url=str(values["base_url"]).rstrip("/"),
        model=str(values["model"]),
        timeout_seconds=float(values["timeout_seconds"]),
        retries=max(0, int(values["retries"])),
        retry_delay_seconds=max(0.0, float(values["retry_delay_seconds"])),
        max_image_dimension=max(0, int(values["max_image_dimension"])),
        jpeg_quality=max(1, min(100, int(values["jpeg_quality"]))),
        max_tokens=max(1, int(values["max_tokens"])),
        temperature=max(0.0, float(values["temperature"])),
        review_mode=str(review_mode).strip().lower(),
    )


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime if mime in {"image/png", "image/jpeg", "image/webp", "image/gif"} else "image/png"


def image_to_data_url(
    image: str | Path | bytes,
    *,
    mime_type: str | None = None,
    max_dimension: int | None = None,
    jpeg_quality: int = 90,
) -> str:
    """Convert an image path or bytes to a multimodal ``data:`` URL.

    Pillow is used when available to normalize EXIF orientation for every valid
    image, including images below the size limit, and to downscale oversized
    inputs. This keeps localization coordinates and later crops in the same
    upright coordinate space while preserving the original file on disk.
    Pillow is imported lazily so the client remains importable in API-only
    tests.
    """

    if isinstance(image, str) and image.startswith("data:"):
        return image

    source_path: Path | None = None
    if isinstance(image, (str, Path)):
        source_path = Path(image)
        payload = source_path.read_bytes()
        mime = mime_type or _guess_mime(source_path)
    elif isinstance(image, bytes):
        payload = image
        mime = mime_type or "image/png"
    else:
        raise TypeError("image must be a path or bytes")

    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(payload)) as opened:
            normalized = ImageOps.exif_transpose(opened).copy()
            if max_dimension and max_dimension > 0 and max(normalized.size) > max_dimension:
                scale = max_dimension / max(normalized.size)
                normalized = normalized.resize(
                    (max(1, round(normalized.width * scale)), max(1, round(normalized.height * scale))),
                    Image.Resampling.LANCZOS,
                )

            # JPEG cannot represent alpha/palette modes. PNG is retained for
            # other inputs so transparent crop images remain lossless.
            save_format = "JPEG" if mime in {"image/jpeg", "image/jpg"} else "PNG"
            if save_format == "JPEG" and normalized.mode not in {"RGB", "L"}:
                normalized = normalized.convert("RGB")
            output = io.BytesIO()
            save_kwargs = {"quality": jpeg_quality} if save_format == "JPEG" else {}
            normalized.save(output, format=save_format, **save_kwargs)
            payload = output.getvalue()
            mime = "image/jpeg" if save_format == "JPEG" else "image/png"
    except (ImportError, OSError, ValueError):
        # Encoding the original bytes is a safe fallback for an unusual or
        # partially written image. The VLM can still return REVIEW.
        pass

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


class LMStudioClient:
    """HTTP client with bounded retries and JSON-repair turns."""

    def __init__(
        self,
        config: object | None = None,
        *,
        session: _Session | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        retries: int | None = None,
        retry_delay_seconds: float | None = None,
        max_image_dimension: int | None = None,
        jpeg_quality: int | None = None,
        review_mode: str | None = None,
    ) -> None:
        overrides = {
            key: value
            for key, value in {
                "base_url": base_url,
                "model": model,
                "timeout_seconds": timeout_seconds,
                "retries": retries,
                "retry_delay_seconds": retry_delay_seconds,
                "max_image_dimension": max_image_dimension,
                "jpeg_quality": jpeg_quality,
                "review_mode": review_mode,
            }.items()
            if value is not None
        }
        self.config = config_from_settings(config, **overrides)
        self.session: _Session = session or requests.Session()

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    def _url(self, suffix: str) -> str:
        base = self.base_url
        if base.endswith(suffix):
            return base
        if base.endswith("/v1"):
            return f"{base}{suffix}"
        return f"{base}/v1{suffix}"

    @property
    def chat_url(self) -> str:
        return self._url("/chat/completions")

    @property
    def models_url(self) -> str:
        return self._url("/models")

    def _request_headers(self) -> dict[str, str]:
        # LM Studio does not require authentication by default.  Keeping this
        # header set explicit avoids sending credentials from the environment.
        return {"Content-Type": "application/json", "Accept": "application/json"}

    def _post(self, messages: list[dict[str, Any]]) -> str:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            # LM Studio supports the OpenAI JSON response hint for current
            # releases.  The parser below still validates the actual content.
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.session.post(
                self.chat_url,
                headers=self._request_headers(),
                json=body,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise LMStudioError(f"LM Studio request failed: {exc}") from exc
        status = getattr(response, "status_code", 200)
        if status >= 400:
            text = str(getattr(response, "text", ""))[:500]
            raise LMStudioHTTPError(f"LM Studio HTTP {status}: {text}")
        try:
            payload = response.json()
        except (ValueError, AttributeError) as exc:
            raise LMStudioResponseError("LM Studio did not return a JSON envelope") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LMStudioResponseError("LM Studio response had no message content") from exc
        text = _content_text(content).strip()
        if not text:
            raise LMStudioResponseError("LM Studio returned empty message content")
        return text

    def request_json(
        self,
        messages: list[dict[str, Any]],
        *,
        correction_prompt: str | None = None,
        validator: Any | None = None,
        response_name: str = "classification",
    ) -> dict[str, Any]:
        """Send a chat request and return a decoded JSON object.

        ``retries`` applies to both transport errors and malformed model
        responses.  For malformed content each retry appends a correction turn
        that explicitly asks for JSON-only output. ``correction_prompt`` and
        ``validator`` allow another response contract, such as localization,
        to use matching repair instructions and structural validation.
        """

        attempts = self.config.retries + 1
        last_error: Exception | None = None
        invalid_text: str | None = None
        current_messages = messages
        for attempt in range(attempts):
            if attempt:
                current_messages = build_correction_messages(
                    messages,
                    invalid_text,
                    correction_prompt=correction_prompt,
                )
            try:
                text = self._post(current_messages)
                invalid_text = text
                payload = decode_json_object(text)
                return validator(payload) if callable(validator) else payload
            except (LMStudioError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < attempts and self.config.retry_delay_seconds > 0:
                    time.sleep(self.config.retry_delay_seconds)
        if isinstance(last_error, LMStudioHTTPError):
            raise last_error
        raise LMStudioResponseError(
            f"LM Studio did not provide valid {response_name} JSON after "
            f"{attempts} attempt(s): {last_error}"
        ) from last_error

    def classify_image(
        self,
        image: str | Path | bytes,
        *,
        image_name: str | None = None,
        target: str = "full",
        region_index: int | None = None,
    ) -> ClassificationResult:
        """Inspect one image or target crop and return a normalised result."""

        selected_target = normalize_target(target)
        data_url = image_to_data_url(
            image,
            max_dimension=self.config.max_image_dimension,
            jpeg_quality=self.config.jpeg_quality,
        )
        payload = self.request_json(
            build_messages_for_target(
                data_url,
                image_name=image_name,
                target=selected_target,
                mode=self.config.review_mode,
                region_index=region_index,
            )
        )
        return _classification_from_model_payload(payload)

    def locate_regions(
        self,
        image: str | Path | bytes,
        *,
        image_name: str | None = None,
    ) -> dict[str, Any]:
        """Locate visible regions using a separate, validated JSON contract.

        ``person_present: false`` with ``not_visible`` is a successful absence
        claim. A transport, model, or schema failure becomes explicit unknown
        (``person_present: null`` with no ``not_visible`` claims), allowing the
        crop pipeline to apply its own REVIEW-oriented failure policy.
        """

        try:
            data_url = image_to_data_url(
                image,
                max_dimension=self.config.max_image_dimension,
                jpeg_quality=self.config.jpeg_quality,
            )
            payload = self.request_json(
                build_localization_messages(data_url, image_name=image_name),
                correction_prompt=LOCALIZATION_CORRECTION_PROMPT,
                validator=normalize_localization_payload,
                response_name="localization",
            )
            return normalize_localization_payload(payload)
        except (LMStudioError, OSError, ValueError) as exc:
            return unknown_localization(f"localization failed: {type(exc).__name__}")

    # ``analyze_image`` reads naturally in pipeline code and is kept as an
    # alias for callers that do not want to couple themselves to classifier.py.
    analyze_image = classify_image
    classify = classify_image

    def get_models(self) -> list[dict[str, Any]]:
        try:
            response = self.session.get(self.models_url, timeout=self.config.timeout_seconds)
        except requests.RequestException as exc:
            raise LMStudioError(f"LM Studio model query failed: {exc}") from exc
        if getattr(response, "status_code", 200) >= 400:
            raise LMStudioHTTPError(f"LM Studio model query returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except (ValueError, AttributeError) as exc:
            raise LMStudioResponseError("LM Studio model query was not JSON") from exc
        data = payload.get("data", []) if isinstance(payload, Mapping) else []
        return list(data) if isinstance(data, Sequence) and not isinstance(data, (str, bytes)) else []

    def check_connection(self) -> bool:
        """Return whether the LM Studio ``/v1/models`` endpoint responds."""

        try:
            self.get_models()
        except LMStudioError:
            return False
        return True

    test_connection = check_connection

    def close(self) -> None:
        close = getattr(self.session, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "LMStudioClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "LOCALIZATION_OUTPUT_SCHEMA",
    "LMStudioClient",
    "LMStudioClientConfig",
    "LMStudioError",
    "LMStudioHTTPError",
    "LMStudioResponseError",
    "config_from_settings",
    "image_to_data_url",
]
