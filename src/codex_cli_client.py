"""ChatGPT-authenticated Codex CLI backend for image classification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from .models import ClassificationResult, decode_json_object
from .prompts import build_system_prompt, build_user_prompt


CLASSIFICATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "result": {"type": "string", "enum": ["PASS", "REVIEW", "FAIL"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "scores": {
            "type": "object",
            "properties": {
                "anatomy": {"type": "integer", "minimum": 1, "maximum": 10},
                "hands": {"type": "integer", "minimum": 1, "maximum": 10},
                "face": {"type": "integer", "minimum": 1, "maximum": 10},
                "artifacts": {"type": "integer", "minimum": 1, "maximum": 10},
                "composition": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["anatomy", "hands", "face", "artifacts", "composition"],
            "additionalProperties": False,
        },
        "problems": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["result", "confidence", "scores", "problems", "summary"],
    "additionalProperties": False,
}


class CodexCLIError(RuntimeError):
    """Base class for Codex CLI backend failures."""


class CodexCLIAuthError(CodexCLIError):
    """Codex is not authenticated with the required ChatGPT subscription."""


class CodexCLIResponseError(CodexCLIError):
    """Codex completed but did not return a usable classification object."""


@dataclass(frozen=True, slots=True)
class CodexCLIClientConfig:
    executable: str = "codex"
    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "low"
    timeout_seconds: float = 180.0
    retries: int = 1
    retry_delay_seconds: float = 2.0
    max_image_dimension: int = 2048
    jpeg_quality: int = 90
    working_directory: Path = Path(".")
    require_chatgpt_login: bool = True
    ignore_user_config: bool = True
    ephemeral: bool = True
    review_mode: str = "standard"


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _setting(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def config_from_settings(settings: object | None = None, **overrides: Any) -> CodexCLIClientConfig:
    """Adapt ``AppConfig.codex_cli`` or a compatible mapping/object."""

    source = settings
    rules = _setting(settings, "rules", None) if settings is not None else None
    review_mode = overrides.get(
        "review_mode",
        _setting(rules, "mode", _setting(source, "review_mode", "standard")),
    )
    nested = _setting(source, "codex_cli", None) if source is not None else None
    if nested is not None:
        source = nested
    defaults = CodexCLIClientConfig()

    def value(name: str) -> Any:
        return overrides.get(name, _setting(source, name, getattr(defaults, name)))

    return CodexCLIClientConfig(
        executable=str(value("executable")).strip(),
        model=str(value("model")).strip(),
        reasoning_effort=str(value("reasoning_effort")).strip().lower(),
        timeout_seconds=float(value("timeout_seconds")),
        retries=max(0, int(value("retries"))),
        retry_delay_seconds=max(0.0, float(value("retry_delay_seconds"))),
        max_image_dimension=max(256, int(value("max_image_dimension"))),
        jpeg_quality=max(1, min(100, int(value("jpeg_quality")))),
        working_directory=Path(value("working_directory")).expanduser().resolve(),
        require_chatgpt_login=bool(value("require_chatgpt_login")),
        ignore_user_config=bool(value("ignore_user_config")),
        ephemeral=bool(value("ephemeral")),
        review_mode=str(review_mode).strip().lower(),
    )


class CodexCLIClient:
    """Run one schema-constrained ``codex exec`` turn per image.

    The default authentication guard refuses API-key sessions. This prevents a
    scan configured for ChatGPT Pro usage from silently falling back to OpenAI
    Platform API billing.
    """

    def __init__(
        self,
        config: object | None = None,
        *,
        runner: CommandRunner | None = None,
        executable_path: str | Path | None = None,
    ) -> None:
        self.config = config_from_settings(config)
        self._runner: CommandRunner = runner or subprocess.run
        self._executable_path = str(executable_path) if executable_path is not None else None
        self._authentication: str | None = None
        self._status: dict[str, str] | None = None

    def _resolve_executable(self) -> str:
        if self._executable_path:
            return self._executable_path
        resolved = shutil.which(self.config.executable)
        if not resolved:
            raise CodexCLIError(
                f"Codex CLI executable '{self.config.executable}' was not found in PATH"
            )
        self._executable_path = resolved
        return resolved

    def _run(
        self,
        args: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.config.working_directory.mkdir(parents=True, exist_ok=True)
        process_options: dict[str, Any] = {}
        if sys.platform == "win32":
            # pythonw has no parent console. Without this flag, invoking the
            # codex.cmd wrapper creates a short-lived console window for every
            # version, login-status, and image-classification subprocess.
            process_options["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            return self._runner(
                list(args),
                cwd=str(self.config.working_directory),
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout or self.config.timeout_seconds,
                check=False,
                encoding="utf-8",
                errors="replace",
                **process_options,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexCLIError(
                f"Codex CLI timed out after {timeout or self.config.timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise CodexCLIError(f"Codex CLI could not be started: {exc}") from exc

    @staticmethod
    def _combined_output(completed: subprocess.CompletedProcess[str]) -> str:
        return "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part).strip()

    def get_status(self, *, refresh: bool = False) -> dict[str, str]:
        """Return CLI version and authentication method without model usage."""

        if self._status is not None and not refresh:
            return dict(self._status)
        executable = self._resolve_executable()
        version_result = self._run((executable, "--version"), timeout=min(30.0, self.config.timeout_seconds))
        if version_result.returncode != 0:
            detail = self._combined_output(version_result)[:500]
            raise CodexCLIError(f"Codex CLI version check failed: {detail or 'unknown error'}")

        login_result = self._run(
            (executable, "login", "status"),
            timeout=min(30.0, self.config.timeout_seconds),
        )
        login_text = self._combined_output(login_result)
        if login_result.returncode != 0:
            raise CodexCLIAuthError(f"Codex login status check failed: {login_text[:500]}")

        folded = login_text.casefold()
        if "logged in using chatgpt" in folded or "signed in with chatgpt" in folded:
            authentication = "chatgpt"
        elif "api key" in folded or "api-key" in folded:
            authentication = "api_key"
        else:
            authentication = "unknown"

        if self.config.require_chatgpt_login and authentication != "chatgpt":
            raise CodexCLIAuthError(
                "Codex CLI is not signed in with ChatGPT. Classification was stopped to avoid "
                "OpenAI Platform API billing. Run 'codex login' and choose ChatGPT sign-in."
            )

        self._authentication = authentication
        self._status = {
            "version": self._combined_output(version_result),
            "authentication": authentication,
            "login_status": login_text,
            "model": self.config.model,
        }
        return dict(self._status)

    def check_connection(self) -> bool:
        try:
            self.get_status(refresh=True)
        except CodexCLIError:
            return False
        return True

    test_connection = check_connection

    def _command(self, image: Path, schema_path: Path, output_path: Path) -> list[str]:
        command = [
            self._resolve_executable(),
            "exec",
            "--model",
            self.config.model,
            "--image",
            str(image),
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
        ]
        if self.config.reasoning_effort:
            command.extend(("--config", f"model_reasoning_effort={self.config.reasoning_effort}"))
        if self.config.ignore_user_config:
            command.append("--ignore-user-config")
        if self.config.ephemeral:
            command.append("--ephemeral")
        command.append("-")
        return command

    def _prompt(self, image_name: str | None) -> str:
        return (
            build_system_prompt(self.config.review_mode)
            + "\n\n"
            + build_user_prompt(image_name, mode=self.config.review_mode)
            + "\n\nAnalyze only the attached image. Do not inspect repository files, run shell "
            "commands, use external tools, or modify any file. Return only the requested JSON object."
        )

    def _prepare_image(self, source: Path, temp_root: Path) -> Path:
        """Downscale oversized inputs without changing the original image."""

        try:
            from PIL import Image, ImageOps

            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened)
                if max(image.size) <= self.config.max_image_dimension:
                    return source
                image.thumbnail(
                    (self.config.max_image_dimension, self.config.max_image_dimension),
                    Image.Resampling.LANCZOS,
                )
                if image.mode in {"RGBA", "LA"}:
                    rgba = image.convert("RGBA")
                    flattened = Image.new("RGB", rgba.size, "white")
                    flattened.paste(rgba, mask=rgba.getchannel("A"))
                    image = flattened
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                prepared = temp_root / "prepared-image.jpg"
                image.save(prepared, format="JPEG", quality=self.config.jpeg_quality)
                return prepared
        except (ImportError, OSError, ValueError):
            return source

    def classify_image(
        self,
        image: str | Path,
        *,
        image_name: str | None = None,
    ) -> ClassificationResult:
        """Classify one image using the ChatGPT-authenticated Codex CLI."""

        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise CodexCLIError(f"image file was not found: {image_path}")

        # Re-check before every model request. A long-running watch process must
        # stop if another Codex session switches this Windows user to API-key auth.
        self.get_status(refresh=True)

        attempts = self.config.retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with tempfile.TemporaryDirectory(prefix="ai-image-reviewer-codex-") as temp_dir:
                    temp_root = Path(temp_dir)
                    schema_path = temp_root / "classification.schema.json"
                    output_path = temp_root / "classification.result.json"
                    prepared_image = self._prepare_image(image_path, temp_root)
                    schema_path.write_text(
                        json.dumps(CLASSIFICATION_OUTPUT_SCHEMA, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    completed = self._run(
                        self._command(prepared_image, schema_path, output_path),
                        input_text=self._prompt(image_name or image_path.name),
                    )
                    if completed.returncode != 0:
                        detail = self._combined_output(completed)[:1000]
                        raise CodexCLIError(
                            f"Codex CLI exited with code {completed.returncode}: {detail or 'unknown error'}"
                        )
                    text = (
                        output_path.read_text(encoding="utf-8")
                        if output_path.is_file()
                        else completed.stdout
                    )
                    payload = decode_json_object(text)
                    return ClassificationResult.from_mapping(payload)
            except (CodexCLIError, OSError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < attempts and self.config.retry_delay_seconds > 0:
                    time.sleep(self.config.retry_delay_seconds)

        raise CodexCLIResponseError(
            f"Codex CLI did not provide valid classification JSON after {attempts} attempt(s): {last_error}"
        ) from last_error

    analyze_image = classify_image
    classify = classify_image

    def close(self) -> None:
        """Match the shared analysis-client lifecycle; no persistent process exists."""

    def __enter__(self) -> "CodexCLIClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "CLASSIFICATION_OUTPUT_SCHEMA",
    "CodexCLIAuthError",
    "CodexCLIClient",
    "CodexCLIClientConfig",
    "CodexCLIError",
    "CodexCLIResponseError",
    "config_from_settings",
]
