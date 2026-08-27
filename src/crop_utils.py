"""EXIF-upright, bounded, source-preserving crop generation with owned cleanup."""

from __future__ import annotations

from contextlib import AbstractContextManager
import math
from pathlib import Path
import tempfile

from PIL import Image, ImageOps

from .models import CropBox, RegionKind
from .utils import sha256_file


class CropTooSmallError(ValueError):
    pass


class CropWorkspace(AbstractContextManager):
    """Only files created by this instance can be removed on exit.

    Unique per-run directories avoid races and preserve thumbnails from older
    reports even if a source image is inspected again with different settings.
    """

    def __init__(self, source: Path, cache_dir: Path, *, keep: bool = True) -> None:
        self.source = source
        self.cache_dir = cache_dir.resolve()
        self.keep = keep
        self.image: Image.Image | None = None
        self.root: Path | None = None
        self.created: list[Path] = []

    def __enter__(self) -> "CropWorkspace":
        with Image.open(self.source) as opened:
            upright = ImageOps.exif_transpose(opened)
            if upright.mode in {"RGBA", "LA"} or "transparency" in upright.info:
                rgba = upright.convert("RGBA")
                self.image = Image.new("RGB", rgba.size, "white")
                self.image.paste(rgba, mask=rgba.getchannel("A"))
                rgba.close()
            else:
                self.image = upright.convert("RGB")
            upright.close()
        try:
            digest_dir = self.cache_dir / sha256_file(self.source)
            digest_dir.mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix="run-", dir=digest_dir)).resolve()
        except Exception:
            self.image.close()
            self.image = None
            raise
        return self

    def generate(
        self, box: CropBox, kind: RegionKind, index: int, *, padding: float = 0.15,
        min_size: int = 96, max_dimension: int = 2048,
    ) -> tuple[Path, CropBox]:
        if self.image is None or self.root is None:
            raise RuntimeError("crop workspace is not open")
        padded = box.padded(padding)
        width, height = self.image.size
        left, top = math.floor(padded.x1 * width), math.floor(padded.y1 * height)
        right, bottom = math.ceil(padded.x2 * width), math.ceil(padded.y2 * height)
        if right - left < min_size or bottom - top < min_size:
            raise CropTooSmallError(f"{kind.value}[{index}] is smaller than {min_size}px")
        destination = self.root / f"{kind.value}-{index}.png"
        if destination.exists():
            raise ValueError("crop index already exists in this run")
        actual_box = CropBox(left / width, top / height, right / width, bottom / height)
        self.created.append(destination)
        with self.image.crop((left, top, right, bottom)) as cropped:
            cropped.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            cropped.save(destination, format="PNG")
        return destination, actual_box

    def __exit__(self, *exc: object) -> None:
        if self.image is not None:
            self.image.close()
        if self.root is not None and not self.keep:
            for path in self.created:
                # No recursive deletion, globs, or deletion of the source/cache root.
                if path.parent.resolve() == self.root and path.is_file():
                    path.unlink()
            try:
                self.root.rmdir()
                self.root.parent.rmdir()
            except OSError:
                pass
