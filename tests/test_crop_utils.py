from pathlib import Path

import pytest
from PIL import Image

from src.crop_utils import CropTooSmallError, CropWorkspace
from src.models import CropBox, RegionKind
from src.utils import sha256_file


def test_crops_do_not_modify_source_and_cleanup_is_scoped(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 400), "blue").save(source)
    before = sha256_file(source)
    cache = tmp_path / "crops"
    cache.mkdir()
    unrelated = cache / "keep.txt"
    unrelated.write_text("do not delete")
    with CropWorkspace(source, cache, keep=False) as crops:
        path, box = crops.generate(CropBox(0, 0, .5, .5), RegionKind.HAND, 0)
        assert path.is_file()
        assert box.area > .25
    assert not path.exists()
    assert unrelated.exists()
    assert sha256_file(source) == before


def test_crop_exif_orientation_and_minimum_size(tmp_path: Path) -> None:
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (200, 400), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)
    with CropWorkspace(source, tmp_path / "crops") as crops:
        assert crops.image.size == (400, 200)
        path, _ = crops.generate(CropBox(0, 0, .5, 1), RegionKind.FACE, 0, padding=0)
        with Image.open(path) as cropped:
            assert cropped.size == (200, 200)
            assert not cropped.getexif().get(274)
        with pytest.raises(CropTooSmallError):
            crops.generate(CropBox(0, 0, .1, .1), RegionKind.HAND, 0)
    assert path.exists()


def test_rescans_preserve_previous_thumbnail(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (200, 200)).save(source)
    paths = []
    for _ in range(2):
        with CropWorkspace(source, tmp_path / "crops") as crops:
            paths.append(crops.generate(CropBox(0, 0, 1, 1), RegionKind.FACE, 0)[0])
    assert paths[0] != paths[1]
    assert all(path.is_file() for path in paths)


def test_transparent_hidden_pixels_do_not_appear_as_crop_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "transparent.png"
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(source)
    with CropWorkspace(source, tmp_path / "crops", keep=False) as crops:
        path, _ = crops.generate(CropBox(0, 0, 1, 1), RegionKind.HAND, 0)
        with Image.open(path) as crop:
            assert crop.getpixel((50, 50)) == (255, 255, 255)
