"""Tests for qr_pixel_art.

Runs with `pytest test_qr_pixel_art.py`. Verifies the key invariant
(every module's center pixel in the output matches the underlying QR
matrix), plus a couple of correctness checks on helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
import qrcode
from PIL import Image

from qr_pixel_art import (
    EC_LEVELS,
    build_reserved_mask,
    generate,
)


@pytest.fixture
def art_path(tmp_path):
    """A small checkerboard PNG to feed into generate()."""
    img = Image.new("L", (16, 16), 255)
    for y in range(16):
        for x in range(16):
            if (x + y) % 2 == 0:
                img.putpixel((x, y), 0)
    p = tmp_path / "art.png"
    img.save(p)
    return str(p)


@pytest.mark.parametrize("version, ec, module_px, core_px", [
    (1, "L", 9, 3),
    (3, "H", 9, 3),
    (5, "M", 11, 5),
    (7, "Q", 9, 1),
    (10, "H", 15, 7),
])
def test_center_samples_match_qr_matrix(
    tmp_path, art_path, version, ec, module_px, core_px
):
    """The pixel at each module's geometric center must equal the QR
    matrix value a scanner expects — the whole point of the halftone
    trick."""
    url = "https://example.com/abc"
    out = tmp_path / "qr.png"
    border = 4
    generate(url, art_path, str(out),
             module_px=module_px, core_px=core_px,
             border_modules=border, version=version, ec_level=ec)

    ref = qrcode.QRCode(
        version=version, error_correction=EC_LEVELS[ec],
        box_size=1, border=0,
    )
    ref.add_data(url)
    ref.make(fit=True)
    expected = ref.modules
    size = ref.modules_count

    img = Image.open(out).convert("L")
    for my in range(size):
        for mx in range(size):
            px = (mx + border) * module_px + module_px // 2
            py = (my + border) * module_px + module_px // 2
            dark = img.getpixel((px, py)) < 128
            assert dark == expected[my][mx], (
                f"center mismatch at module ({my}, {mx}) for "
                f"v{version}/{ec}/m{module_px}/c{core_px}"
            )


def test_strict_version_overflow_raises(tmp_path, art_path):
    out = tmp_path / "qr.png"
    with pytest.raises(ValueError, match="does not fit"):
        generate(
            "https://example.com/this/is/a/very/long/url/that/will/not/fit",
            art_path, str(out),
            version=1, ec_level="H", strict_version=True,
        )


def test_reserved_mask_covers_dark_module():
    """Dark module lives at (size-8, 8) and must be reserved for every
    version from 1 to 40."""
    for v in range(1, 41):
        size = 17 + 4 * v
        mask = build_reserved_mask(size, v)
        assert mask[size - 8, 8], f"dark module not reserved at v{v}"


def test_reserved_mask_version_info_block_only_for_v7_plus():
    # v6: 41x41, no version info blocks.
    mask_v6 = build_reserved_mask(41, 6)
    # Cell (0, 30) = (row 0, col size-11) — outside finders and timing.
    assert not mask_v6[0, 41 - 11], "v6 must not reserve version info"

    # v7: 45x45, version info present.
    mask_v7 = build_reserved_mask(45, 7)
    assert mask_v7[0, 45 - 11], "v7 top-right version info missing"
    assert mask_v7[45 - 11, 0], "v7 bottom-left version info missing"


def test_reserved_mask_timing_patterns():
    """Row 6 and column 6 must be fully reserved."""
    for v in (1, 10, 25, 40):
        size = 17 + 4 * v
        mask = build_reserved_mask(size, v)
        assert mask[6, :].all(), f"timing row missing at v{v}"
        assert mask[:, 6].all(), f"timing column missing at v{v}"


def test_reserved_mask_finder_patterns_at_corners():
    """All three finder corners must be solidly reserved."""
    size = 21  # v1
    mask = build_reserved_mask(size, 1)
    assert mask[0:9, 0:9].all(), "top-left finder area incomplete"
    assert mask[0:9, size - 8:size].all(), "top-right finder area incomplete"
    assert mask[size - 8:size, 0:9].all(), "bottom-left finder area incomplete"


def test_generate_produces_grayscale_output(tmp_path, art_path):
    out = tmp_path / "qr.jpg"
    generate("x", art_path, str(out),
             module_px=9, core_px=3, version=3, ec_level="H")
    # JPEG only supports modes 1, L, RGB, CMYK; 'L' lets us save JPEG
    # without PIL needing to convert.
    with Image.open(out) as img:
        assert img.mode == "L"
