"""
QR Pixel Art Generator (halftone technique).

Produces a scannable QR code whose data region shows a pixel-art image.

How it works
------------
QR scanners sample each module at its geometric center. So we render every
module as an N x N block of pixels where:

  * the CENTER c x c sub-block carries the QR's required value — this is
    what the scanner reads, so the code is always decodable;
  * the remaining outer pixels carry the pixel-art color — this is what
    the human eye integrates, so the art is what people see.

Function patterns (finders, timing, alignment, format/version info) are
rendered as SOLID blocks with no art overlay, so the scanner's locator
logic works reliably.

No reliance on Reed-Solomon "error budget" — the QR data is intact, just
surrounded by art pixels. Any image works at any coverage.

Key CLI flags: --version (1-40), --ec (L/M/Q/H), --module-px, --core-px.
Run with --help for full usage.

Usage:
    python qr_pixel_art.py --url "https://example.com" \\
                           --art cat.png \\
                           --out qr.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import qrcode
from qrcode.constants import (
    ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H,
)
from PIL import Image


EC_LEVELS = {
    "L": ERROR_CORRECT_L,  # ~7%  recovery
    "M": ERROR_CORRECT_M,  # ~15% recovery
    "Q": ERROR_CORRECT_Q,  # ~25% recovery
    "H": ERROR_CORRECT_H,  # ~30% recovery
}

LUMA_THRESHOLD = 128  # grayscale value below which an art pixel is "dark"


# Alignment pattern center coordinates per QR version (1..40).
# Source: ISO/IEC 18004, Annex E.
ALIGNMENT_PATTERN_POSITIONS = [
    [], [],
    [6, 18], [6, 22], [6, 26], [6, 30], [6, 34],
    [6, 22, 38], [6, 24, 42], [6, 26, 46], [6, 28, 50],
    [6, 30, 54], [6, 32, 58], [6, 34, 62],
    [6, 26, 46, 66], [6, 26, 48, 70], [6, 26, 50, 74],
    [6, 30, 54, 78], [6, 30, 56, 82], [6, 30, 58, 86],
    [6, 34, 62, 90],
    [6, 28, 50, 72, 94], [6, 26, 50, 74, 98], [6, 30, 54, 78, 102],
    [6, 28, 54, 80, 106], [6, 32, 58, 84, 110], [6, 30, 58, 86, 114],
    [6, 34, 62, 90, 118],
    [6, 26, 50, 74, 98, 122], [6, 30, 54, 78, 102, 126],
    [6, 26, 52, 78, 104, 130], [6, 30, 56, 82, 108, 134],
    [6, 34, 60, 86, 112, 138], [6, 30, 58, 86, 114, 142],
    [6, 34, 62, 90, 118, 146],
    [6, 30, 54, 78, 102, 126, 150], [6, 24, 50, 76, 102, 128, 154],
    [6, 28, 54, 80, 106, 132, 158], [6, 32, 58, 84, 110, 136, 162],
    [6, 26, 54, 82, 110, 138, 166], [6, 30, 58, 86, 114, 142, 170],
]


def build_reserved_mask(size: int, version: int) -> np.ndarray:
    """Compute the mask of QR function-pattern modules.

    Function-pattern modules (finders + separators + format-info strips,
    timing patterns, alignment patterns, version-info blocks for v7+,
    dark module) must be rendered exactly as the QR matrix dictates, with
    no art overlay. Everything else is a data module and is free to carry
    art pixels around its center.

    Args:
        size: Grid side length in modules (= 17 + 4 * version).
        version: QR version in [1, 40].

    Returns:
        (size, size) bool ndarray, True at every function-pattern module.
    """
    mask = np.zeros((size, size), dtype=bool)

    def fill(x0: int, y0: int, w: int, h: int) -> None:
        """Set mask[y0:y0+h, x0:x0+w] = True, clipped to grid bounds."""
        x1, x2 = max(0, x0), min(size, x0 + w)
        y1, y2 = max(0, y0), min(size, y0 + h)
        mask[y1:y2, x1:x2] = True

    # Finder patterns + separators + format-info strips. The top-left
    # corner has format strips on its right and bottom sides (so 9x9);
    # the other two finders have format strips only on their inner side.
    # These three blocks also cover the dark module at (size-8, 8).
    fill(0, 0, 9, 9)
    fill(size - 8, 0, 8, 9)
    fill(0, size - 8, 9, 8)

    # Timing patterns (row 6 and column 6, full length).
    mask[6, :] = True
    mask[:, 6] = True

    # Alignment patterns (5x5 centered at tabulated positions).
    centers = ALIGNMENT_PATTERN_POSITIONS[version]
    for cy in centers:
        for cx in centers:
            in_finder = (
                (cx <= 8 and cy <= 8)
                or (cx >= size - 9 and cy <= 8)
                or (cx <= 8 and cy >= size - 9)
            )
            if in_finder:
                continue
            fill(cx - 2, cy - 2, 5, 5)

    # Version info blocks (v7+).
    if version >= 7:
        fill(size - 11, 0, 3, 6)
        fill(0, size - 11, 6, 3)

    return mask


def load_art_mask(path: str, size: int) -> np.ndarray:
    """Load an image from disk and reduce it to a dark-pixel bool mask.

    The image is converted to grayscale, nearest-neighbor resized to
    (size, size), and thresholded at LUMA_THRESHOLD. Any Pillow-
    supported format works (PNG, JPEG, BMP, GIF, …).

    Args:
        path: Path to the art image.
        size: Output side length in pixels (= QR grid module count).

    Returns:
        (size, size) bool ndarray, True where the art is dark.

    Raises:
        FileNotFoundError: If `path` does not exist.
        OSError / PIL.UnidentifiedImageError: If Pillow cannot decode it.
    """
    img = Image.open(path).convert("L").resize(
        (size, size), Image.Resampling.NEAREST
    )
    return np.asarray(img) < LUMA_THRESHOLD


def _upscale(a: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor block upscale.

    Each (i, j) cell of `a` becomes a contiguous factor x factor block in
    the output. Shape (h, w) -> (h * factor, w * factor); dtype preserved.

    Args:
        a: Any 2-D ndarray.
        factor: Positive integer scale factor.

    Returns:
        A fresh (not a view) upscaled 2-D ndarray.
    """
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)


def render(qr_modules: np.ndarray | list[list[bool]],
           reserved: np.ndarray,
           art: np.ndarray,
           module_px: int,
           core_px: int,
           border_modules: int) -> Image.Image:
    """Render a halftone QR code as a PIL image.

    Each QR module becomes a module_px x module_px pixel block. Function-
    pattern modules (reserved=True) render solid at the QR value. Data
    modules render their centered core_px x core_px sub-block at the QR
    value and the surrounding ring at the art value. The whole grid is
    then wrapped in a border_modules-thick quiet zone of light pixels.

    No validation happens here — callers should have already run
    `_validate` (or, for the CLI, let argparse bound the inputs).

    Args:
        qr_modules: (size, size) QR matrix. Accepts the list-of-list form
            returned by `qrcode.QRCode.modules` or a bool ndarray. True =
            dark.
        reserved: (size, size) bool ndarray; output of
            `build_reserved_mask`.
        art: (size, size) bool ndarray; output of `load_art_mask`. True =
            dark.
        module_px: Pixels per module (>= 1). Must share parity with
            core_px.
        core_px: Pixel size of the centered QR-value sub-block.
            1 <= core_px <= module_px.
        border_modules: Quiet-zone width in modules (>= 0).

    Returns:
        PIL Image in mode "L" (8-bit grayscale) of side length
        (size + 2 * border_modules) * module_px pixels.
    """
    qr = np.asarray(qr_modules, dtype=bool)
    size = qr.shape[0]

    # Upsample each per-module layer to pixel resolution.
    qr_up = _upscale(qr, module_px)
    rsv_up = _upscale(reserved, module_px)
    art_up = _upscale(art, module_px)

    # Per-module mask marking the centered core region, tiled across the grid.
    core_lo = (module_px - core_px) // 2
    core_hi = core_lo + core_px
    core_small = np.zeros((module_px, module_px), dtype=bool)
    core_small[core_lo:core_hi, core_lo:core_hi] = True
    core_tiled = np.tile(core_small, (size, size))

    # Use QR value inside reserved regions OR inside the core; art elsewhere.
    dark_region = np.where(rsv_up | core_tiled, qr_up, art_up)

    # Surround with quiet-zone border.
    border_px = border_modules * module_px
    dark = np.pad(dark_region, border_px, constant_values=False)

    # Mode "L" (grayscale): works with PNG / JPEG / BMP alike.
    # uint8 scalars make np.where yield uint8 directly — no astype copy.
    arr = np.where(dark, np.uint8(0), np.uint8(255))
    return Image.fromarray(arr, mode="L")


def _validate(module_px: int, core_px: int,
              version: int, ec_level: str) -> None:
    """Check geometry and QR parameters before building anything.

    Args:
        module_px: Pixels per module.
        core_px: Size of the centered QR-value sub-block.
        version: QR version (expected in [1, 40]).
        ec_level: Error-correction level; must already be upper-cased.

    Raises:
        ValueError: On any out-of-range or inconsistent parameter. The
            exception message names the offending constraint.
    """
    if core_px < 1 or core_px > module_px:
        raise ValueError("core_px must be in [1, module_px]")
    if (module_px - core_px) % 2 != 0:
        raise ValueError(
            "module_px and core_px must share parity so the core "
            "sub-block sits exactly at the module center."
        )
    if not (1 <= version <= 40):
        raise ValueError("version must be in [1, 40]")
    if ec_level not in EC_LEVELS:
        raise ValueError(f"ec_level must be one of {list(EC_LEVELS)}")


def generate(url: str, art_path: str, out_path: str,
             module_px: int = 9, core_px: int = 3,
             border_modules: int = 4, version: int = 3,
             strict_version: bool = False,
             ec_level: str = "H",
             verify: bool = False) -> None:
    """End-to-end: encode `url`, overlay art, write image to `out_path`.

    This is what the CLI wraps and is the intended public entry point
    for programmatic use. On success, prints two or three stdout lines
    summarizing the result (version, grid size, art ratio, output path).

    Args:
        url: Payload to encode. Any string; typically a URL.
        art_path: Path to a pixel-art image; any Pillow-supported format.
            Auto-converted to grayscale and thresholded.
        out_path: Destination path. File extension selects the format
            (.png / .jpg / .bmp / ...). Silently overwritten if it
            exists.
        module_px: Pixels per module. Must share parity with core_px.
        core_px: Pixel size of the centered QR-value sub-block;
            1 <= core_px <= module_px. Larger = more scanner-robust,
            less art visible.
        border_modules: Quiet-zone width in modules.
        version: QR version in [1, 40]. Controls grid size
            (= 17 + 4 * version). Acts as a minimum unless
            strict_version=True.
        strict_version: If True, raise when data overflows `version`
            instead of silently bumping to a larger version.
        ec_level: Error-correction level: "L", "M", "Q", or "H"
            (case-insensitive). Higher levels reserve more modules for
            parity and leave less data capacity per version.
        verify: If True, re-decode the saved image with OpenCV and print
            the outcome. Requires opencv-python; no-op if not installed.

    Raises:
        ValueError: On invalid parameters, or on overflow when
            strict_version=True.
        FileNotFoundError / OSError: If `art_path` can't be read or
            `out_path` can't be written.
    """
    ec_level = ec_level.upper()
    _validate(module_px, core_px, version, ec_level)

    # box_size and border below are unused by us (we render from qr.modules
    # directly), but qrcode requires positive values. Keep as 1 / 0.
    qr = qrcode.QRCode(
        version=version,
        error_correction=EC_LEVELS[ec_level],
        box_size=1,
        border=0,
    )
    qr.add_data(url)
    try:
        # fit=False forces exactly `version`; fit=True treats it as a
        # minimum and bumps up if the data doesn't fit.
        qr.make(fit=not strict_version)
    except qrcode.exceptions.DataOverflowError as e:
        raise ValueError(
            f"Data does not fit in QR version {version} at EC level "
            f"{ec_level}. Use a higher --version, a lower --ec, or drop "
            f"--strict-version."
        ) from e

    size = qr.modules_count
    final_version = qr.version
    reserved = build_reserved_mask(size, final_version)
    art = load_art_mask(art_path, size)

    img = render(qr.modules, reserved, art,
                 module_px=module_px, core_px=core_px,
                 border_modules=border_modules)
    img.save(out_path)

    art_ratio = 1.0 - (core_px * core_px) / (module_px * module_px)
    bumped = f" (bumped from {version})" if final_version != version else ""
    print(f"QR version {final_version}{bumped}, "
          f"{size}x{size} modules, level {ec_level} EC")
    print(f"Module {module_px}x{module_px} px, core {core_px}x{core_px} px "
          f"({art_ratio:.0%} of each data module shows art)")
    print(f"Wrote {out_path}")

    if verify:
        _verify_with_opencv(out_path, url)


def _verify_with_opencv(out_path: str, url: str) -> None:
    """Decode `out_path` with cv2.QRCodeDetector and print the outcome.

    Prints (and returns normally) a status line for any of these cases:
      * opencv-python isn't installed;
      * cv2.imread returns None (file unreadable by OpenCV);
      * OpenCV decodes successfully and the value matches `url`;
      * OpenCV decodes successfully but the value is wrong (very unusual);
      * OpenCV fails to decode (common for small --core-px — real phone
        scanners tend to succeed where OpenCV's detector fails).

    Never raises.

    Args:
        out_path: Path to an already-written output image.
        url: Expected decoded string.
    """
    try:
        import cv2  # optional dependency
    except ImportError:
        print("Verify skipped: install opencv-python for --verify.")
        return

    img = cv2.imread(out_path)
    if img is None:
        print(f"Verify skipped: could not read back {out_path}.")
        return

    decoded = cv2.QRCodeDetector().detectAndDecode(img)[0]
    if decoded == url:
        print("Verify (OpenCV): OK — decoded matches input URL.")
    elif decoded:
        print(f"Verify (OpenCV): decoded mismatched value: {decoded!r}")
    else:
        print(
            "Verify (OpenCV): could not decode. Note: OpenCV's detector "
            "is strict; phone scanners (ZXing / iOS / Android) are "
            "typically more tolerant of small cores. If scanning fails "
            "on your phone too, increase --core-px."
        )


class _HelpFmt(argparse.RawDescriptionHelpFormatter,
               argparse.ArgumentDefaultsHelpFormatter):
    """RawDescription (preserves docstring formatting) + ArgumentDefaults
    (auto-appends '(default: X)' to every argument help string)."""


def main() -> None:
    """CLI entry point.

    Parses argv, calls `generate`, and translates expected errors
    (ValueError, FileNotFoundError, OSError) into a one-line stderr
    message and exit code 2 — so normal user mistakes don't produce
    a Python traceback.
    """
    ap = argparse.ArgumentParser(
        description=__doc__.strip(),
        formatter_class=_HelpFmt,
    )
    ap.add_argument("--url", required=True, help="URL or text to encode")
    ap.add_argument("--art", required=True,
                    help="Path to pixel-art image. Any format Pillow reads; "
                         "auto-converted to grayscale and thresholded.")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--module-px", type=int, default=9,
                    help="Pixels per QR module. Must share parity with "
                         "--core-px. Larger = more art detail.")
    ap.add_argument("--core-px", type=int, default=3,
                    help="Size of center sub-block carrying QR value. "
                         "Larger = more scanner-robust, less art visible.")
    ap.add_argument("--border", type=int, default=4,
                    help="Quiet-zone width in modules.")
    ap.add_argument("--version", type=int, default=3,
                    help="QR version 1-40. Controls grid size: "
                         "v1=21x21, v5=37x37, v10=57x57, v40=177x177. "
                         "Treated as a minimum (auto-bumps if data doesn't "
                         "fit) unless --strict-version is set.")
    ap.add_argument("--strict-version", action="store_true",
                    help="Fail instead of auto-bumping when data overflows "
                         "the chosen --version.")
    ap.add_argument("--ec", choices=list(EC_LEVELS), type=str.upper,
                    default="H",
                    help="Error correction level: L (~7%%), M (~15%%), "
                         "Q (~25%%), H (~30%%). Higher EC uses more modules "
                         "for parity, leaving less data capacity per "
                         "version.")
    ap.add_argument("--verify", action="store_true",
                    help="Decode output with OpenCV as a sanity check.")
    args = ap.parse_args()

    try:
        generate(args.url, args.art, args.out,
                 module_px=args.module_px, core_px=args.core_px,
                 border_modules=args.border, version=args.version,
                 strict_version=args.strict_version,
                 ec_level=args.ec, verify=args.verify)
    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
