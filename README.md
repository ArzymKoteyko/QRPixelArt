# QR Pixel Art

Generate scannable QR codes whose data region shows a pixel-art image.

The QR code is always decodable — this isn't an "artistic QR" that relies on
Reed-Solomon error correction to swallow damage. The QR data is intact; art
just fills the pixels around each module's sampling center.

## How it works

QR scanners (ZXing on Android, Apple's iOS scanner, etc.) sample each module
at its geometric center. So every module is rendered as an `N × N` block of
pixels where:

- the **center `c × c` sub-block** carries the QR's required value — this is
  what the scanner reads, so the code decodes correctly;
- the **surrounding pixels** carry the pixel-art color — this is what the
  human eye integrates, so the art is what people see.

Function patterns (three finders, timing, alignment, format/version info,
quiet zone) are rendered as solid blocks so scanners lock on reliably.

With the default `--module-px 9 --core-px 3`, **89% of each data module**
shows art and only a small dot in the center carries the QR value.

## Installation

```bash
pip install qrcode pillow numpy
# optional:
pip install opencv-python   # enables --verify
pip install pytest          # to run the test suite
```

Python 3.9+ recommended (uses built-in `list[...]` generics via
`from __future__ import annotations`).

## Quick start

```bash
python3 qr_pixel_art.py \
    --url "https://example.com" \
    --art cat.png \
    --out qr.png
```

The `--art` input can be any image Pillow can open (PNG, JPEG, BMP, GIF, …).
It is auto-converted to grayscale and thresholded — you don't need to
pre-process it to black-and-white.

## CLI reference

```
--url URL              URL or text to encode.
--art PATH             Pixel-art image (any Pillow-supported format).
--out PATH             Output image path (.png, .jpg, .bmp, …).

--version N            QR version 1–40 (default 3). Grid sizes:
                         v1=21x21, v5=37x37, v10=57x57, v40=177x177.
                       Acts as a minimum unless --strict-version.
--strict-version       Fail instead of auto-bumping on overflow.
--ec {L,M,Q,H}         Error correction level (default H, ~30%).

--module-px N          Pixels per QR module (default 9). Must share
                       parity with --core-px.
--core-px N            Center sub-block carrying QR value (default 3).
                       Larger = more scanner-robust, less art visible.
--border N             Quiet-zone width in modules (default 4).

--verify               Decode the output with OpenCV as a sanity check.
                       Note: OpenCV's built-in detector is strict and
                       often fails on halftone QRs that phones read
                       just fine.
```

## Examples

Basic, with defaults:

```bash
python3 qr_pixel_art.py --url https://example.com --art cat.png --out qr.png
```

Larger QR for a long URL, maximum error correction, chunkier modules:

```bash
python3 qr_pixel_art.py \
    --url "https://example.com/very/long/path/with/query?x=1&y=2" \
    --art dragon.png --out qr.png \
    --version 10 --ec H --module-px 15 --core-px 5
```

Force an exact version (fail instead of auto-bumping):

```bash
python3 qr_pixel_art.py --url https://example.com --art cat.png --out qr.png \
    --version 5 --strict-version
```

Low error correction for more data capacity:

```bash
python3 qr_pixel_art.py --url "$(cat long.txt)" --art cat.png --out qr.png \
    --ec L --version 20
```

## Tuning guide

| If you want…                   | Do this                               |
|--------------------------------|---------------------------------------|
| More visible art               | Larger `--module-px`, smaller `--core-px` |
| More scanner-robust output     | Larger `--core-px`, or higher `--ec` |
| Finer art resolution           | Higher `--version` (more modules → more art cells) |
| Smaller output file            | Smaller `--module-px` |
| More data per QR               | Lower `--ec`, or higher `--version` |

`--module-px` and `--core-px` must share parity (both even or both odd) so
the core sub-block sits exactly at the module center. A `ValueError` up
front tells you if they don't.

## Scannability

Any output from this tool is readable by standard phone scanners
(tested with iOS Camera and Android/ZXing). If your phone struggles:

1. Make sure the printed/displayed QR has a clean quiet zone around it.
2. Increase `--core-px` (more of each module carries the QR value).
3. Increase `--module-px` so the whole image is physically larger.
4. Switch to a sparser art image — heavy detail inside the data region
   can visually confuse some detectors even though the data is intact.

`--verify` uses OpenCV's `QRCodeDetector` for an automated check, but it
is strict and regularly fails on halftone QRs that phones decode fine.
Use it as a floor, not a ceiling — passing means "definitely readable",
failing does **not** mean "broken".

## Programmatic use

```python
from qr_pixel_art import generate

generate(
    url="https://example.com",
    art_path="cat.png",
    out_path="qr.png",
    version=5,
    ec_level="H",
    module_px=9,
    core_px=3,
)
```

`generate()` raises `ValueError` on invalid parameters or `--strict-version`
overflow, and `FileNotFoundError` / `OSError` on art-loading issues.

Lower-level helpers are also exposed:

- `build_reserved_mask(size, version) -> np.ndarray` — function-pattern mask
- `load_art_mask(path, size) -> np.ndarray` — thresholded art
- `render(qr_modules, reserved, art, module_px, core_px, border_modules) -> PIL.Image.Image`

## Testing

```bash
pip install pytest
python3 -m pytest
```

The suite covers the core invariant (every module's center pixel in the
output matches the QR matrix) across multiple versions, error-correction
levels, and module/core sizes, plus the reserved-mask structure.

## Files

- `qr_pixel_art.py` — the generator (library + CLI)
- `test_qr_pixel_art.py` — pytest suite
- `make_test_art.py` — generates a small heart PNG for testing
- `heart.png` — sample input
- `qr_heart.png` — sample output
