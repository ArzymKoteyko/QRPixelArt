"""Generate a simple pixel-art heart (B/W) for testing."""
from PIL import Image

# 16x16 heart
h = [
    "................",
    "..XX.......XX...",
    ".XXXX....XXXXX..",
    "XXXXXX..XXXXXXX.",
    "XXXXXXXXXXXXXXX.",
    "XXXXXXXXXXXXXXX.",
    "XXXXXXXXXXXXXXX.",
    ".XXXXXXXXXXXXX..",
    "..XXXXXXXXXXX...",
    "...XXXXXXXXX....",
    "....XXXXXXX.....",
    ".....XXXXX......",
    "......XXX.......",
    ".......X........",
    "................",
    "................",
]

img = Image.new("L", (16, 16), 255)
for y, row in enumerate(h):
    for x, c in enumerate(row):
        if c == "X":
            img.putpixel((x, y), 0)
img.save("heart.png")
print("Wrote heart.png")
