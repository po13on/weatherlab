"""Fixed-cell figure renderer and reader for the deterministic OCR stand-in.

Glyphs live in ``glyphs.png``. Drawing and reading slice that sheet, so a test
can recover the painted string from pixels without a recognition model.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL_W = 22
CELL_H = 24
MARGIN = 8
FONT_SIZE = 18
ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "/-:_."
)
GLYPHS_PATH = Path(__file__).with_name("glyphs.png")


def build_sprite(path: Path | None = None) -> Path:
    destination = Path(path) if path else GLYPHS_PATH
    font = ImageFont.load_default(size=FONT_SIZE)
    sheet = Image.new("L", (CELL_W * len(ALPHABET), CELL_H), 255)
    draw = ImageDraw.Draw(sheet)
    for index, char in enumerate(ALPHABET):
        bbox = draw.textbbox((0, 0), char, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        x = index * CELL_W + (CELL_W - width) // 2 - bbox[0]
        y = (CELL_H - height) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination


def ensure_sprite() -> Path:
    if not GLYPHS_PATH.exists():
        build_sprite(GLYPHS_PATH)
    return GLYPHS_PATH


def _glyph_bytes() -> dict[str, bytes]:
    ensure_sprite()
    sheet = Image.open(GLYPHS_PATH).convert("L")
    expected = (CELL_W * len(ALPHABET), CELL_H)
    if sheet.size != expected:
        raise RuntimeError(f"glyphs.png size is {sheet.size}, expected {expected}")
    glyphs: dict[str, bytes] = {}
    for index, char in enumerate(ALPHABET):
        cell = sheet.crop((index * CELL_W, 0, (index + 1) * CELL_W, CELL_H))
        glyphs[char] = cell.tobytes()
    return glyphs


_GLYPHS: dict[str, bytes] | None = None


def glyphs() -> dict[str, bytes]:
    global _GLYPHS
    if _GLYPHS is None:
        _GLYPHS = _glyph_bytes()
    return _GLYPHS


def render_lines(lines: list[str], path: str | Path) -> Path:
    if not lines:
        raise ValueError("lines must not be empty")
    known = glyphs()
    for line in lines:
        unknown = sorted({char for char in line if char != " " and char not in known})
        if unknown:
            raise ValueError(f"unsupported characters: {''.join(unknown)}")
    width = MARGIN * 2 + CELL_W * max(len(line) for line in lines)
    height = MARGIN * 2 + CELL_H * len(lines)
    canvas = Image.new("L", (width, height), 255)
    sheet = Image.open(ensure_sprite()).convert("L")
    for row, line in enumerate(lines):
        for column, char in enumerate(line):
            if char == " ":
                continue
            index = ALPHABET.index(char)
            cell = sheet.crop((index * CELL_W, 0, (index + 1) * CELL_W, CELL_H))
            canvas.paste(cell, (MARGIN + column * CELL_W, MARGIN + row * CELL_H))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def read_lines(image: Image.Image) -> list[str]:
    img = image.convert("L")
    known = glyphs()
    blank = bytes([255]) * (CELL_W * CELL_H)
    by_bytes = {raw: char for char, raw in known.items()}
    lines: list[str] = []
    y = MARGIN
    while y + CELL_H <= img.height - MARGIN:
        chars: list[str] = []
        x = MARGIN
        matched = 0
        while x + CELL_W <= img.width - MARGIN:
            raw = img.crop((x, y, x + CELL_W, y + CELL_H)).tobytes()
            if raw == blank:
                chars.append(" ")
            elif raw in by_bytes:
                chars.append(by_bytes[raw])
                matched += 1
            else:
                return []
            x += CELL_W
        line = "".join(chars).rstrip()
        if not line.strip():
            break
        if matched == 0:
            break
        lines.append(line)
        y += CELL_H
    return lines
