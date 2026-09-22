from __future__ import annotations

import struct
import zlib
from typing import Iterable

# 5x7 glyphs. Every glyph except space has ink in each column so segmentation
# does not split a character. "." is background, "#" is ink.
_GLYPHS = {
    "A": (
        ".###.",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#...#",
    ),
    "B": (
        "#####",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#####",
    ),
    "C": (
        ".###.",
        "#...#",
        "#....",
        "#....",
        "#....",
        "#...#",
        ".###.",
    ),
    "D": (
        "#####",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#####",
    ),
    "E": (
        "#####",
        "#....",
        "#....",
        "#####",
        "#....",
        "#....",
        "#####",
    ),
    "H": (
        "#...#",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#...#",
    ),
    "L": (
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#####",
    ),
    "M": (
        "#...#",
        "##.##",
        "#.#.#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
    ),
    "O": (
        ".###.",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ),
    "T": (
        "#####",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
    ),
    "U": (
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ),
    "V": (
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".#.#.",
        "..#..",
    ),
    " ": (
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
    ),
}

CHAR_W = 5
CHAR_H = 7
GAP = 1
MARGIN = 2


def ocr_engine_name() -> str:
    return "template"


def render_text(text: str) -> list[list[int]]:
    content = text.upper()
    unknown = [ch for ch in content if ch not in _GLYPHS]
    if unknown:
        raise ValueError("unsupported OCR charset: " + "".join(sorted(set(unknown))))
    width = MARGIN * 2 + len(content) * CHAR_W + max(0, len(content) - 1) * GAP
    height = MARGIN * 2 + CHAR_H
    rows = [[255 for _ in range(width)] for _ in range(height)]
    x = MARGIN
    for ch in content:
        glyph = _GLYPHS[ch]
        for dy, line in enumerate(glyph):
            for dx, pixel in enumerate(line):
                if pixel == "#":
                    rows[MARGIN + dy][x + dx] = 0
        x += CHAR_W + GAP
    return rows


def encode_gray_png(rows: list[list[int]]) -> bytes:
    height = len(rows)
    width = len(rows[0]) if rows else 0
    raw = bytearray()
    for row in rows:
        raw.append(0)
        raw.extend(row)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def decode_gray_png(blob: bytes) -> list[list[int]]:
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not_png")
    offset = 8
    width = height = None
    compressed = b""
    while offset + 8 <= len(blob):
        length = struct.unpack(">I", blob[offset : offset + 4])[0]
        tag = blob[offset + 4 : offset + 8]
        data = blob[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if tag == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", data[:10])
            if depth != 8 or color != 0:
                raise ValueError("unsupported_png")
        elif tag == b"IDAT":
            compressed += data
        elif tag == b"IEND":
            break
    if not width or not height:
        raise ValueError("png_missing_ihdr")
    raw = zlib.decompress(compressed)
    rows: list[list[int]] = []
    stride = width + 1
    for y in range(height):
        scan = raw[y * stride : (y + 1) * stride]
        if scan[0] != 0:
            raise ValueError("unsupported_png_filter")
        rows.append(list(scan[1:]))
    return rows


def _ink(value: int) -> bool:
    return value < 128


def _column_ink(rows: list[list[int]], x: int) -> bool:
    return any(_ink(row[x]) for row in rows)


def _nearest_char(patch: list[str]) -> str | None:
    best = None
    best_dist = CHAR_W * CHAR_H
    for ch, glyph in _GLYPHS.items():
        if ch == " ":
            continue
        dist = 0
        for y in range(CHAR_H):
            for x in range(CHAR_W):
                ink = patch[y][x] == "#"
                if ink != (glyph[y][x] == "#"):
                    dist += 1
        if dist < best_dist:
            best = ch
            best_dist = dist
    if best is None or best_dist > 4:
        return None
    return best


def ocr_raster(rows: list[list[int]]) -> dict[str, object]:
    if not rows or not rows[0]:
        return {"ok": False, "text": "", "engine": ocr_engine_name(), "why": "empty_image"}
    height = len(rows)
    width = len(rows[0])
    ink_cols = [x for x in range(width) if _column_ink(rows, x)]
    ink_rows = [y for y in range(height) if any(_ink(value) for value in rows[y])]
    if not ink_cols or not ink_rows:
        return {"ok": False, "text": "", "engine": ocr_engine_name(), "why": "no_ink"}
    top, bottom = ink_rows[0], ink_rows[-1]
    if bottom - top + 1 != CHAR_H:
        return {"ok": False, "text": "", "engine": ocr_engine_name(), "why": "bad_line_height"}
    band = [row[ink_cols[0] : ink_cols[-1] + 1] for row in rows[top : bottom + 1]]
    chars: list[str] = []
    x = 0
    while x < len(band[0]):
        if not _column_ink(band, x):
            white = 0
            while x < len(band[0]) and not _column_ink(band, x):
                white += 1
                x += 1
            if white >= 4:
                chars.append(" ")
            continue
        patch_cols = []
        while x < len(band[0]) and _column_ink(band, x) and len(patch_cols) < CHAR_W:
            patch_cols.append(x)
            x += 1
        if len(patch_cols) != CHAR_W:
            return {"ok": False, "text": "", "engine": ocr_engine_name(), "why": "bad_glyph_width"}
        patch = []
        for y in range(CHAR_H):
            patch.append("".join("#" if _ink(band[y][col]) else "." for col in patch_cols))
        ch = _nearest_char(patch)
        if ch is None:
            return {"ok": False, "text": "", "engine": ocr_engine_name(), "why": "unrecognized_glyph"}
        chars.append(ch)
    text = "".join(chars).strip()
    if not text:
        return {"ok": False, "text": "", "engine": ocr_engine_name(), "why": "empty_text"}
    return {"ok": True, "text": text, "engine": ocr_engine_name()}


def ocr_png(blob: bytes) -> dict[str, object]:
    try:
        rows = decode_gray_png(blob)
    except Exception as exc:
        return {"ok": False, "text": "", "engine": ocr_engine_name(), "why": str(exc)}
    return ocr_raster(rows)


def render_png(text: str) -> bytes:
    return encode_gray_png(render_text(text))


def supported_charset() -> Iterable[str]:
    return tuple(ch for ch in _GLYPHS if ch != " ")
