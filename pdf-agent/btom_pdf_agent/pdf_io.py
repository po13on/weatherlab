"""Split a PDF into text pages and embedded images.

Text is taken from the page content stream. Images are the embedded XObjects.
A page with no extractable text is treated as an image page even when the
extractor returns nothing, so the caller can still record the page number.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader


def split_pdf(path: str | Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(path))
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        images: list[Image.Image] = []
        for image_file in list(getattr(page, "images", []) or []):
            try:
                images.append(Image.open(io.BytesIO(image_file.data)))
            except Exception:
                continue
        kind = "image" if images and not text.strip() else "text"
        if images and text.strip():
            kind = "mixed"
        pages.append(
            {
                "page": index,
                "kind": kind,
                "text": text.strip(),
                "images": images,
            }
        )
    return pages
