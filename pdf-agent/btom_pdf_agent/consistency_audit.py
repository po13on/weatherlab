"""Audit whether each attention chart agrees with the conclusion on that page.

Prediction reads the PDF text layer and the embedded chart. It does not read
``true_peak``, the ``inconsistent`` field, or ``gold_inconsistent_ids.json``.
Gold ids are loaded only by :func:`score_inconsistent`.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader

from btom_pdf_agent.chart_bars import peak_key_indices

_CLAIM = re.compile(r"该头在最后一个 query 位置上，注意力最高点位于 key 位置\s*(\d+)\s*。")


def parse_claim_pos(text: str) -> int:
    found = _CLAIM.findall(text or "")
    if len(found) != 1:
        raise ValueError(f"结论句不是恰好一句，匹配到 {len(found)} 处")
    return int(found[0])


def is_inconsistent(key_indices: list[int], claim_pos: int) -> bool:
    """Inconsistent only when the chart has one index and it differs from the claim.

    Two indices mean the pixel tops are tied. That page is not called inconsistent.
    """

    unique = sorted(set(key_indices))
    return len(unique) == 1 and unique[0] != claim_pos


def audit_pdf(path: str | Path) -> dict[str, Any]:
    """Predict inconsistent pages from one PDF.

    Page ``p0000`` is the first page. Each later page adds one.
    """

    reader = PdfReader(str(path))
    pages: list[dict[str, Any]] = []
    inconsistent: list[str] = []
    tied: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages):
        page_id = f"p{index:04d}"
        text = page.extract_text() or ""
        claim_pos = parse_claim_pos(text)
        image = _page_image(page, page_id)
        key_indices = peak_key_indices(image)
        flagged = is_inconsistent(key_indices, claim_pos)
        row = {
            "page_id": page_id,
            "claim_pos": claim_pos,
            "key_indices": key_indices,
            "inconsistent": flagged,
        }
        pages.append(row)
        if flagged:
            inconsistent.append(page_id)
        if len(key_indices) != 1:
            tied.append({"page_id": page_id, "key_indices": key_indices, "claim_pos": claim_pos})
    return {
        "pages": pages,
        "inconsistent_ids": inconsistent,
        "tied": tied,
        "page_count": len(pages),
    }


def score_inconsistent(predicted_ids: list[str], gold_path: str | Path) -> dict[str, Any]:
    """Score a finished prediction. This is the only step that opens the gold file."""

    gold = _load_gold_ids(gold_path)
    predicted = set(predicted_ids)
    hit = predicted & gold
    false_positive = sorted(predicted - gold)
    missed = sorted(gold - predicted)
    return {
        "gold_count": len(gold),
        "predicted_count": len(predicted),
        "hit_count": len(hit),
        "false_positive_count": len(false_positive),
        "miss_count": len(missed),
        "recall": f"{len(hit)}/{len(gold)}",
        "false_positive_ids": false_positive,
        "missed_ids": missed,
        "hit_ids": sorted(hit),
    }


def _load_gold_ids(path: str | Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("ids"), list):
        return {str(item) for item in data["ids"]}
    if isinstance(data, list):
        return {str(item) for item in data}
    raise ValueError("金标文件不是 id 列表")


def _page_image(page: Any, page_id: str) -> Image.Image:
    images = list(getattr(page, "images", []) or [])
    if len(images) != 1:
        raise ValueError(f"{page_id} 嵌了 {len(images)} 张图，需要恰好一张")
    with Image.open(io.BytesIO(images[0].data)) as image:
        return image.convert("RGB")
