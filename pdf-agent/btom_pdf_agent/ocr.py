"""OCR engines.

The real engine is RapidOCR (https://github.com/RapidAI/RapidOCR):

    from rapidocr import RapidOCR
    engine = RapidOCR()
    result = engine(image)
    result.txts

``DeterministicRapidOCR`` uses that same call and only reads PNGs drawn by
``render_lines``. It does not download weights.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from btom_pdf_agent.grid_image import CELL_H, CELL_W, MARGIN, read_lines


@dataclass
class RapidOCROutput:
    boxes: tuple[list[list[float]], ...]
    txts: tuple[str, ...]
    scores: tuple[float, ...]
    elapse: float


class DeterministicRapidOCR:
    def __call__(self, img_content: Any, **kwargs: Any) -> RapidOCROutput:
        del kwargs
        image = open_image(img_content)
        lines = tuple(read_lines(image))
        boxes = tuple(_line_box(index, len(line)) for index, line in enumerate(lines))
        return RapidOCROutput(
            boxes=boxes,
            txts=lines,
            scores=tuple(1.0 for _ in lines),
            elapse=0.0,
        )


def open_image(img_content: Any) -> Image.Image:
    if isinstance(img_content, Image.Image):
        return img_content
    if isinstance(img_content, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_content))
    if isinstance(img_content, Path):
        return Image.open(img_content)
    if isinstance(img_content, str):
        return Image.open(img_content)
    shape = getattr(img_content, "shape", None)
    if shape is not None:
        return Image.fromarray(img_content)
    raise TypeError(f"unsupported image input: {type(img_content)}")


def _line_box(index: int, length: int) -> list[list[float]]:
    x0 = float(MARGIN)
    y0 = float(MARGIN + index * CELL_H)
    x1 = float(MARGIN + max(length, 1) * CELL_W)
    y1 = y0 + CELL_H
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def build_ocr_engine(kind: str = "fake") -> Any:
    if kind == "fake":
        return DeterministicRapidOCR()
    if kind == "rapidocr":
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "未安装 rapidocr。执行 pip install rapidocr 后使用 --ocr rapidocr。"
                "首次调用会下载 ONNX 模型，测试默认不走这条路径。"
            ) from exc
        return RapidOCR()
    raise ValueError(f"unknown ocr engine: {kind}")


def ocr_lines(engine: Any, image: Any) -> list[str]:
    result = engine(image)
    txts = getattr(result, "txts", None)
    if not txts:
        return []
    return [str(line) for line in txts if str(line).strip()]
