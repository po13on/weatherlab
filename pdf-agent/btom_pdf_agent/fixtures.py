"""Build the small demo PDF from glyph images and an ASCII rule page."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from btom_pdf_agent.grid_image import render_lines

RULE_TEXT = (
    "Color ids: 0 black, 1 blue, 2 red, 3 green, 4 yellow, "
    "5 grey, 6 pink, 7 orange, 8 teal, 9 maroon. "
    "Color 10 is a canvas separator and never appears in the answer."
)

ATTENTION_LINES = [
    "L3H7 / token:V",
    "L5H2 / token:QK_I",
]

GRID_LINES = [
    "Candidate A:",
    "0 1 2",
    "3 4 5",
    "6 7 8",
    "Candidate B:",
    "1 1 1",
    "2 2 2",
    "3 3 3",
    "Candidate C:",
    "4 4 4",
    "5 5 5",
    "6 6 6",
    "Query:",
    "0 1",
    "2 3",
    "Output:",
    "0 1",
    "2 3",
]


def build_demo_pdf(directory: str | Path) -> Path:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    attention = render_lines(ATTENTION_LINES, folder / "attention.png")
    grid = render_lines(GRID_LINES, folder / "grid.png")
    pdf_path = folder / "demo.pdf"
    writer = canvas.Canvas(str(pdf_path), pagesize=A4)
    writer.setFont("Helvetica", 11)
    text = writer.beginText(48, 800)
    for part in _wrap(RULE_TEXT, 90):
        text.textLine(part)
    writer.drawText(text)
    writer.showPage()
    writer.drawImage(str(attention), 48, 620, width=500, height=80, preserveAspectRatio=True, mask="auto")
    writer.showPage()
    writer.drawImage(str(grid), 48, 360, width=420, height=420, preserveAspectRatio=True, mask="auto")
    writer.showPage()
    writer.save()
    return pdf_path


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if len(trial) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines
