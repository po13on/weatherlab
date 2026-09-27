"""Chart-conclusion audit: tick geometry, ties, and a prediction that ignores gold."""

import inspect

from PIL import Image, ImageDraw

from btom_pdf_agent import chart_bars
from btom_pdf_agent.chart_bars import peak_key_indices
from btom_pdf_agent.consistency_audit import audit_pdf, is_inconsistent, parse_claim_pos


def _chart(peaks: list[int]) -> Image.Image:
    image = Image.new("RGB", (900, 320), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    left, right, spine = 30, 870, 240
    draw.line((left, spine, right, spine), fill=(0, 0, 0))
    draw.line((left, 40, left, spine), fill=(0, 0, 0))
    draw.line((right, 40, right, spine), fill=(0, 0, 0))
    origin = 50
    step = 4.0
    for tick in range(11):
        x = int(round(origin + tick * 20 * step))
        draw.line((x, spine + 2, x, spine + 5), fill=(0, 0, 0))
    for key in peaks:
        center = origin + key * step
        top = 70 if key == peaks[0] else 70
        draw.rectangle((center - 2, top, center + 2, spine - 1), fill=(76, 120, 168))
    return image


def test_unique_peak_maps_through_ticks():
    assert peak_key_indices(_chart([40])) == [40]


def test_tied_peaks_keep_both_indices():
    assert peak_key_indices(_chart([0, 80])) == [0, 80]


def test_tie_is_not_called_inconsistent():
    assert is_inconsistent([0], 102) is True
    assert is_inconsistent([0], 0) is False
    assert is_inconsistent([0, 195], 0) is False
    assert is_inconsistent([0, 195], 102) is False


def test_claim_sentence_parses_one_index():
    text = "该头在最后一个 query 位置上，注意力最高点位于 key 位置 102。"
    assert parse_claim_pos(text) == 102


def test_prediction_source_does_not_name_gold_fields():
    source = inspect.getsource(chart_bars) + inspect.getsource(audit_pdf) + inspect.getsource(parse_claim_pos)
    assert "true_peak" not in source
    assert "gold_inconsistent" not in source
    assert "inconsistent" not in inspect.getsource(chart_bars)
