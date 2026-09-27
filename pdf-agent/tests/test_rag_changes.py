"""Parent chunks, offline query rewrite, field filters, and contradictory passages."""

import inspect
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from btom_pdf_agent import chunking, contradict, query_rewrite
from btom_pdf_agent.chunking import build_page_chunks, page_title, split_parents
from btom_pdf_agent.contradict import passages_contradict
from btom_pdf_agent.embeddings import HashEmbeddings
from btom_pdf_agent.ingest import build_ingest_chain
from btom_pdf_agent.query_rewrite import expand_pronouns, rewrite_query, split_questions
from btom_pdf_agent.retrieve import HybridIndex
from btom_pdf_agent.tools import search_text


class _NoVectors:
    def similarity_search_with_score(self, query: str, k: int = 4):
        del query, k
        return []


def _index(chunks):
    # BM25Okapi gives a non-positive idf when a term is in every document.
    # A few unrelated parents keep the queried terms rare, as they are in a real PDF.
    padded = list(chunks)
    for index in range(4):
        padded.append(
            {
                "chunk_id": f"pad-{index}",
                "text": f"unrelated filler paragraph number {index} about rivers and mountains",
                "metadata": {"page": 9, "kind": "text", "source": "pdf", "origin": "body"},
            }
        )
    return HybridIndex(padded, HashEmbeddings(), _NoVectors())


def _write_pdf(path: Path, pages: list[str]) -> None:
    writer = canvas.Canvas(str(path), pagesize=A4)
    for page in pages:
        block = writer.beginText(48, 800)
        for line in page.splitlines() or [""]:
            block.textLine(line)
        writer.drawText(block)
        writer.showPage()
    writer.save()


def test_rules_do_not_call_a_model():
    for module in (chunking, query_rewrite, contradict):
        source = inspect.getsource(module)
        lowered = source.lower()
        assert "openai" not in lowered
        assert "requests" not in lowered
        assert "http" not in lowered


def test_heading_split_and_short_merge():
    alpha = "Alpha paragraph is long enough to stand alone in the index. " * 3
    beta = "Beta paragraph is long enough to stand alone in the index. " * 3
    assert len(split_parents(alpha + "\n\n" + beta)) == 2
    absorbed = "Hi\n\nOk\n\n" + ("This paragraph is long enough to remain a parent. " * 4)
    merged = split_parents(absorbed)
    assert len(merged) == 1
    assert "Hi" in merged[0]["text"] and "Ok" in merged[0]["text"]
    headed = "# Alpha\n\nHi\n\n# Beta\n\n" + ("Beta section stays apart from the short line. " * 4)
    parts = split_parents(headed)
    assert len(parts) == 2
    assert "Hi" in parts[0]["text"]
    assert "Hi" not in parts[1]["text"]
    assert page_title(headed) == "Alpha"


def test_story_page_is_one_parent_and_several_children():
    story = "\n".join(
        [
            "Jacob and Jack entered the den.",
            "The belt is in the tank.",
            "Jacob moved the belt to the case.",
            "Jacob exited the den.",
            "Jack moved the belt to the urn.",
            "Jack exited the den.",
            "Jacob and Jack entered the waiting_room.",
        ]
    )
    parts = split_parents(story)
    assert len(parts) == 1
    assert len(parts[0]["children"]) > 1


def test_search_returns_parent_text():
    sentence_a = "The alpha signal is stored beside the northern window for the whole winter."
    sentence_b = "The beta signal is stored beside the southern door for the whole summer."
    chunks = build_page_chunks(1, "text", "body", sentence_a + " " + sentence_b, title=None)
    assert len(chunks) == 1
    assert len(chunks[0]["children"]) >= 2
    hits = _index(chunks).search("beta signal southern")
    assert hits
    assert hits[0]["chunk_id"] == chunks[0]["chunk_id"]
    assert "alpha signal" in hits[0]["text"]
    assert "beta signal" in hits[0]["text"]
    assert "-c" not in hits[0]["chunk_id"]


def test_filter_page_origin_and_title():
    fee = build_page_chunks(1, "text", "body", "# Fee schedule\n\nThe fee is 10 yuan.", title="Fee schedule")
    other = build_page_chunks(2, "ocr", "ocr", "The fee is 20 yuan on the scanned page.", title=None)
    index = _index(fee + other)
    by_page = index.search("fee", page=1)
    assert by_page and all(hit["metadata"]["page"] == 1 for hit in by_page)
    assert index.search("fee", page=1, origin="ocr") == []
    titled = index.search("fee", title="Fee schedule")
    assert titled and all(hit["metadata"].get("title") == "Fee schedule" for hit in titled)
    ocr_hits = index.search("fee", origin="ocr")
    assert ocr_hits and all(hit["metadata"]["origin"] == "ocr" for hit in ocr_hits)


def test_pronoun_split_and_synonym_are_offline():
    assert expand_pronouns("Where is it?") is None
    assert expand_pronouns("Where is the belt? Is it in the urn?") == "Where is the belt? Is the belt in the urn?"
    assert expand_pronouns("Where are the belt and the urn? Is it full?") is None
    assert expand_pronouns("它在哪？") is None
    assert expand_pronouns("皮带在哪？它在箱子里吗？") == "皮带在哪？皮带在箱子里吗？"
    assert expand_pronouns("皮带和帽子在哪？它呢？") is None
    assert split_questions("Where is the belt?") == []
    assert split_questions("Where is the belt? Who moved the belt?") == [
        "Where is the belt?",
        "Who moved the belt?",
    ]
    extras = rewrite_query("what is the date?")
    assert extras
    assert "what is the date?" not in extras
    assert any("日期" in item for item in extras)
    assert rewrite_query("Where is it?") == []


def test_original_query_still_retrieves():
    chunks = build_page_chunks(
        1,
        "text",
        "body",
        "The total revenue is listed on the cover sheet for this quarter.",
        title=None,
    )
    hits = _index(chunks).search("total")
    assert hits and hits[0]["chunk_id"] == chunks[0]["chunk_id"]
    assert any("合计" in item for item in rewrite_query("total"))


def test_contradictory_passages_are_insufficient():
    left = build_page_chunks(1, "text", "body", "The fee is 10 yuan.", title=None)
    right = build_page_chunks(2, "text", "body", "The fee is 20 yuan.", title=None)
    session = {"index": _index(left + right)}
    answer = search_text(session, "What is the fee?")
    assert answer == "证据不足"
    assert "10" not in answer and "20" not in answer
    same = build_page_chunks(2, "text", "body", "The fee is 10 yuan.", title=None)
    agreed = search_text({"index": _index(left + same)}, "What is the fee?")
    assert agreed != "证据不足"
    assert "出处" in agreed
    assert "10 yuan" in agreed
    assert passages_contradict("Where is it?", [{"text": "The fee is 10 yuan."}, {"text": "The fee is 20 yuan."}]) is False


def test_ingest_parent_metadata_and_filters(tmp_path: Path):
    pdf = tmp_path / "pages.pdf"
    _write_pdf(
        pdf,
        [
            "# Fee schedule\n\nThe alpha signal is stored beside the northern window for the whole winter. The beta signal is stored beside the southern door for the whole summer.",
            "The fee is 10 yuan.",
            "The fee is 20 yuan.",
        ],
    )
    session = build_ingest_chain().invoke(
        {
            "pdf_path": str(pdf),
            "state": {},
            "persist_dir": str(tmp_path / "index"),
            "ocr_kind": "fake",
        }
    )
    hits = session["index"].search("beta signal southern")
    assert hits
    assert "alpha signal" in hits[0]["text"]
    assert hits[0]["metadata"]["origin"] == "body"
    assert hits[0]["metadata"]["page"] == 1
    assert hits[0]["metadata"]["title"] == "Fee schedule"
    assert session["index"].search("qmlock", page=1) == []
    assert search_text(session, "What is the fee?") == "证据不足"
