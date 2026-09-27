import json
from pathlib import Path

from langchain_core.tools import tool

from btom_pdf_agent.agent import ask, build_agent, stream_answer
from btom_pdf_agent.embeddings import HashEmbeddings
from btom_pdf_agent.eval_run import run_eval
from btom_pdf_agent.fixtures import GRID_LINES, build_demo_pdf
from btom_pdf_agent.grids import parse_grid_lines
from btom_pdf_agent.ingest import build_ingest_chain
from btom_pdf_agent.ledger import judge_claim
from btom_pdf_agent.ocr import DeterministicRapidOCR, ocr_lines
from btom_pdf_agent.pdf_io import split_pdf
from btom_pdf_agent.prompts import load_prompt, prompt_version
from btom_pdf_agent.tools import grid_text, ledger_text, search_text


def _session(tmp_path: Path):
    pdf = build_demo_pdf(tmp_path)
    return build_ingest_chain().invoke(
        {
            "pdf_path": str(pdf),
            "state_path": str(Path(__file__).resolve().parents[1] / "examples" / "run_state.json"),
            "persist_dir": str(tmp_path / "index"),
            "ocr_kind": "fake",
        }
    )


def test_prompt_version_and_few_shots():
    text = load_prompt()
    assert prompt_version() == "v1"
    assert "check_ledger" in text
    assert "grid_to_json" in text
    assert "证据不足" in text


def test_ingest_chain_ocr_and_pages(tmp_path: Path):
    chain = build_ingest_chain()
    assert len(chain.steps) == 4
    session = _session(tmp_path)
    pages = split_pdf(session["pdf_path"])
    assert pages[0]["text"].startswith("Color ids:")
    assert ocr_lines(DeterministicRapidOCR(), pages[1]["images"][0])[0].startswith("L3H7")
    assert any(chunk["metadata"]["page"] == 2 for chunk in session["chunks"])


def test_ledger_uses_retrieved_passage_only(tmp_path: Path):
    session = _session(tmp_path)
    answer = ledger_text(session, 2)
    assert "Answer value reader" in answer
    assert "页 2" in answer
    assert "账本块" in answer
    missed = judge_claim(
        {"layer": 3, "head": 7, "token": "V", "figure_role": None},
        [],
    )
    assert missed["consistent"] is False
    assert missed["matched_role"] is None


def test_grid_json_cites_color_rule(tmp_path: Path):
    session = _session(tmp_path)
    answer = grid_text(session, 3)
    payload = json.loads(answer.split("页 3 网格 JSON：", 1)[1].split("\n\n", 1)[0])
    assert payload["candidates"]["A"][0] == [0, 1, 2]
    assert payload["output"] == [[0, 1], [2, 3]]
    assert "Color ids" in answer
    assert "页 1" in answer
    parsed = parse_grid_lines(["Output:", "1 10 2"])
    assert parsed["grids"]["output"] == [[1, 2]]
    assert parsed["dropped_separator"] == 1
    assert "10" not in GRID_LINES[-1]


def test_unrelated_query_is_insufficient(tmp_path: Path):
    session = _session(tmp_path)
    assert search_text(session, "火星基地的坐标") == "证据不足"


def test_agent_answers_cite_sources(tmp_path: Path):
    session = _session(tmp_path)
    ledger = ask(session, "请核对第 2 页注意力图和账本", thread_id="ledger", force_fake=True)
    grid = ask(session, "请把第 3 页的网格写成 JSON", thread_id="grid", force_fake=True)
    empty = ask(session, "文档里有没有火星基地的坐标", thread_id="empty", force_fake=True)
    assert "Answer value reader" in ledger and "页 2" in ledger
    assert '"candidates"' in grid and "Color ids" in grid and "页 3" in grid
    assert empty.strip() == "证据不足"


def test_stream_and_memory(tmp_path: Path):
    session = _session(tmp_path)
    text, tokens = stream_answer(session, "请核对第 2 页注意力图和账本")
    assert "Answer value reader" in text
    assert len(tokens) > 1
    assert "".join(tokens) == text
    agent = build_agent(session, force_fake=True)
    config = {"configurable": {"thread_id": "mem"}}
    from langchain_core.messages import HumanMessage

    agent.invoke({"messages": [HumanMessage(content="请核对第 2 页注意力图和账本")]}, config)
    second = agent.invoke({"messages": [HumanMessage(content="文档里有没有火星基地的坐标")]}, config)
    contents = [str(getattr(message, "content", "")) for message in second["messages"]]
    assert any("核对" in item for item in contents)
    assert any("火星" in item for item in contents)


def test_embedding_cache(tmp_path: Path):
    cache = tmp_path / "embed.json"
    first = HashEmbeddings(cache_path=cache)
    first.embed_documents(["layer: 3 head: 7"])
    assert first.calls == 1
    second = HashEmbeddings(cache_path=cache)
    second.embed_documents(["layer: 3 head: 7"])
    assert second.calls == 0


def test_tool_retries_once():
    calls = {"n": 0}

    @tool
    def flaky_lookup(query: str) -> str:
        """Look up a value. Fails once."""

        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary")
        return f"出处：页 1 块 p1-text-0\n{query}"

    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolRetryMiddleware
    from btom_pdf_agent.agent import ScriptedToolModel

    agent = create_agent(
        ScriptedToolModel(),
        tools=[flaky_lookup],
        system_prompt="retry",
        middleware=[ToolRetryMiddleware(max_retries=1, initial_delay=0.0, backoff_factor=0.0, jitter=False)],
    )
    # The scripted model only calls search_chunks / check_ledger / grid_to_json / ocr_page.
    # Drive the tool through the middleware by invoking it as the model would, via a custom question
    # is not possible. Call the tool node path by asking in a model that selects flaky_lookup.
    from langchain_core.messages import AIMessage, HumanMessage

    class Once(ScriptedToolModel):
        def _message(self, messages):
            from langchain_core.messages import ToolMessage

            tools = [item for item in messages if isinstance(item, ToolMessage)]
            if tools:
                return AIMessage(content=str(tools[-1].content))
            return AIMessage(
                content="",
                tool_calls=[{"name": "flaky_lookup", "args": {"query": "rule"}, "id": "c1", "type": "tool_call"}],
            )

    agent = create_agent(
        Once(),
        tools=[flaky_lookup],
        system_prompt="retry",
        middleware=[ToolRetryMiddleware(max_retries=1, initial_delay=0.0, backoff_factor=0.0, jitter=False)],
    )
    result = agent.invoke({"messages": [HumanMessage(content="重试")]})
    assert calls["n"] == 2
    assert "页 1" in str(result["messages"][-1].content)


def test_eval_reports_hit_count(tmp_path: Path):
    state = Path(__file__).resolve().parents[1] / "examples" / "run_state.json"
    result = run_eval(tmp_path / "eval", state)
    assert result["summary"] == f"夹具引用命中 {result['cited']}/{result['checked']}"
    assert result["checked"] == 3
    assert result["cited"] == 3
