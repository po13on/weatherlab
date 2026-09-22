from fastapi.testclient import TestClient

import app as app_module


def test_analyze_without_key_is_503(monkeypatch):
    monkeypatch.setattr(app_module, "has_api_key", lambda: False)
    client = TestClient(app_module.app)
    response = client.post(
        "/analyze",
        json={"question": "why", "scope": {"run_id": "run-fail-nolog"}},
    )
    assert response.status_code == 503


def test_health_and_presets():
    client = TestClient(app_module.app)
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["fixture_loaded"] is True
    assert body["storage"] in {"sqlite", "postgres"}
    assert body["cache"] in {"memory", "redis"}
    assert body["eval_n"] == 20
    assert body["article_type"] == "business_assistant_agent"
    assert body["min_python_extras"] == "3.10"
    assert body["not_a_microservice"] is True
    assert body["ocr_engine"] == "template"
    assert body["ocr_pages"] >= 1
    assert body["rag_unit"] == "page"
    assert "cite_rule" in body
    presets = client.get("/presets")
    assert {item["id"] for item in presets.json()["presets"]} == {"q1", "q4", "q7"}


def test_bugs_and_eval_and_replay_routes():
    client = TestClient(app_module.app)
    bugs = client.get("/bugs")
    assert bugs.status_code == 200
    ids = {item["id"] for item in bugs.json()["bugs"]}
    assert ids == {f"B{i}" for i in range(1, 9)}
    assert bugs.json()["template"] == [
        "symptom",
        "wrong_guess",
        "root_cause",
        "fix",
        "next_question",
    ]
    ev = client.get("/eval/runs")
    assert ev.status_code == 200
    assert "cite_rule" in ev.json()
    missing = client.get("/jobs/job-missing/replay")
    assert missing.status_code == 404


def test_eval_without_key_is_503(monkeypatch):
    monkeypatch.setattr(app_module, "has_api_key", lambda: False)
    client = TestClient(app_module.app)
    response = client.post("/eval/runs", json={"kinds": ["system"]})
    assert response.status_code == 503
