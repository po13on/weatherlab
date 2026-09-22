from prompts.loader import load_prompt, prompt_template, prompt_version
from policy import PROMPT_VERSION


def test_system_prompt_is_versioned_yaml():
    payload = load_prompt("system")
    assert payload["version"] == "wl-sys-v3"
    assert "compare_metrics" in prompt_template("system")
    assert prompt_version("system") == PROMPT_VERSION


def test_critic_prompt_card_exists():
    payload = load_prompt("critic")
    assert payload["version"] == "wl-critic-v2"
    assert payload["rules"]
