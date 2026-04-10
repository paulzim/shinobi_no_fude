from unittest.mock import Mock

from scribe.config import BlogModeSettings, DEFAULT_BLOG_MODE_SETTINGS
from scribe.models import BlogMode, BlogRequest
from scribe.pipeline.orchestrator import (
    build_around_hook,
    draft_from_outline,
    polish_draft,
    rewrite_with_instruction,
)


def _passage(idx: int, text: str = "Weapon: Hanbo") -> dict:
    return {
        "text": f"8th Kyu\n{text}\nKihon Happo: Ichimonji no Kata",
        "source": f"data/source_{idx}.txt",
        "meta": {"source": f"data/source_{idx}.txt"},
    }


class KeywordLLM:
    def __init__(self, response: str = "LLM response") -> None:
        self.response = response
        self.calls: list[dict] = []

    def __call__(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response, "{}"


def test_blog_mode_settings_defaults_and_deep_mode():
    cfg = DEFAULT_BLOG_MODE_SETTINGS

    assert cfg.context_limit == 32000
    assert cfg.active_context_limit == 32000
    assert cfg.verify_claims_enabled is False
    assert cfg.verify_claims_max_chars == 600
    assert cfg.verify_claims_max_tokens == 180
    assert cfg.verify_claims_temperature == 0.1
    assert cfg.rag_top_k_retrieve == 18
    assert cfg.rag_top_k_keep == 8
    assert cfg.rag_budget_tokens == 8000
    assert cfg.rag_budget_chars == 32000
    assert cfg.rewrite_focus_top_k_retrieve == 6
    assert cfg.rewrite_focus_top_k_keep == 3
    assert cfg.rewrite_focus_budget_chars == 1200
    assert cfg.max_tokens_for(BlogMode.OUTLINE) == 900
    assert cfg.max_tokens_for(BlogMode.DRAFT) == 2500
    assert cfg.max_tokens_for(BlogMode.POLISH) == 2000
    assert cfg.temperature_for(BlogMode.HOOK_EXPANSION) == 0.9
    assert cfg.temperature_for(BlogMode.OUTLINE) == 0.9
    assert cfg.temperature_for(BlogMode.DRAFT) == 0.8
    assert cfg.temperature_for(BlogMode.POLISH) == 0.4

    deep_cfg = BlogModeSettings(deep_mode=True)
    assert deep_cfg.active_context_limit == 64000


def test_orchestrator_uses_config_for_retrieval_and_brief_caps():
    long_text = "Very long source material that should be compacted. " * 80
    retriever = Mock(return_value=[_passage(i, long_text) for i in range(10)])
    llm = KeywordLLM("## Title Variants\n- Hanbo title\n\n## Hook Expansions\n- Hanbo hook")
    cfg = BlogModeSettings(rag_top_k_retrieve=5, rag_top_k_keep=3, rag_budget_tokens=80)
    req = BlogRequest(hook_title="What weapon do I learn at 8th kyu?")

    result = build_around_hook(req, retriever=retriever, llm=llm, settings=cfg)

    retriever.assert_called_once()
    assert retriever.call_args.kwargs["k"] == 5
    assert result.brief is not None
    assert len(result.brief.brief_markdown) <= cfg.rag_budget_chars
    assert result.brief.metadata["kept_count"] == 3
    assert result.metadata["settings"]["rag_top_k_retrieve"] == 5
    assert result.metadata["settings"]["rag_top_k_keep"] == 3


def test_orchestrator_passes_stage_temperature_and_output_tokens():
    retriever = Mock(return_value=[_passage(0)])
    llm = KeywordLLM("Draft body")
    req = BlogRequest(hook_title="Why Hanbo Still Matters")

    draft_from_outline(req, "## Outline", retriever=retriever, llm=llm)

    assert llm.calls[0]["temperature"] == 0.8
    assert llm.calls[0]["max_tokens"] == 2500

    llm.calls.clear()
    polish_draft(req, "Draft text", retriever=retriever, llm=llm)

    assert llm.calls[0]["temperature"] == 0.4
    assert 1800 <= llm.calls[0]["max_tokens"] <= 2200

    llm.calls.clear()
    rewrite_with_instruction(req, "Draft text", "More direct", retriever=retriever, llm=llm)

    assert llm.calls[0]["temperature"] == 0.4
    assert llm.calls[0]["max_tokens"] > 600
    assert llm.calls[0]["max_tokens"] == 2000
