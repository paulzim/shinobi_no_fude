from unittest.mock import Mock

from scribe.models import BlogRequest
from scribe.pipeline.orchestrator import (
    build_around_hook,
    draft_from_outline,
    polish_draft,
    rewrite_with_instruction,
)


def _passages() -> list[dict]:
    return [
        {
            "text": "\n".join(
                [
                    "8th Kyu",
                    "Weapon: Hanbo",
                    "Kihon Happo: Ichimonji no Kata",
                ]
            ),
            "source": "data/nttv rank requirements.txt",
            "meta": {"source": "data/nttv rank requirements.txt"},
        }
        for _ in range(10)
    ]


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str, system: str = "") -> tuple[str, str]:
        self.prompts.append(prompt)
        if "Stage: hook_expansion" in prompt:
            return (
                "## Title Variants\n"
                "- Why Hanbo Still Matters\n"
                "- The First Weapon Lesson\n\n"
                "## Hook Expansions\n"
                "- At 8th kyu, hanbo turns distance into something practical.\n"
                "- The first weapon can reshape how a beginner understands reach.",
                '{"stage":"hook"}',
            )
        if "Stage: outline" in prompt:
            return (
                "## Outline\n- Opening hook\n- Hanbo at 8th kyu\n- Practical close",
                '{"stage":"outline"}',
            )
        if "Stage: draft" in prompt:
            return ("Full draft grounded in hanbo anchors.", '{"stage":"draft"}')
        if "Stage: polish" in prompt:
            return ("Polished draft with tighter pacing.", '{"stage":"polish"}')
        if "Stage: rewrite" in prompt:
            return ("Targeted rewrite with the requested tone.", '{"stage":"rewrite"}')
        return ("Fallback response", "{}")


def test_build_around_hook_returns_titles_hooks_outline_and_anchor_metadata():
    retriever = Mock(return_value=_passages())
    llm = FakeLLM()
    req = BlogRequest(hook_title="What weapon do I learn at 8th kyu?")

    result = build_around_hook(req, retriever=retriever, llm=llm)

    retriever.assert_called_once()
    assert retriever.call_args.kwargs["k"] == 18
    assert result.title_variants == [
        "Why Hanbo Still Matters",
        "The First Weapon Lesson",
    ]
    assert "8th kyu" in result.hook_expansions[0].lower()
    assert "Hanbo at 8th kyu" in result.outline
    assert result.metadata["anchor_metadata"]["anchor_count"] > 0
    assert len(llm.prompts) == 2


def test_draft_from_outline_returns_brief_anchors_and_draft():
    retriever = Mock(return_value=_passages())
    llm = FakeLLM()
    req = BlogRequest(hook_title="Why Hanbo Still Matters")

    result = draft_from_outline(req, "## Outline\n- Start with hanbo", retriever=retriever, llm=llm)

    retriever.assert_called_once()
    assert result.brief.brief_markdown.startswith("### Blog Brief")
    assert result.anchors.anchor_block.startswith("### Blog Anchors")
    assert result.draft.body == "Full draft grounded in hanbo anchors."
    assert "Stage: draft" in llm.prompts[0]


def test_polish_and_rewrite_are_mockable_headlessly():
    retriever = Mock(return_value=_passages())
    llm = FakeLLM()
    req = BlogRequest(hook_title="Why Hanbo Still Matters")

    polished = polish_draft(req, "A draft that needs tightening.", retriever=retriever, llm=llm)
    rewritten = rewrite_with_instruction(
        req,
        "## Intro\nA draft that needs a new angle.",
        "More direct",
        retriever=retriever,
        llm=llm,
    )

    assert polished.draft.body == "Polished draft with tighter pacing."
    assert rewritten.draft.body == "Targeted rewrite with the requested tone."
    assert "Stage: polish" in llm.prompts[0]
    assert "Stage: rewrite" in llm.prompts[1]
    assert "Make the draft more direct" in llm.prompts[1]
    assert "raw chunk" not in llm.prompts[1].lower()
    assert rewritten.metadata["command"].preset == "more_direct"
    assert retriever.call_count == 2
