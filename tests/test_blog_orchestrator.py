from unittest.mock import Mock

from scribe.config import BlogModeSettings
from scribe.models import BlogMode, BlogRequest
from scribe.pipeline.orchestrator import (
    build_around_hook,
    draft_from_outline,
    is_likely_truncated_output,
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


def _katana_passages() -> list[dict]:
    return [
        {
            "text": "\n".join(
                [
                    "[WEAPON] Katana",
                    "TYPE: sword",
                    "RANKS: 7th Kyu",
                    "CORE ACTIONS: drawing, cutting, distance, and control.",
                ]
            ),
            "source": "data/NTTV Weapons Reference.txt",
            "meta": {"source": "data/NTTV Weapons Reference.txt"},
        }
    ]


class RecordingRetriever:
    def __init__(self, focused: list[dict] | None = None) -> None:
        self.focused = focused or []
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, *, k: int) -> list[dict]:
        self.calls.append((query, k))
        if "katana" in query.lower():
            return self.focused
        return _passages()


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


class VerifyingLLM:
    def __init__(self) -> None:
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
        if "checking a blog draft" in prompt:
            return (
                "- This sweeping history claim needs more support than the compact brief gives it. "
                "It is intentionally long so the cap has to trim it.\n"
                "- Timeline claim is vague.",
                '{"stage":"verify"}',
            )
        return ("Draft body with one ambitious historical claim.", '{"stage":"draft"}')


class TruncatingRewriteLLM:
    def __init__(self) -> None:
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
        if len(self.calls) == 1:
            return "The rewritten draft starts strongly but", '{"attempt":1}'
        return "The rewritten draft now ends cleanly.", '{"attempt":2}'


class KatanaRegressionLLM:
    def __init__(self) -> None:
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
        assert "Stage: rewrite" in prompt
        assert "Edit instruction: add more detail on katana" in prompt
        assert "## Focused Brief" in prompt
        assert "Weapon: Katana" in prompt
        assert "## Opening" in prompt
        assert "## What 7th Kyu Is Asking For" in prompt
        assert "## Keeping The Training Honest" in prompt
        return (
            "## Training at 7th Kyu\n\n"
            "## Opening\n"
            "This article still stays with the student's experience of 7th kyu: learning how the rank "
            "asks for cleaner movement, attention, and restraint.\n\n"
            "## What 7th Kyu Is Asking For\n"
            "At 7th kyu, the draft can add katana as a specific curriculum detail rather than a new topic. "
            "The focused brief supports saying that katana is introduced as a weapon at this rank, so the "
            "addition should stay practical: posture, spacing, and careful handling inside training.\n\n"
            "## Keeping The Training Honest\n"
            "The Sanshin no Kata references should remain curriculum anchors, not invented symbolism. "
            "Keep them as named training elements and avoid replacing them with generic martial arts claims.",
            '{"stage":"rewrite","regression":"katana"}',
        )


def test_rewrite_truncation_detector_samples():
    truncated = [
        "This rewrite stops in the middle because",
        "This paragraph ends with a dangling clause,",
        "This paragraph trails off after a dash -",
        "## Katana Details",
        "The final sentence stops in the middle of the idea",
    ]
    complete = [
        "This rewrite ends cleanly.",
        "Does this rewrite end cleanly?",
        "This rewrite ends cleanly!",
        "This rewrite ends cleanly.)",
    ]

    for text in truncated:
        assert is_likely_truncated_output(text) is True
    for text in complete:
        assert is_likely_truncated_output(text) is False


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
    assert result.draft.sources_used == ["nttv rank requirements.txt"]
    assert result.draft.verify_claims == []
    assert result.metadata["verify_claims"]["enabled"] is False
    assert "Stage: draft" in llm.prompts[0]


def test_draft_from_outline_can_generate_capped_verify_claims():
    retriever = Mock(return_value=_passages())
    llm = VerifyingLLM()
    req = BlogRequest(hook_title="Why Hanbo Still Matters")
    cfg = BlogModeSettings(
        verify_claims_enabled=True,
        verify_claims_max_chars=80,
        verify_claims_max_tokens=42,
        verify_claims_temperature=0.0,
    )

    result = draft_from_outline(
        req,
        "## Outline\n- Start with hanbo",
        retriever=retriever,
        llm=llm,
        settings=cfg,
    )

    assert result.draft.sources_used == ["nttv rank requirements.txt"]
    assert result.draft.verify_claims
    assert sum(len(claim) for claim in result.draft.verify_claims) <= 80
    assert result.metadata["verify_claims"]["enabled"] is True
    assert result.metadata["verify_claims"]["max_chars"] == 80
    assert "checking a blog draft" in llm.calls[1]["prompt"]
    assert llm.calls[1]["temperature"] == 0.0
    assert llm.calls[1]["max_tokens"] == 42


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


def test_rewrite_prompt_receives_full_normal_length_draft():
    retriever = Mock(return_value=_passages())
    llm = FakeLLM()
    req = BlogRequest(hook_title="Why Hanbo Still Matters")
    paragraphs = [
        (
            f"Paragraph {idx}: Hanbo practice gives the article a steady rhythm, "
            "with enough concrete detail to resemble a normal blog draft."
        )
        for idx in range(1, 120)
    ]
    long_draft = "\n\n".join(paragraphs) + "\n\nFINAL_KATANA_DETAIL_MARKER"

    rewrite_with_instruction(
        req,
        long_draft,
        "add more detail on katana",
        retriever=retriever,
        llm=llm,
    )

    prompt = llm.prompts[0]
    assert "Edit instruction: add more detail on katana" in prompt
    assert "Paragraph 119" in prompt
    assert "FINAL_KATANA_DETAIL_MARKER" in prompt
    assert len(prompt) <= BlogModeSettings().active_context_limit


def test_rewrite_instruction_with_focus_term_adds_small_focused_brief():
    retriever = RecordingRetriever(focused=_katana_passages())
    llm = FakeLLM()
    req = BlogRequest(hook_title="Why Hanbo Still Matters")
    draft = "## Title\n\n## Training Detail\nCurrent draft body.\n\nFINAL_DRAFT_MARKER"

    result = rewrite_with_instruction(
        req,
        draft,
        "add more detail on katana",
        retriever=retriever,
        llm=llm,
    )

    prompt = llm.prompts[0]
    assert len(retriever.calls) == 2
    assert "katana curriculum details" in retriever.calls[1][0]
    assert retriever.calls[1][1] == BlogModeSettings().rewrite_focus_top_k_retrieve
    assert "FINAL_DRAFT_MARKER" in prompt
    assert "## Focused Brief" in prompt
    assert "Focused rewrite detail: katana" in prompt
    assert "NTTV Weapons Reference.txt" in prompt
    assert "NTTV Weapons Reference.txt" in result.draft.sources_used
    assert result.metadata["focused_rewrite"]["focus_term"] == "katana"
    assert result.metadata["focused_rewrite"]["char_count"] <= BlogModeSettings().rewrite_focus_budget_chars


def test_rewrite_katana_regression_preserves_7th_kyu_article_structure():
    retriever = RecordingRetriever(focused=_katana_passages())
    llm = KatanaRegressionLLM()
    req = BlogRequest(
        hook_title="Training at 7th Kyu",
        premise="A reflective article about what 7th kyu training asks from the student.",
    )
    draft = (
        "## Training at 7th Kyu\n\n"
        "## Opening\n"
        "The article is about 7th kyu as a stage of training, not a glossary entry.\n\n"
        "## What 7th Kyu Is Asking For\n"
        "The rank asks the student to connect basics without rushing past the curriculum. "
        "It mentions Sanshin no Kata as a training anchor, including Chi no Kata and Sui no Kata, "
        "without trying to turn those names into invented symbolic lessons.\n\n"
        "## Keeping The Training Honest\n"
        "The close reminds the reader to preserve source-grounded details and avoid generic filler."
    )

    result = rewrite_with_instruction(
        req,
        draft,
        "add more detail on katana",
        retriever=retriever,
        llm=llm,
    )

    body = result.draft.body
    assert body.startswith("## Training at 7th Kyu")
    assert "## Opening" in body
    assert body.index("## Opening") < body.index("## What 7th Kyu Is Asking For")
    assert body.index("## What 7th Kyu Is Asking For") < body.index("## Keeping The Training Honest")
    assert "katana is introduced as a weapon at this rank" in body
    assert "what is 7th kyu" not in body.lower()
    assert "7th kyu is the seventh student rank" not in body.lower()
    assert "earth element" not in body.lower()
    assert "water element" not in body.lower()
    assert "fire element" not in body.lower()
    assert is_likely_truncated_output(body) is False
    assert result.metadata["focused_rewrite"]["focus_term"] == "katana"
    assert result.metadata["truncation"]["detected"] is False


def test_rewrite_instruction_without_focus_term_skips_focused_retrieval():
    retriever = RecordingRetriever(focused=_katana_passages())
    llm = FakeLLM()
    req = BlogRequest(hook_title="Why Hanbo Still Matters")
    draft = "## Title\n\nCurrent draft body.\n\nFINAL_DRAFT_MARKER"

    result = rewrite_with_instruction(
        req,
        draft,
        "More direct",
        retriever=retriever,
        llm=llm,
    )

    prompt = llm.prompts[0]
    assert len(retriever.calls) == 1
    assert "FINAL_DRAFT_MARKER" in prompt
    assert "## Focused Brief" not in prompt
    assert result.metadata["focused_rewrite"] == {
        "focus_term": None,
        "retrieval_attempted": False,
    }


def test_rewrite_retries_once_when_output_looks_truncated():
    retriever = RecordingRetriever()
    llm = TruncatingRewriteLLM()
    req = BlogRequest(hook_title="Why Hanbo Still Matters")

    result = rewrite_with_instruction(
        req,
        "## Title\n\nCurrent draft body.",
        "More direct",
        retriever=retriever,
        llm=llm,
    )

    assert result.draft.body == "The rewritten draft now ends cleanly."
    assert len(llm.calls) == 2
    assert llm.calls[0]["max_tokens"] == BlogModeSettings().max_tokens_for(BlogMode.REWRITE)
    assert llm.calls[1]["max_tokens"] > llm.calls[0]["max_tokens"]
    assert result.metadata["truncation"]["detected"] is True
    assert result.metadata["truncation"]["retried"] is True
    assert result.metadata["truncation"]["final_detected"] is False


def test_polish_prompt_receives_full_normal_length_draft():
    retriever = Mock(return_value=_passages())
    llm = FakeLLM()
    req = BlogRequest(hook_title="Why Hanbo Still Matters")
    paragraphs = [
        (
            f"Paragraph {idx}: The current draft should stay visible for polish, "
            "not just the opening slice."
        )
        for idx in range(1, 120)
    ]
    long_draft = "\n\n".join(paragraphs) + "\n\nFINAL_POLISH_MARKER"

    polish_draft(
        req,
        long_draft,
        retriever=retriever,
        llm=llm,
    )

    prompt = llm.prompts[0]
    assert "Stage: polish" in prompt
    assert "Paragraph 119" in prompt
    assert "FINAL_POLISH_MARKER" in prompt
    assert len(prompt) <= BlogModeSettings().active_context_limit
