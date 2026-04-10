from scribe.models import AnchorResult, BlogRequest, BriefResult
from scribe.writers.prompt_builder import build_writer_prompt


def _sample_request(mode: str) -> BlogRequest:
    return BlogRequest(
        hook_title="Why Hanbo Still Matters",
        premise="Build a practical blog post grounded in rank requirements.",
        include_terms=["hanbo", "distance"],
        avoid_terms=["fluff"],
        mode=mode,
    )


def _sample_anchors() -> AnchorResult:
    return AnchorResult(
        anchor_block=(
            "### Blog Anchors\n"
            "- Mode: draft\n"
            "- Hook: Why Hanbo Still Matters\n"
            "- Extractor anchor: At 8th Kyu: Weapon: Hanbo.\n"
            "- [Rank] Weapon: Hanbo"
        ),
        anchors=[
            "- Extractor anchor: At 8th Kyu: Weapon: Hanbo.",
            "- [Rank] Weapon: Hanbo",
        ],
        metadata={"anchor_count": 2},
    )


def _sample_brief() -> BriefResult:
    return BriefResult(
        title="Why Hanbo Still Matters",
        sections=[
            "- Hook: Why Hanbo Still Matters",
            "- Retrieval note: nttv rank requirements.txt: Weapon: Hanbo",
            "### Sources Used",
            "- nttv rank requirements.txt",
        ],
        brief_markdown=(
            "### Blog Brief\n"
            "- Hook: Why Hanbo Still Matters\n"
            "- Anchors: Extractor anchor: At 8th Kyu: Weapon: Hanbo.\n"
            "### Sources Used\n"
            "- nttv rank requirements.txt"
        ),
        sources_used=["nttv rank requirements.txt"],
        metadata={"kept_count": 1},
    )


def test_build_writer_prompt_includes_required_sections_for_each_stage():
    for mode in ["hook_expansion", "outline", "draft", "polish", "rewrite"]:
        prompt = build_writer_prompt(
            _sample_request(mode),
            _sample_anchors(),
            _sample_brief(),
            outline="Outline input",
            draft="Draft input",
        )

        assert f"Stage: {mode}" in prompt
        assert "## Stage Task" in prompt
        assert "## Output" in prompt
        assert "## Request" in prompt
        assert "## Anchors" in prompt
        assert "## Brief" in prompt


def test_build_writer_prompt_includes_optional_outline_and_draft_inputs():
    prompt = build_writer_prompt(
        _sample_request("polish"),
        _sample_anchors(),
        _sample_brief(),
        outline="Outline with H2 sections",
        draft="Existing draft body",
    )

    assert "## Outline Input" in prompt
    assert "Outline with H2 sections" in prompt
    assert "## Draft Input" in prompt
    assert "Existing draft body" in prompt


def test_build_writer_prompt_enforces_caps_and_keeps_structure():
    huge_anchor = "ANCHOR " * 500
    huge_brief = "BRIEF " * 700
    huge_outline = "OUTLINE " * 400
    huge_draft = "DRAFT " * 500

    prompt = build_writer_prompt(
        _sample_request("rewrite"),
        AnchorResult(anchor_block=huge_anchor),
        BriefResult(title="Why Hanbo Still Matters", brief_markdown=huge_brief),
        outline=huge_outline,
        draft=huge_draft,
        max_chars=1100,
    )

    assert len(prompt) <= 1100
    assert "## Stage Task" in prompt
    assert "## Output" in prompt
    assert "## Draft Input" in prompt
    assert huge_anchor not in prompt
    assert huge_brief not in prompt


def test_draft_mode_includes_full_prior_draft_when_explicit_redraft():
    prior_draft = "\n\n".join(
        f"Paragraph {idx}: redraft context should remain available."
        for idx in range(1, 120)
    )
    prior_draft += "\n\nFINAL_REDRAFT_MARKER"

    prompt = build_writer_prompt(
        _sample_request("draft"),
        _sample_anchors(),
        _sample_brief(),
        outline="## Outline\n- Keep the structure.",
        draft=prior_draft,
        max_chars=32000,
    )

    assert "## Draft Input" in prompt
    assert "Paragraph 119" in prompt
    assert "FINAL_REDRAFT_MARKER" in prompt


def test_hook_expansion_ignores_draft_input():
    prompt = build_writer_prompt(
        _sample_request("hook_expansion"),
        _sample_anchors(),
        _sample_brief(),
        draft="This draft should not be included in hook expansion.",
    )

    assert "## Draft Input" not in prompt
    assert "This draft should not be included" not in prompt


def test_rewrite_prompt_includes_structure_preserving_constraints():
    prompt = build_writer_prompt(
        _sample_request("rewrite"),
        _sample_anchors(),
        _sample_brief(),
        draft="## Existing Title\n\n## First Section\nKeep this order.",
    )

    assert "## Rewrite Constraints" in prompt
    assert "Preserve the existing title and section order unless the user explicitly asks to restructure." in prompt
    assert "Expand only the requested concept or section." in prompt
    assert "Do not replace grounded curriculum details with generic martial arts filler." in prompt
    assert "Do not invent unsupported meanings, benefits, or symbolism." in prompt
    assert "Prefer insertion/expansion over full regeneration." in prompt


def test_rewrite_constraints_do_not_appear_in_draft_prompt():
    prompt = build_writer_prompt(
        _sample_request("draft"),
        _sample_anchors(),
        _sample_brief(),
        outline="## Outline",
    )

    assert "## Rewrite Constraints" not in prompt
    assert "Prefer insertion/expansion over full regeneration." not in prompt


def test_rank_curriculum_draft_prompt_includes_grounding_constraints():
    prompt = build_writer_prompt(
        _sample_request("draft"),
        AnchorResult(
            anchor_block="### Blog Anchors\n- Rank scope: 6th kyu only",
            metadata={"rank_overview": True, "rank": "6th kyu"},
        ),
        BriefResult(
            title="6th Kyu overview",
            brief_markdown=(
                "### Blog Brief\n"
                "### Exact Rank Requirements\n"
                "- Rank: 6th Kyu\n"
                "- Weapon: Rokushakubo"
            ),
            sources_used=["nttv rank requirements.txt"],
            metadata={"rank_brief": True, "rank": "6th kyu"},
        ),
        outline="## Outline\n- Stay rank-specific.",
    )

    assert "## Rank/Curriculum Draft Constraints" in prompt
    assert (
        "Do not invent techniques, training equipment, benefits, or requirements not present in the fact sheet or brief."
        in prompt
    )
    assert "Do not generalize with generic martial arts filler when exact source facts are sparse." in prompt
    assert "If exact source material is limited, stay concise rather than expanding with invented content." in prompt
    assert (
        "Do not add conditioning, sparring, meditation, emotional control, or gear sections unless grounded in provided sources."
        in prompt
    )
    assert "Preserve rank specificity." in prompt


def test_non_rank_draft_prompt_omits_rank_curriculum_constraints():
    prompt = build_writer_prompt(
        BlogRequest(hook_title="A personal writing reflection", mode="draft"),
        AnchorResult(anchor_block="### Blog Anchors\n- Theme: voice"),
        BriefResult(
            title="A personal writing reflection",
            brief_markdown="### Blog Brief\n- Note: write about attention and revision.",
            sources_used=["notes.txt"],
        ),
        outline="## Outline\n- Opening\n- Close",
    )

    assert "## Rank/Curriculum Draft Constraints" not in prompt
    assert "Preserve rank specificity." not in prompt
