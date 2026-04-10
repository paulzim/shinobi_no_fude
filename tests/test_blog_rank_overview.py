import pathlib

from scribe.models import AnchorResult, BlogRequest, BriefResult
import scribe.pipeline.orchestrator as orchestrator
from scribe.pipeline.orchestrator import build_around_hook, draft_from_outline
from scribe.pipeline.rank_overview import detect_rank_overview_request


RANK_FILE = pathlib.Path("data") / "nttv rank requirements.txt"


def _rank_passages() -> list[dict]:
    return [
        {
            "text": RANK_FILE.read_text(encoding="utf-8"),
            "source": "data/nttv rank requirements.txt",
            "meta": {"source": "data/nttv rank requirements.txt"},
        }
    ]


class CaptureLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        self.prompts.append(prompt)
        if "Stage: hook_expansion" in prompt:
            return "## Title Variants\n- Rank overview\n\n## Hook Expansions\n- Rank overview hook.", "{}"
        if "Stage: outline" in prompt:
            return "## Outline\n- Rank facts\n- Training shape", "{}"
        return "Draft body.", "{}"


class CaptureRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, *, k: int) -> list[dict]:
        self.calls.append((query, k))
        return _rank_passages()


class SixthKyuRegressionLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        self.prompts.append(prompt)
        low = prompt.lower()

        assert "stage: draft" in low
        assert "what are the skills for 6th kyu?" in low
        assert "give an overview of 6th kyu for a beginning student." in low
        assert "### exact rank requirements" in low
        assert "## rank/curriculum draft constraints" in low
        assert "preserve rank specificity" in low
        assert "6th kyu" in low
        assert "rokushakubo" in low

        for contaminant in [
            "7th kyu",
            "8th kyu",
            "9th kyu",
            "sassho-ouchi",
            "kote-ouchi",
            "black gi",
            "hanbo",
            "bokken",
            "tanto",
        ]:
            assert contaminant not in low

        return (
            "## 6th Kyu Overview\n\n"
            "For a beginning student, 6th kyu stays centered on the Rokushakubo material listed "
            "in the rank requirements: weapon kamae, basic staff strikes, spinning work, "
            "taihenjutsu, striking, grappling and escapes, and nage waza. The scope should stay "
            "with those listed requirements rather than expanding into other ranks.",
            "{}",
        )


def test_detect_rank_overview_request_for_6th_kyu_skills():
    assert detect_rank_overview_request("What are the skills for 6th kyu?") == "6th kyu"


def test_detect_rank_overview_request_for_7th_kyu_overview():
    assert detect_rank_overview_request("Give an overview of 7th kyu") == "7th kyu"


def test_rank_overview_context_uses_only_6th_kyu_fact_sheet():
    retriever = CaptureRetriever()
    llm = CaptureLLM()
    req = BlogRequest(hook_title="What are the skills for 6th kyu?")

    result = build_around_hook(req, retriever=retriever, llm=llm)

    assert "6th kyu" in retriever.calls[0][0].lower()
    assert result.anchors.metadata["rank_overview"] is True
    assert result.anchors.metadata["rank"] == "6th kyu"
    assert "### Rank Overview Fact Sheet" in result.anchors.anchor_block
    assert "- Rank: 6th Kyu" in result.anchors.anchor_block
    assert "Rokushakubo" in result.anchors.anchor_block
    assert "Katana" not in result.anchors.anchor_block
    assert "Hanbo" not in result.anchors.anchor_block
    assert "7th Kyu" not in result.anchors.anchor_block
    assert "5th Kyu" not in result.anchors.anchor_block
    assert "### Rank Overview Fact Sheet" in llm.prompts[0]


def test_rank_overview_draft_prompt_uses_only_7th_kyu_fact_sheet():
    retriever = CaptureRetriever()
    llm = CaptureLLM()
    req = BlogRequest(hook_title="Give an overview of 7th kyu")

    result = draft_from_outline(
        req,
        "## Outline\n- Stay focused on the requested rank.",
        retriever=retriever,
        llm=llm,
    )

    prompt = llm.prompts[0]
    assert result.anchors.metadata["rank_overview"] is True
    assert result.anchors.metadata["rank"] == "7th kyu"
    assert "### Rank Overview Fact Sheet" in prompt
    assert "- Rank: 7th Kyu" in prompt
    assert "Katana" in prompt
    assert "Rokushakubo" not in prompt
    assert "Knife; Shoto" not in prompt
    assert "Hanbo" not in prompt


def test_exact_6th_kyu_beginner_overview_regression():
    retriever = CaptureRetriever()
    llm = SixthKyuRegressionLLM()
    req = BlogRequest(
        hook_title="What are the skills for 6th kyu?",
        premise="Give an overview of 6th kyu for a beginning student.",
    )

    result = draft_from_outline(
        req,
        "## Outline\n- Explain only the 6th kyu requirements.",
        retriever=retriever,
        llm=llm,
    )

    assert result.metadata["rank_validation"]["ok"] is True
    assert result.anchors.metadata["rank"] == "6th kyu"
    assert result.brief.metadata["rank"] == "6th kyu"
    assert "6th kyu" in retriever.calls[0][0].lower()
    assert "### Exact Rank Requirements" in result.brief.brief_markdown
    assert "### Optional Supporting Definitions" in result.brief.brief_markdown

    grounding = "\n".join([result.anchors.anchor_block, result.brief.brief_markdown]).lower()
    draft = result.draft.body.lower()

    assert "6th kyu" in grounding
    assert "rokushakubo" in grounding
    assert "6th kyu" in draft
    assert "rokushakubo" in draft

    for forbidden in [
        "7th kyu",
        "8th kyu",
        "9th kyu",
        "sassho-ouchi",
        "kote-ouchi",
        "conditioning",
        "sparring",
        "meditation",
        "black gi",
        "hanbo",
        "bokken",
        "tanto",
    ]:
        assert forbidden not in grounding
        assert forbidden not in draft

    prompt = llm.prompts[0]
    assert "Do not generalize with generic martial arts filler when exact source facts are sparse." in prompt
    assert (
        "Do not add conditioning, sparring, meditation, emotional control, or gear sections unless grounded in provided sources."
        in prompt
    )
    assert "Sassho-ouchi" not in prompt
    assert "Kote-ouchi" not in prompt
    assert "Black Gi" not in prompt
    assert "Bokken" not in prompt
    assert "Tanto" not in prompt


def _contaminated_rank_context(text: str) -> tuple[BriefResult, AnchorResult]:
    metadata = {"rank_overview": True, "rank": "6th kyu"}
    brief = BriefResult(
        title="6th Kyu overview",
        brief_markdown=text,
        sources_used=["nttv rank requirements.txt"],
        metadata=dict(metadata),
    )
    anchors = AnchorResult(
        anchor_block=text,
        anchors=[line for line in text.splitlines() if line.startswith("- ")],
        metadata=dict(metadata),
    )
    return brief, anchors


def test_rank_overview_draft_aborts_on_neighboring_rank_contamination(monkeypatch):
    contaminated = "\n".join(
        [
            "### Blog Brief",
            "### Exact Rank Requirements",
            "- Rank: 6th Kyu",
            "- Weapon: Rokushakubo",
            "- 7th Kyu material: Katana cuts",
            "- 8th Kyu material: Hanbo basics",
            "- 9th Kyu material: foundational kamae",
        ]
    )
    brief, anchors = _contaminated_rank_context(contaminated)
    llm = CaptureLLM()

    monkeypatch.setattr(
        orchestrator,
        "_collect_context",
        lambda *args, **kwargs: (brief, anchors),
    )

    result = draft_from_outline(
        BlogRequest(hook_title="What are the skills for 6th kyu?"),
        "## Outline\n- Stay focused on 6th kyu.",
        retriever=CaptureRetriever(),
        llm=llm,
    )

    assert llm.prompts == []
    assert result.metadata["aborted"] is True
    assert result.metadata["abort_reason"] == "rank_grounding_validation_failed"
    assert result.metadata["rank_validation"]["ok"] is False
    assert set(result.metadata["rank_validation"]["unrelated_ranks"]) == {
        "7th kyu",
        "8th kyu",
        "9th kyu",
    }
    assert "Draft generation aborted" in result.draft.body
    assert "Unexpected rank references" in result.draft.body


def test_rank_overview_draft_aborts_on_unrelated_gear_contamination(monkeypatch):
    contaminated = "\n".join(
        [
            "### Blog Brief",
            "### Exact Rank Requirements",
            "- Rank: 6th Kyu",
            "- Weapon: Rokushakubo",
            "### Optional Supporting Definitions",
            "- Katana: long sword",
            "- Hanbo: three-foot staff",
        ]
    )
    brief, anchors = _contaminated_rank_context(contaminated)
    llm = CaptureLLM()

    monkeypatch.setattr(
        orchestrator,
        "_collect_context",
        lambda *args, **kwargs: (brief, anchors),
    )

    result = draft_from_outline(
        BlogRequest(hook_title="What are the skills for 6th kyu?"),
        "## Outline\n- Stay focused on 6th kyu.",
        retriever=CaptureRetriever(),
        llm=llm,
    )

    assert llm.prompts == []
    assert result.metadata["aborted"] is True
    assert result.metadata["rank_validation"]["ok"] is False
    assert result.metadata["rank_validation"]["unrelated_ranks"] == []
    assert set(result.metadata["rank_validation"]["unrelated_gear"]) == {
        "hanbo",
        "katana",
    }
    assert "Unexpected weapon/gear references" in result.draft.body
