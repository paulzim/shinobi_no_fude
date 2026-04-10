import pathlib

from scribe.models import BlogRequest
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
