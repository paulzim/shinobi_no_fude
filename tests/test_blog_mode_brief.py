import pathlib
from unittest.mock import Mock

from scribe.models import BlogRequest, BriefResult
from scribe.pipeline.blog_mode import build_brief_result

RANK_FILE = pathlib.Path("data") / "nttv rank requirements.txt"


def _fake_passage(idx: int) -> dict:
    rank = idx + 1
    return {
        "text": "\n".join(
            [
                f"{rank}th Kyu",
                f"Weapon: Hanbo Variation {idx}",
                "Kihon Happo: Ichimonji no Kata",
                "San Shin no Kata: Chi no Kata; Sui no Kata",
            ]
        ),
        "source": f"data/source_{idx}.txt",
        "meta": {"source": f"data/source_{idx}.txt"},
        "score": 1.0 - (idx * 0.01),
    }


def test_build_brief_result_wraps_retriever_and_keeps_top_k_keep():
    passages = [_fake_passage(i) for i in range(12)]
    retriever = Mock(return_value=passages)
    req = BlogRequest(hook_title="What weapon do I learn at 8th kyu?", mode="outline")

    result = build_brief_result(req, retriever=retriever)

    assert isinstance(result, BriefResult)
    retriever.assert_called_once()
    args, kwargs = retriever.call_args
    assert "What weapon do I learn at 8th kyu?" in args[0]
    assert kwargs["k"] == 18
    assert result.metadata["candidate_count"] == 12
    assert result.metadata["kept_count"] == 8
    assert result.metadata["top_k_keep"] == 8
    assert len(result.sources_used) == 8
    assert "### Sources Used" in result.brief_markdown


def test_build_brief_result_respects_cap_and_avoids_dumping_raw_chunks():
    long_text = (
        "This is a very long raw chunk that should never be copied directly into the brief. "
        * 40
    )
    retriever = Mock(
        return_value=[
            {
                "text": "\n".join(
                    [
                        "8th Kyu",
                        "Weapon: Hanbo",
                        long_text,
                    ]
                ),
                "source": "data/nttv rank requirements.txt",
                "meta": {"source": "data/nttv rank requirements.txt"},
            }
        ]
    )
    req = BlogRequest(
        hook_title="What weapon do I learn at 8th kyu?",
        premise="Keep the brief concise.",
        include_terms=["hanbo"],
    )

    result = build_brief_result(req, retriever=retriever, max_chars=420)

    assert len(result.brief_markdown) <= 420
    assert result.metadata["char_count"] == len(result.brief_markdown)
    assert result.brief_markdown.startswith("### Blog Brief")
    assert "### Exact Rank Requirements" in result.brief_markdown
    assert "### Sources Used" in result.brief_markdown
    assert "very long raw chunk" not in result.brief_markdown.lower()


def test_rank_specific_brief_prioritizes_exact_6th_kyu_requirements():
    retriever = Mock(
        return_value=[
            {
                "text": "Generic conditioning advice, sparring advice, and equipment advice.",
                "source": "data/generic training advice.txt",
                "meta": {"source": "data/generic training advice.txt"},
            },
            {
                "text": RANK_FILE.read_text(encoding="utf-8"),
                "source": "data/nttv rank requirements.txt",
                "meta": {"source": "data/nttv rank requirements.txt"},
            },
            {
                "text": "\n".join(
                    [
                        "Glossary",
                        "Rokushakubo - Six-foot staff",
                        "Hira Ichimonji no Kamae - Flat figure number one posture",
                        "Katana - Long sword",
                        "Hanbo - Three-foot staff",
                        "Sparring - Free exchange training",
                    ]
                ),
                "source": "data/Glossary - edit.txt",
                "meta": {"source": "data/Glossary - edit.txt"},
            },
        ]
    )
    req = BlogRequest(hook_title="What are the skills for 6th kyu?", mode="outline")

    result = build_brief_result(req, retriever=retriever, max_chars=2400)
    brief = result.brief_markdown
    low = brief.lower()

    assert "### Exact Rank Requirements" in brief
    assert "### Optional Supporting Definitions" in brief
    assert "- Rank: 6th Kyu" in brief
    assert "- Weapon: Rokushakubo" in brief
    assert "Weapon Spinning:" in brief
    assert "Rokushakubo: Six-foot staff" in brief
    assert "Hira Ichimonji no Kamae: Flat figure number one posture" in brief
    assert "katana" not in low
    assert "hanbo" not in low
    assert "sparring" not in low
    assert "conditioning advice" not in low
    assert "equipment advice" not in low
    assert "7th kyu" not in low
    assert "8th kyu" not in low
    assert "9th kyu" not in low
    assert result.sources_used == ["nttv rank requirements.txt", "Glossary - edit.txt"]
    assert result.metadata["rank_brief"] is True
    assert result.metadata["rank"] == "6th kyu"


def test_rank_specific_brief_ignores_glossary_terms_absent_from_6th_kyu_block():
    retriever = Mock(
        return_value=[
            {
                "text": RANK_FILE.read_text(encoding="utf-8"),
                "source": "data/nttv rank requirements.txt",
                "meta": {"source": "data/nttv rank requirements.txt"},
            },
            {
                "text": "Glossary\nKatana - Long sword\nHanbo - Three-foot staff",
                "source": "data/Glossary - edit.txt",
                "meta": {"source": "data/Glossary - edit.txt"},
            },
        ]
    )
    req = BlogRequest(hook_title="Give an overview of 6th kyu", mode="outline")

    result = build_brief_result(req, retriever=retriever, max_chars=1800)
    low = result.brief_markdown.lower()

    assert "### Optional Supporting Definitions" in result.brief_markdown
    assert "katana" not in low
    assert "hanbo" not in low
    assert "none from retrieved glossary passages" in low
    assert result.sources_used == ["nttv rank requirements.txt"]
