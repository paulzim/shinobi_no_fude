from unittest.mock import Mock

from scribe.models import BlogRequest, BriefResult
from scribe.pipeline.blog_mode import build_brief_result


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
    assert "- Anchors:" in result.brief_markdown
    assert "### Sources Used" in result.brief_markdown
    assert "very long raw chunk" not in result.brief_markdown.lower()
