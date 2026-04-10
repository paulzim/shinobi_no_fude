from scribe.models import AnchorResult, BlogRequest
from scribe.pipeline.blog_mode import build_anchor_result


def _rank_passages() -> list[dict]:
    return [
        {
            "text": "\n".join(
                [
                    "8th Kyu",
                    "Weapon: Hanbo",
                    "Kihon Happo: Ichimonji no Kata",
                    "San Shin no Kata: Chi no Kata; Sui no Kata",
                ]
            ),
            "source": "data/nttv rank requirements.txt",
            "meta": {"source": "data/nttv rank requirements.txt"},
        }
    ]


def test_build_anchor_result_returns_compact_structured_markdown():
    req = BlogRequest(
        hook_title="What weapon do I learn at 8th kyu?",
        premise="Anchor a short blog brief around the first weapon requirement.",
        mode="outline",
    )

    result = build_anchor_result(req, _rank_passages(), max_chars=320, max_items=6)

    assert isinstance(result, AnchorResult)
    assert result.anchor_block.startswith("### Blog Anchors")
    assert "- Mode: outline" in result.anchor_block
    assert "- Hook: What weapon do I learn at 8th kyu?" in result.anchor_block
    assert "Hanbo" in result.anchor_block
    assert len(result.anchor_block) <= 320
    assert result.metadata["used_extractor"] is True
    assert result.metadata["char_count"] == len(result.anchor_block)


def test_build_anchor_result_is_deterministic_for_same_inputs():
    req = BlogRequest(
        hook_title="What weapon do I learn at 8th kyu?",
        include_terms=["hanbo"],
        mode="draft",
    )

    first = build_anchor_result(req, _rank_passages(), max_chars=360, max_items=7)
    second = build_anchor_result(req, _rank_passages(), max_chars=360, max_items=7)

    assert first.anchor_block == second.anchor_block
    assert first.anchors == second.anchors
    assert first.metadata == second.metadata
