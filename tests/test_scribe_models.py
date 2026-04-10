import pytest

from scribe.models import (
    AnchorResult,
    BlogMode,
    BlogRequest,
    BriefResult,
    CreativityLevel,
    DraftResult,
)


def test_blog_request_defaults():
    req = BlogRequest(hook_title="Why Hanbo Still Matters")

    assert req.hook_title == "Why Hanbo Still Matters"
    assert req.premise is None
    assert req.length_target_words == 2000
    assert req.creativity_level == CreativityLevel.MED
    assert req.include_terms == []
    assert req.avoid_terms == []
    assert req.mode == BlogMode.DRAFT


def test_blog_request_coerces_enum_strings_and_copies_term_lists():
    include_terms = ["hanbo", "distance"]
    avoid_terms = ["fluff"]

    req = BlogRequest(
        hook_title="Expand this hook",
        creativity_level="high",
        include_terms=include_terms,
        avoid_terms=avoid_terms,
        mode="outline",
    )

    include_terms.append("timing")
    avoid_terms.append("filler")

    assert req.creativity_level == CreativityLevel.HIGH
    assert req.mode == BlogMode.OUTLINE
    assert req.include_terms == ["hanbo", "distance"]
    assert req.avoid_terms == ["fluff"]


def test_blog_request_rejects_invalid_required_fields():
    with pytest.raises(ValueError, match="hook_title is required"):
        BlogRequest(hook_title=" ")

    with pytest.raises(ValueError, match="length_target_words must be positive"):
        BlogRequest(hook_title="Test", length_target_words=0)

    with pytest.raises(ValueError, match="creativity_level must be one of"):
        BlogRequest(hook_title="Test", creativity_level="wild")

    with pytest.raises(ValueError, match="mode must be one of"):
        BlogRequest(hook_title="Test", mode="notes")


def test_result_objects_create_with_minimal_data():
    anchors = AnchorResult(
        anchor_block="### Blog Anchors\n- Extractor anchor: Weapon: Hanbo",
        anchors=["- Extractor anchor: Weapon: Hanbo", "- [Rank] 8th Kyu"],
        metadata={"anchor_count": 2},
    )
    brief = BriefResult(
        title="Hanbo post outline",
        sections=["Hook", "Basics", "Closing"],
        brief_markdown="### Blog Brief\n- Hook: Hanbo post outline",
        sources_used=["nttv rank requirements.txt"],
        metadata={"kept_count": 3},
    )
    draft = DraftResult(title="Hanbo post", body="Hanbo training starts at 8th kyu.")

    assert anchors.anchor_block.startswith("### Blog Anchors")
    assert anchors.anchors == ["- Extractor anchor: Weapon: Hanbo", "- [Rank] 8th Kyu"]
    assert anchors.metadata["anchor_count"] == 2
    assert brief.title == "Hanbo post outline"
    assert brief.sections == ["Hook", "Basics", "Closing"]
    assert brief.brief_markdown.startswith("### Blog Brief")
    assert brief.sources_used == ["nttv rank requirements.txt"]
    assert brief.metadata["kept_count"] == 3
    assert draft.title == "Hanbo post"
    assert "8th kyu" in draft.body.lower()
