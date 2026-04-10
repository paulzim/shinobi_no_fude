import pytest

from scribe.images import (
    ImagePromptPackage,
    build_image_prompt_package,
    render_image_prompt_package,
)
from scribe.models import DraftResult


def _draft() -> DraftResult:
    return DraftResult(
        title="Why Hanbo Still Matters",
        body=(
            "The first weapon lesson changes how a beginner reads distance, "
            "timing, and restraint in training."
        ),
        sources_used=["nttv rank requirements.txt"],
    )


def test_build_image_prompt_package_is_deterministic_and_structured():
    section = "## Hanbo changes distance\nA short staff makes reach visible without making the lesson flashy."

    first = build_image_prompt_package(_draft(), section)
    second = build_image_prompt_package(_draft(), section)

    assert first == second
    assert isinstance(first, ImagePromptPackage)
    assert first.style_preset_name == "editorial_ink_wash"
    assert "Hanbo changes distance" in first.positive_prompt
    assert "Why Hanbo Still Matters" in first.positive_prompt
    assert "watermark" in first.negative_prompt
    assert "wide blog feature image" in first.composition_notes
    assert first.filename_suggestion == "why-hanbo-still-matters-hanbo-changes-distance.png"


def test_render_image_prompt_package_uses_stable_markdown_formatting():
    package = build_image_prompt_package(
        _draft(),
        "## Opening image\nA quiet dojo scene centered on the hanbo.",
    )

    rendered = render_image_prompt_package(package)

    assert rendered.startswith("### Image Prompt Package\n")
    assert "\nPositive prompt:\n" in rendered
    assert "\nNegative prompt:\n" in rendered
    assert "\nComposition notes:\n" in rendered
    assert rendered.endswith(package.filename_suggestion)


def test_build_image_prompt_package_enforces_caps():
    long_section = "## A very long visual section\n" + ("hanbo distance timing posture " * 80)

    package = build_image_prompt_package(
        _draft(),
        long_section,
        positive_max_chars=120,
        negative_max_chars=90,
        composition_max_chars=100,
        filename_max_chars=36,
    )

    assert len(package.positive_prompt) <= 120
    assert len(package.negative_prompt) <= 90
    assert len(package.composition_notes) <= 100
    assert len(package.filename_suggestion) <= 36
    assert package.filename_suggestion.endswith(".png")


def test_build_image_prompt_package_rejects_empty_section_and_unknown_style():
    with pytest.raises(ValueError, match="section_text is required"):
        build_image_prompt_package(_draft(), " ")

    with pytest.raises(ValueError, match="style_preset_name must be one of"):
        build_image_prompt_package(_draft(), "Section text", style_preset_name="unknown")
