from scribe.writers.rewrite_commands import (
    extract_headings,
    normalize_rewrite_instruction,
    parse_rewrite_command,
)


DRAFT = """# Working Title

## Intro
Opening paragraph.

## Body Mechanics
Technical material.

## Ending
Closing paragraph.
"""


def test_rewrite_command_presets_normalize_to_explicit_instructions():
    cases = {
        "Cut 20%": "about 20%",
        "More story": "story",
        "More direct": "direct",
        "Less technical": "technical language",
        "Stronger ending": "ending",
        "Rewrite intro": "introduction/opening",
    }

    for command, expected in cases.items():
        instruction = normalize_rewrite_instruction(command, draft=DRAFT)
        assert expected.lower() in instruction.lower()
        assert "anchors" in instruction.lower() or command.lower().startswith("rewrite")


def test_rewrite_section_by_heading_uses_existing_heading():
    parsed = parse_rewrite_command("Rewrite section: Body Mechanics", draft=DRAFT)

    assert parsed.preset == "rewrite_section"
    assert parsed.target_heading == "Body Mechanics"
    assert parsed.heading_found is True
    assert 'section headed "Body Mechanics"' in parsed.instruction


def test_rewrite_section_by_heading_handles_missing_heading_conservatively():
    parsed = parse_rewrite_command("Rewrite section: Safety Notes", draft=DRAFT)

    assert parsed.preset == "rewrite_section"
    assert parsed.target_heading == "Safety Notes"
    assert parsed.heading_found is False
    assert "If no matching heading exists" in parsed.instruction


def test_extract_headings_reads_markdown_headings():
    assert extract_headings(DRAFT) == [
        "Working Title",
        "Intro",
        "Body Mechanics",
        "Ending",
    ]
