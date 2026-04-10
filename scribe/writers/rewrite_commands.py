"""Rewrite command presets for the separate blog workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewriteCommand:
    original: str
    instruction: str
    preset: str = "custom"
    target_heading: str | None = None
    heading_found: bool = False


PRESET_INSTRUCTIONS = {
    "cut_20": "Cut the draft by about 20% while preserving factual anchors, key claims, and the core structure.",
    "more_story": "Add more story, concrete scene-setting, and narrative flow while preserving the factual anchors.",
    "more_direct": "Make the draft more direct, plainspoken, and decisive while preserving the factual anchors. Remove hedging and filler.",
    "less_technical": "Reduce technical language and explain necessary terms simply while preserving the factual anchors.",
    "stronger_ending": "Rewrite the ending so it lands with a stronger takeaway while preserving the factual anchors.",
    "rewrite_intro": "Rewrite only the introduction/opening. Keep the rest of the draft unchanged except for light transitions.",
}


def _norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\s_-]+", " ", text)
    text = text.strip(" .!?:;")
    return text


def extract_headings(draft: str) -> list[str]:
    headings: list[str] = []
    for raw in (draft or "").splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", raw)
        if match:
            heading = match.group(1).strip().rstrip("#").strip()
            if heading:
                headings.append(heading)
    return headings


def _section_heading_from_command(command: str) -> str | None:
    patterns = [
        r"^rewrite\s+section\s+by\s+heading\s*:\s*(.+)$",
        r"^rewrite\s+section\s*:\s*(.+)$",
        r"^rewrite\s+heading\s*:\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, command, flags=re.IGNORECASE)
        if match:
            heading = match.group(1).strip().strip("\"'")
            return heading or None
    return None


def parse_rewrite_command(command: str, draft: str = "") -> RewriteCommand:
    original = (command or "").strip()
    normalized = _norm(original)

    heading = _section_heading_from_command(original)
    if heading:
        headings = extract_headings(draft)
        matched = next((item for item in headings if item.lower() == heading.lower()), None)
        if matched:
            return RewriteCommand(
                original=original,
                instruction=(
                    f'Rewrite only the section headed "{matched}". '
                    "Preserve all other sections unless one transition needs a light adjustment."
                ),
                preset="rewrite_section",
                target_heading=matched,
                heading_found=True,
            )
        return RewriteCommand(
            original=original,
            instruction=(
                f'Rewrite the section requested by heading "{heading}" if it exists. '
                "If no matching heading exists, make the closest targeted section edit without inventing facts."
            ),
            preset="rewrite_section",
            target_heading=heading,
            heading_found=False,
        )

    aliases = {
        "cut 20": "cut_20",
        "cut 20%": "cut_20",
        "cut by 20": "cut_20",
        "cut by 20%": "cut_20",
        "shorten 20": "cut_20",
        "shorten 20%": "cut_20",
        "more story": "more_story",
        "make it more story": "more_story",
        "more direct": "more_direct",
        "make it more direct": "more_direct",
        "less technical": "less_technical",
        "make it less technical": "less_technical",
        "stronger ending": "stronger_ending",
        "make the ending stronger": "stronger_ending",
        "rewrite intro": "rewrite_intro",
        "rewrite introduction": "rewrite_intro",
        "rewrite the intro": "rewrite_intro",
        "rewrite the introduction": "rewrite_intro",
    }

    preset = aliases.get(normalized)
    if preset:
        return RewriteCommand(
            original=original,
            instruction=PRESET_INSTRUCTIONS[preset],
            preset=preset,
        )

    instruction = original or "Rewrite for clarity while preserving factual anchors and core intent."
    return RewriteCommand(original=original, instruction=instruction, preset="custom")


def normalize_rewrite_instruction(command: str, draft: str = "") -> str:
    return parse_rewrite_command(command, draft=draft).instruction
