"""Stage-aware prompt builder for the separate blog workflow."""

from __future__ import annotations

from typing import Optional

from scribe.models import AnchorResult, BlogMode, BlogRequest, BriefResult


DEFAULT_PROMPT_MAX_CHARS = 2600
SECTION_LIMITS = {
    "request": 420,
    "anchors": 700,
    "brief": 900,
    "outline": 500,
    "draft": 700,
}

STAGE_TASKS = {
    BlogMode.HOOK_EXPANSION: (
        "Expand the hook into sharper angles, promising directions, and a clear blog premise."
    ),
    BlogMode.OUTLINE: (
        "Turn the brief into a practical outline with a strong opening, clean section flow, and a useful close."
    ),
    BlogMode.DRAFT: (
        "Write a full draft grounded in the anchors and brief. Keep facts aligned with the supplied material."
    ),
    BlogMode.POLISH: (
        "Polish the supplied draft for clarity, pacing, and readability while preserving the underlying facts."
    ),
    BlogMode.REWRITE: (
        "Rewrite the supplied draft with a fresher structure and voice while keeping the core facts and intent."
    ),
}

STAGE_OUTPUT_HINTS = {
    BlogMode.HOOK_EXPANSION: "Return 3-5 hook options plus a recommended direction.",
    BlogMode.OUTLINE: "Return a tight outline with headings and short bullet notes.",
    BlogMode.DRAFT: "Return a complete draft only, not notes about the draft.",
    BlogMode.POLISH: "Return the polished article only.",
    BlogMode.REWRITE: "Return the rewritten article only.",
}


def _clean(text: str) -> str:
    return " ".join((text or "").strip().split())


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _request_block(request: BlogRequest) -> str:
    lines = [
        f"- Hook title: {_clip(request.hook_title, 120)}",
        f"- Mode: {request.mode.value}",
        f"- Target length: ~{request.length_target_words} words",
        f"- Creativity: {request.creativity_level.value}",
    ]
    if request.premise:
        lines.append(f"- Premise: {_clip(request.premise, 140)}")
    if request.include_terms:
        lines.append(f"- Include terms: {_clip(', '.join(request.include_terms[:8]), 140)}")
    if request.avoid_terms:
        lines.append(f"- Avoid terms: {_clip(', '.join(request.avoid_terms[:8]), 140)}")
    return "\n".join(lines)


def _anchor_block(anchors: AnchorResult) -> str:
    if anchors.anchor_block.strip():
        return anchors.anchor_block.strip()
    if anchors.anchors:
        return "### Blog Anchors\n" + "\n".join(anchors.anchors)
    return "### Blog Anchors\n- No deterministic anchors were available."


def _brief_block(brief: BriefResult) -> str:
    if brief.brief_markdown.strip():
        return brief.brief_markdown.strip()
    if brief.sections:
        return "### Blog Brief\n" + "\n".join(brief.sections)
    return "### Blog Brief\n- No brief was available."


def _add_block(
    parts: list[str],
    block: str,
    *,
    remaining: int,
    required: bool,
    reserve_for_rest: int = 0,
) -> int:
    block = block.strip()
    if not block or remaining <= 0:
        return remaining

    sep = "\n\n" if parts else ""
    room = remaining - len(sep) - reserve_for_rest
    if room <= 0:
        room = remaining - len(sep)
    if room <= 0:
        return remaining

    if len(block) > room:
        if not required:
            return remaining
        block = _clip(block, room)
        if not block:
            return remaining

    parts.append(sep + block)
    return remaining - len(sep) - len(block)


def _required_min_chars(block: str) -> int:
    title = block.strip().splitlines()[0]
    return len(title) + 8


def build_writer_prompt(
    request: BlogRequest,
    anchors: AnchorResult,
    brief: BriefResult,
    *,
    outline: Optional[str] = None,
    draft: Optional[str] = None,
    max_chars: int = DEFAULT_PROMPT_MAX_CHARS,
) -> str:
    """Build a compact stage-aware writer prompt for blog mode."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    mode = request.mode
    intro = (
        f"You are writing in blog mode.\n"
        f"Stage: {mode.value}\n"
        "Use the anchors as factual guardrails, not as the final prose."
    )
    stage_task = f"## Stage Task\n{STAGE_TASKS[mode]}"
    output_hint = f"## Output\n{STAGE_OUTPUT_HINTS[mode]}"
    request_block = "## Request\n" + _clip(_request_block(request), SECTION_LIMITS["request"])
    anchor_block = "## Anchors\n" + _clip(_anchor_block(anchors), SECTION_LIMITS["anchors"])
    brief_block = "## Brief\n" + _clip(_brief_block(brief), SECTION_LIMITS["brief"])

    optional_blocks: list[str] = []
    if outline:
        optional_blocks.append("## Outline Input\n" + _clip(outline, SECTION_LIMITS["outline"]))
    if draft:
        optional_blocks.append("## Draft Input\n" + _clip(draft, SECTION_LIMITS["draft"]))

    required_blocks = [intro, stage_task, output_hint, request_block, anchor_block, brief_block]
    parts: list[str] = []
    remaining = max_chars
    for idx, block in enumerate(required_blocks):
        reserve_for_rest = sum(_required_min_chars(next_block) for next_block in required_blocks[idx + 1 :])
        remaining = _add_block(
            parts,
            block,
            remaining=remaining,
            required=True,
            reserve_for_rest=reserve_for_rest,
        )

    for block in optional_blocks:
        remaining = _add_block(parts, block, remaining=remaining, required=False)

    prompt = "".join(parts).strip()
    if len(prompt) > max_chars:
        prompt = _clip(prompt, max_chars)
    return prompt
