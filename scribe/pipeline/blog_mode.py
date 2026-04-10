"""Deterministic anchor scaffolding for the separate blog-mode pipeline."""

from __future__ import annotations

from typing import Any

from extractors import try_extract_answer
from scribe.models import AnchorResult, BlogRequest
from scribe.text_seam import build_extraction_context


def _clean_line(text: str) -> str:
    return " ".join((text or "").strip().split())


def _clip(text: str, max_len: int = 180) -> str:
    text = _clean_line(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _candidate_queries(request: BlogRequest) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if not value:
            return
        cleaned = _clean_line(value)
        if not cleaned:
            return
        folded = cleaned.lower()
        if folded in seen:
            return
        seen.add(folded)
        candidates.append(cleaned)

    add(request.hook_title)
    add(request.premise)
    for term in request.include_terms[:4]:
        add(f"What is {term}?")

    return candidates


def build_anchor_result(
    request: BlogRequest,
    passages: list[dict[str, Any]],
    *,
    max_chars: int = 900,
    max_items: int = 10,
) -> AnchorResult:
    """Build a compact markdown anchor block for blog-mode drafting."""
    lines: list[str] = []
    seen: set[str] = set()
    total = 0
    extractor_queries = _candidate_queries(request)
    used_extractor = False

    def add_line(line: str) -> None:
        nonlocal total
        cleaned = _clean_line(line)
        if not cleaned:
            return
        folded = cleaned.lower()
        if folded in seen:
            return
        if len(lines) >= max_items:
            return
        projected = total + len(cleaned) + 1
        if projected > max_chars:
            return
        lines.append(cleaned)
        seen.add(folded)
        total = projected

    add_line(f"- Mode: {request.mode.value}")
    add_line(f"- Hook: {_clip(request.hook_title, 120)}")
    if request.premise:
        add_line(f"- Premise: {_clip(request.premise, 140)}")

    for query in extractor_queries:
        answer = try_extract_answer(query, passages)
        if not answer:
            continue
        used_extractor = True
        add_line(f"- Extractor anchor: {_clip(answer, 220)}")

    passage_anchor_block = build_extraction_context(
        passages,
        max_chars=max_chars,
        max_items=max_items,
    )
    for line in passage_anchor_block.splitlines():
        add_line(line)

    block = "### Blog Anchors"
    if lines:
        block = f"{block}\n" + "\n".join(lines)

    if len(block) > max_chars:
        block = block[: max_chars - 3].rstrip() + "..."

    return AnchorResult(
        anchor_block=block,
        anchors=list(lines),
        metadata={
            "mode": request.mode.value,
            "passage_count": len(passages),
            "queries": extractor_queries,
            "anchor_count": len(lines),
            "char_count": len(block),
            "used_extractor": used_extractor,
        },
    )
