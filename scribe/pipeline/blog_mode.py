"""Deterministic anchor scaffolding for the separate blog-mode pipeline."""

from __future__ import annotations

import os
from typing import Any

from extractors import try_extract_answer
from scribe.models import AnchorResult, BlogRequest, BriefResult
from scribe.text_seam import build_extraction_context, get_passage_extractions


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


def _build_retrieval_query(request: BlogRequest) -> str:
    parts = [request.hook_title]
    if request.premise:
        parts.append(request.premise)
    if request.include_terms:
        parts.append("Focus terms: " + ", ".join(request.include_terms[:5]))
    if request.avoid_terms:
        parts.append("Avoid: " + ", ".join(request.avoid_terms[:5]))
    return " | ".join(parts)


def _default_retriever(query: str, *, k: int) -> list[dict[str, Any]]:
    from app import retrieve

    return retrieve(query, k=k)


def _source_name(passage: dict[str, Any]) -> str:
    source = passage.get("source") or (passage.get("meta") or {}).get("source") or ""
    return os.path.basename(source) or "<unknown>"


def _source_note(passage: dict[str, Any]) -> str:
    source_name = _source_name(passage)
    extractions = get_passage_extractions(passage)

    bits: list[str] = []
    for title in (extractions.get("titles") or [])[:1]:
        bits.append(_clip(str(title), 60))
    for anchor in (extractions.get("anchors") or [])[:2]:
        bits.append(_clip(str(anchor), 90))

    if not bits:
        return source_name
    return f"{source_name}: {'; '.join(bits)}"


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


def build_brief_result(
    request: BlogRequest,
    *,
    retriever: Any | None = None,
    top_k: int = 18,
    top_k_keep: int = 8,
    max_chars: int = 1600,
) -> BriefResult:
    """Wrap retrieval for blog mode and render a compact, bounded brief."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k_keep <= 0:
        raise ValueError("top_k_keep must be positive")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    retriever = retriever or _default_retriever
    query = _build_retrieval_query(request)
    candidates = list(retriever(query, k=top_k) or [])
    kept = candidates[:top_k_keep]
    anchor_result = build_anchor_result(
        request,
        kept,
        max_chars=min(700, max_chars),
        max_items=8,
    )

    lines: list[str] = []
    seen: set[str] = set()
    total = len("### Blog Brief\n")

    def add_line(line: str) -> None:
        nonlocal total
        cleaned = _clean_line(line)
        if not cleaned:
            return
        folded = cleaned.lower()
        if folded in seen:
            return
        projected = total + len(cleaned) + 1
        if projected > max_chars:
            return
        lines.append(cleaned)
        seen.add(folded)
        total = projected

    add_line(f"- Hook: {_clip(request.hook_title, 120)}")
    add_line(f"- Mode: {request.mode.value}")
    add_line(f"- Target length: ~{request.length_target_words} words")
    add_line(f"- Creativity: {request.creativity_level.value}")
    if request.premise:
        add_line(f"- Premise: {_clip(request.premise, 140)}")

    anchor_lines = anchor_result.anchors[:3]
    if anchor_lines:
        joined = "; ".join(line.lstrip("- ").strip() for line in anchor_lines)
        add_line(f"- Anchors: {_clip(joined, 220)}")

    for passage in kept[: min(top_k_keep, 4)]:
        add_line(f"- Retrieval note: {_clip(_source_note(passage), 220)}")

    source_names: list[str] = []
    for passage in kept:
        source = _source_name(passage)
        if source not in source_names:
            source_names.append(source)

    add_line("### Sources Used")
    for source in source_names:
        add_line(f"- {source}")

    brief_markdown = "### Blog Brief"
    if lines:
        brief_markdown = f"{brief_markdown}\n" + "\n".join(lines)
    if len(brief_markdown) > max_chars:
        brief_markdown = brief_markdown[: max_chars - 3].rstrip() + "..."

    return BriefResult(
        title=request.hook_title,
        sections=list(lines),
        brief_markdown=brief_markdown,
        sources_used=source_names,
        metadata={
            "query": query,
            "top_k_requested": top_k,
            "top_k_keep": top_k_keep,
            "candidate_count": len(candidates),
            "kept_count": len(kept),
            "char_count": len(brief_markdown),
        },
    )
