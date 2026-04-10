"""Headless blog-mode orchestration built from existing RAG/extractor seams."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from scribe.models import AnchorResult, BlogMode, BlogRequest, BriefResult, DraftResult
from scribe.pipeline.blog_mode import (
    _build_retrieval_query,
    _default_retriever,
    build_anchor_result,
    build_brief_result,
)
from scribe.writers.prompt_builder import build_writer_prompt
from scribe.writers.rewrite_commands import parse_rewrite_command


LLMCallable = Callable[..., tuple[str, str]]
RetrieverCallable = Callable[..., list[dict[str, Any]]]


@dataclass(slots=True)
class HookBuildResult:
    title_variants: list[str] = field(default_factory=list)
    hook_expansions: list[str] = field(default_factory=list)
    outline: str = ""
    anchors: AnchorResult = field(default_factory=AnchorResult)
    brief: BriefResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DraftPipelineResult:
    draft: DraftResult
    anchors: AnchorResult
    brief: BriefResult
    metadata: dict[str, Any] = field(default_factory=dict)


def _default_llm(prompt: str, system: str = "You are a concise blog-writing assistant.") -> tuple[str, str]:
    from app import call_llm

    return call_llm(prompt, system=system)


def _stage_request(request: BlogRequest, mode: BlogMode) -> BlogRequest:
    return replace(request, mode=mode)


def _collect_context(
    request: BlogRequest,
    *,
    retriever: RetrieverCallable | None,
    top_k: int,
    top_k_keep: int,
    brief_max_chars: int,
    anchor_max_chars: int,
) -> tuple[BriefResult, AnchorResult]:
    actual_retriever = retriever or _default_retriever
    query = _build_retrieval_query(request)
    candidates = list(actual_retriever(query, k=top_k) or [])
    kept = candidates[:top_k_keep]

    anchors = build_anchor_result(request, kept, max_chars=anchor_max_chars, max_items=10)

    def replay_retriever(_query: str, *, k: int) -> list[dict[str, Any]]:
        return candidates[:k]

    brief = build_brief_result(
        request,
        retriever=replay_retriever,
        top_k=top_k,
        top_k_keep=top_k_keep,
        max_chars=brief_max_chars,
    )
    brief.metadata["retrieval_query"] = query
    return brief, anchors


def _call_llm(llm: LLMCallable | None, prompt: str) -> tuple[str, str]:
    caller = llm or _default_llm
    return caller(prompt, system="You are a concise blog-writing assistant.")


def _heading_key(line: str) -> str:
    return line.strip().strip("#").strip().rstrip(":").lower()


def _section_bullets(text: str, names: set[str]) -> list[str]:
    items: list[str] = []
    in_section = False

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        key = _heading_key(line)
        if key in names:
            in_section = True
            continue
        if in_section and line.startswith("#"):
            break
        if not in_section:
            continue

        if line.startswith(("- ", "* ")):
            items.append(line[2:].strip())
        elif line[0:2].isdigit() and "." in line[:4]:
            items.append(line.split(".", 1)[1].strip())
        else:
            items.append(line)

    return [item for item in items if item]


def _fallback_items(text: str, fallback: str) -> list[str]:
    items = [line.strip("-* ").strip() for line in (text or "").splitlines() if line.strip()]
    return items[:4] or [fallback]


def build_around_hook(
    request: BlogRequest,
    *,
    retriever: RetrieverCallable | None = None,
    llm: LLMCallable | None = None,
    top_k: int = 18,
    top_k_keep: int = 8,
    brief_max_chars: int = 1600,
    anchor_max_chars: int = 900,
    prompt_max_chars: int = 2600,
) -> HookBuildResult:
    """Generate title variants, hook expansions, an outline, and anchor metadata."""
    brief, anchors = _collect_context(
        request,
        retriever=retriever,
        top_k=top_k,
        top_k_keep=top_k_keep,
        brief_max_chars=brief_max_chars,
        anchor_max_chars=anchor_max_chars,
    )

    hook_request = _stage_request(request, BlogMode.HOOK_EXPANSION)
    hook_prompt = build_writer_prompt(
        hook_request,
        anchors,
        brief,
        max_chars=prompt_max_chars,
    )
    hook_text, hook_raw = _call_llm(llm, hook_prompt)

    outline_request = _stage_request(request, BlogMode.OUTLINE)
    outline_prompt = build_writer_prompt(
        outline_request,
        anchors,
        brief,
        outline=hook_text,
        max_chars=prompt_max_chars,
    )
    outline_text, outline_raw = _call_llm(llm, outline_prompt)

    title_variants = _section_bullets(hook_text, {"title variants", "titles"})
    hook_expansions = _section_bullets(hook_text, {"hook expansions", "hooks"})

    return HookBuildResult(
        title_variants=title_variants or [request.hook_title],
        hook_expansions=hook_expansions or _fallback_items(hook_text, request.hook_title),
        outline=outline_text.strip(),
        anchors=anchors,
        brief=brief,
        metadata={
            "anchor_metadata": anchors.metadata,
            "brief_metadata": brief.metadata,
            "hook_raw": hook_raw,
            "outline_raw": outline_raw,
            "hook_prompt_chars": len(hook_prompt),
            "outline_prompt_chars": len(outline_prompt),
        },
    )


def draft_from_outline(
    request: BlogRequest,
    outline: str,
    *,
    retriever: RetrieverCallable | None = None,
    llm: LLMCallable | None = None,
    top_k: int = 18,
    top_k_keep: int = 8,
    brief_max_chars: int = 1600,
    anchor_max_chars: int = 900,
    prompt_max_chars: int = 3000,
) -> DraftPipelineResult:
    """Build a brief and anchors, then draft from a supplied outline."""
    draft_request = _stage_request(request, BlogMode.DRAFT)
    brief, anchors = _collect_context(
        draft_request,
        retriever=retriever,
        top_k=top_k,
        top_k_keep=top_k_keep,
        brief_max_chars=brief_max_chars,
        anchor_max_chars=anchor_max_chars,
    )
    prompt = build_writer_prompt(
        draft_request,
        anchors,
        brief,
        outline=outline,
        max_chars=prompt_max_chars,
    )
    text, raw = _call_llm(llm, prompt)

    return DraftPipelineResult(
        draft=DraftResult(title=request.hook_title, body=text.strip()),
        anchors=anchors,
        brief=brief,
        metadata={"raw": raw, "prompt_chars": len(prompt)},
    )


def polish_draft(
    request: BlogRequest,
    draft: str,
    *,
    retriever: RetrieverCallable | None = None,
    llm: LLMCallable | None = None,
    top_k: int = 18,
    top_k_keep: int = 8,
    brief_max_chars: int = 1200,
    anchor_max_chars: int = 700,
    prompt_max_chars: int = 2800,
) -> DraftPipelineResult:
    """Polish a draft for clarity and tightness while preserving grounded facts."""
    polish_request = _stage_request(request, BlogMode.POLISH)
    brief, anchors = _collect_context(
        polish_request,
        retriever=retriever,
        top_k=top_k,
        top_k_keep=top_k_keep,
        brief_max_chars=brief_max_chars,
        anchor_max_chars=anchor_max_chars,
    )
    prompt = build_writer_prompt(
        polish_request,
        anchors,
        brief,
        draft=draft,
        max_chars=prompt_max_chars,
    )
    text, raw = _call_llm(llm, prompt)

    return DraftPipelineResult(
        draft=DraftResult(title=request.hook_title, body=text.strip()),
        anchors=anchors,
        brief=brief,
        metadata={"raw": raw, "prompt_chars": len(prompt)},
    )


def rewrite_with_instruction(
    request: BlogRequest,
    draft: str,
    instruction: str,
    *,
    retriever: RetrieverCallable | None = None,
    llm: LLMCallable | None = None,
    top_k: int = 18,
    top_k_keep: int = 8,
    brief_max_chars: int = 1200,
    anchor_max_chars: int = 700,
    prompt_max_chars: int = 2800,
) -> DraftPipelineResult:
    """Run a targeted rewrite loop with an explicit edit instruction."""
    rewrite_request = _stage_request(request, BlogMode.REWRITE)
    brief, anchors = _collect_context(
        rewrite_request,
        retriever=retriever,
        top_k=top_k,
        top_k_keep=top_k_keep,
        brief_max_chars=brief_max_chars,
        anchor_max_chars=anchor_max_chars,
    )
    command = parse_rewrite_command(instruction, draft=draft)
    targeted_draft = f"Edit instruction: {command.instruction}\n\nDraft:\n{draft.strip()}"
    prompt = build_writer_prompt(
        rewrite_request,
        anchors,
        brief,
        draft=targeted_draft,
        max_chars=prompt_max_chars,
    )
    text, raw = _call_llm(llm, prompt)

    return DraftPipelineResult(
        draft=DraftResult(title=request.hook_title, body=text.strip()),
        anchors=anchors,
        brief=brief,
        metadata={
            "raw": raw,
            "prompt_chars": len(prompt),
            "instruction": command.instruction,
            "command": command,
        },
    )
