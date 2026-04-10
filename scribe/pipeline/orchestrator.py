"""Headless blog-mode orchestration built from existing RAG/extractor seams."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from scribe.config import DEFAULT_BLOG_MODE_SETTINGS, BlogModeSettings
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


def _default_llm(
    prompt: str,
    system: str = "You are a concise blog-writing assistant.",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, str]:
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


def _resolve_settings(settings: BlogModeSettings | None) -> BlogModeSettings:
    return settings or DEFAULT_BLOG_MODE_SETTINGS


def _resolve_count(explicit: int | None, default: int) -> int:
    return explicit if explicit is not None else default


def _call_llm(
    llm: LLMCallable | None,
    prompt: str,
    *,
    mode: BlogMode,
    settings: BlogModeSettings,
) -> tuple[str, str]:
    caller = llm or _default_llm
    kwargs = {
        "system": "You are a concise blog-writing assistant.",
        "temperature": settings.temperature_for(mode),
        "max_tokens": settings.max_tokens_for(mode),
    }
    try:
        return caller(prompt, **kwargs)
    except TypeError:
        return caller(prompt, system=kwargs["system"])


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
    settings: BlogModeSettings | None = None,
    top_k: int | None = None,
    top_k_keep: int | None = None,
    brief_max_chars: int | None = None,
    anchor_max_chars: int | None = None,
    prompt_max_chars: int | None = None,
) -> HookBuildResult:
    """Generate title variants, hook expansions, an outline, and anchor metadata."""
    cfg = _resolve_settings(settings)
    resolved_top_k = _resolve_count(top_k, cfg.rag_top_k_retrieve)
    resolved_top_k_keep = _resolve_count(top_k_keep, cfg.rag_top_k_keep)
    resolved_brief_max_chars = cfg.brief_char_limit(brief_max_chars)
    resolved_anchor_max_chars = cfg.prompt_char_limit(anchor_max_chars or 900)
    resolved_prompt_max_chars = cfg.prompt_char_limit(prompt_max_chars)

    brief, anchors = _collect_context(
        request,
        retriever=retriever,
        top_k=resolved_top_k,
        top_k_keep=resolved_top_k_keep,
        brief_max_chars=resolved_brief_max_chars,
        anchor_max_chars=resolved_anchor_max_chars,
    )

    hook_request = _stage_request(request, BlogMode.HOOK_EXPANSION)
    hook_prompt = build_writer_prompt(
        hook_request,
        anchors,
        brief,
        max_chars=resolved_prompt_max_chars,
    )
    hook_text, hook_raw = _call_llm(
        llm,
        hook_prompt,
        mode=BlogMode.HOOK_EXPANSION,
        settings=cfg,
    )

    outline_request = _stage_request(request, BlogMode.OUTLINE)
    outline_prompt = build_writer_prompt(
        outline_request,
        anchors,
        brief,
        outline=hook_text,
        max_chars=resolved_prompt_max_chars,
    )
    outline_text, outline_raw = _call_llm(llm, outline_prompt, mode=BlogMode.OUTLINE, settings=cfg)

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
            "settings": {
                "context_limit": cfg.active_context_limit,
                "rag_budget_chars": cfg.rag_budget_chars,
                "rag_top_k_retrieve": resolved_top_k,
                "rag_top_k_keep": resolved_top_k_keep,
                "hook_temperature": cfg.temperature_for(BlogMode.HOOK_EXPANSION),
                "outline_temperature": cfg.temperature_for(BlogMode.OUTLINE),
                "hook_max_output_tokens": cfg.max_tokens_for(BlogMode.HOOK_EXPANSION),
                "outline_max_output_tokens": cfg.max_tokens_for(BlogMode.OUTLINE),
            },
        },
    )


def draft_from_outline(
    request: BlogRequest,
    outline: str,
    *,
    retriever: RetrieverCallable | None = None,
    llm: LLMCallable | None = None,
    settings: BlogModeSettings | None = None,
    top_k: int | None = None,
    top_k_keep: int | None = None,
    brief_max_chars: int | None = None,
    anchor_max_chars: int | None = None,
    prompt_max_chars: int | None = None,
) -> DraftPipelineResult:
    """Build a brief and anchors, then draft from a supplied outline."""
    cfg = _resolve_settings(settings)
    draft_request = _stage_request(request, BlogMode.DRAFT)
    brief, anchors = _collect_context(
        draft_request,
        retriever=retriever,
        top_k=_resolve_count(top_k, cfg.rag_top_k_retrieve),
        top_k_keep=_resolve_count(top_k_keep, cfg.rag_top_k_keep),
        brief_max_chars=cfg.brief_char_limit(brief_max_chars),
        anchor_max_chars=cfg.prompt_char_limit(anchor_max_chars or 900),
    )
    prompt = build_writer_prompt(
        draft_request,
        anchors,
        brief,
        outline=outline,
        max_chars=cfg.prompt_char_limit(prompt_max_chars),
    )
    text, raw = _call_llm(llm, prompt, mode=BlogMode.DRAFT, settings=cfg)

    return DraftPipelineResult(
        draft=DraftResult(title=request.hook_title, body=text.strip()),
        anchors=anchors,
        brief=brief,
        metadata={
            "raw": raw,
            "prompt_chars": len(prompt),
            "temperature": cfg.temperature_for(BlogMode.DRAFT),
            "max_output_tokens": cfg.max_tokens_for(BlogMode.DRAFT),
        },
    )


def polish_draft(
    request: BlogRequest,
    draft: str,
    *,
    retriever: RetrieverCallable | None = None,
    llm: LLMCallable | None = None,
    settings: BlogModeSettings | None = None,
    top_k: int | None = None,
    top_k_keep: int | None = None,
    brief_max_chars: int | None = None,
    anchor_max_chars: int | None = None,
    prompt_max_chars: int | None = None,
) -> DraftPipelineResult:
    """Polish a draft for clarity and tightness while preserving grounded facts."""
    cfg = _resolve_settings(settings)
    polish_request = _stage_request(request, BlogMode.POLISH)
    brief, anchors = _collect_context(
        polish_request,
        retriever=retriever,
        top_k=_resolve_count(top_k, cfg.rag_top_k_retrieve),
        top_k_keep=_resolve_count(top_k_keep, cfg.rag_top_k_keep),
        brief_max_chars=cfg.brief_char_limit(brief_max_chars),
        anchor_max_chars=cfg.prompt_char_limit(anchor_max_chars or 700),
    )
    prompt = build_writer_prompt(
        polish_request,
        anchors,
        brief,
        draft=draft,
        max_chars=cfg.prompt_char_limit(prompt_max_chars),
    )
    text, raw = _call_llm(llm, prompt, mode=BlogMode.POLISH, settings=cfg)

    return DraftPipelineResult(
        draft=DraftResult(title=request.hook_title, body=text.strip()),
        anchors=anchors,
        brief=brief,
        metadata={
            "raw": raw,
            "prompt_chars": len(prompt),
            "temperature": cfg.temperature_for(BlogMode.POLISH),
            "max_output_tokens": cfg.max_tokens_for(BlogMode.POLISH),
        },
    )


def rewrite_with_instruction(
    request: BlogRequest,
    draft: str,
    instruction: str,
    *,
    retriever: RetrieverCallable | None = None,
    llm: LLMCallable | None = None,
    settings: BlogModeSettings | None = None,
    top_k: int | None = None,
    top_k_keep: int | None = None,
    brief_max_chars: int | None = None,
    anchor_max_chars: int | None = None,
    prompt_max_chars: int | None = None,
) -> DraftPipelineResult:
    """Run a targeted rewrite loop with an explicit edit instruction."""
    cfg = _resolve_settings(settings)
    rewrite_request = _stage_request(request, BlogMode.REWRITE)
    brief, anchors = _collect_context(
        rewrite_request,
        retriever=retriever,
        top_k=_resolve_count(top_k, cfg.rag_top_k_retrieve),
        top_k_keep=_resolve_count(top_k_keep, cfg.rag_top_k_keep),
        brief_max_chars=cfg.brief_char_limit(brief_max_chars),
        anchor_max_chars=cfg.prompt_char_limit(anchor_max_chars or 700),
    )
    command = parse_rewrite_command(instruction, draft=draft)
    targeted_draft = f"Edit instruction: {command.instruction}\n\nDraft:\n{draft.strip()}"
    prompt = build_writer_prompt(
        rewrite_request,
        anchors,
        brief,
        draft=targeted_draft,
        max_chars=cfg.prompt_char_limit(prompt_max_chars),
    )
    text, raw = _call_llm(llm, prompt, mode=BlogMode.REWRITE, settings=cfg)

    return DraftPipelineResult(
        draft=DraftResult(title=request.hook_title, body=text.strip()),
        anchors=anchors,
        brief=brief,
        metadata={
            "raw": raw,
            "prompt_chars": len(prompt),
            "instruction": command.instruction,
            "command": command,
            "temperature": cfg.temperature_for(BlogMode.REWRITE),
            "max_output_tokens": cfg.max_tokens_for(BlogMode.REWRITE),
        },
    )
