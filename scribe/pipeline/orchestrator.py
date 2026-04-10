"""Headless blog-mode orchestration built from existing RAG/extractor seams."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
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


DOMAIN_FOCUS_TERMS = {
    "ichimonji no kata": "Ichimonji no Kata",
    "omote gyaku": "omote gyaku",
    "ura gyaku": "ura gyaku",
    "oni kudaki": "oni kudaki",
    "musha dori": "musha dori",
    "ganseki otoshi": "ganseki otoshi",
    "kihon happo": "kihon happo",
    "sanshin": "sanshin",
    "katana": "katana",
    "hanbo": "hanbo",
    "hanbō": "hanbo",
    "tanto": "tanto",
    "tantō": "tanto",
    "shoto": "shoto",
    "shōtō": "shoto",
    "rokushakubo": "rokushakubo",
    "rokushakubō": "rokushakubo",
    "kusari fundo": "kusari fundo",
    "kyoketsu shoge": "kyoketsu shoge",
    "shuriken": "shuriken",
    "jutte": "jutte",
    "tessen": "tessen",
}


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

    return call_llm(
        prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )


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


def _clip_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _build_verify_claims_prompt(
    draft_text: str,
    anchors: AnchorResult,
    brief: BriefResult,
    *,
    settings: BlogModeSettings,
) -> str:
    prompt = "\n\n".join(
        [
            "You are checking a blog draft against compact factual anchors.",
            (
                "## Task\n"
                "Return at most 5 short bullets naming claims that may need source verification. "
                "Only flag claims not clearly supported by the anchors or brief. "
                'If nothing stands out, return exactly "- None flagged."'
            ),
            (
                "## Output Limits\n"
                f"- Total stored claim text must fit within {settings.verify_claims_max_chars} characters.\n"
                "- Keep each bullet terse; no explanations longer than one sentence."
            ),
            "## Draft\n" + _clip_text(draft_text, 900),
            "## Anchors\n" + _clip_text(anchors.anchor_block, 600),
            "## Brief\n" + _clip_text(brief.brief_markdown, 700),
        ]
    )
    return _clip_text(prompt, min(settings.active_context_limit, 2600))


def _strip_bullet(line: str) -> str:
    line = line.strip()
    if line.startswith(("- ", "* ")):
        return line[2:].strip()
    if line[:1].isdigit() and "." in line[:5]:
        return line.split(".", 1)[1].strip()
    return line


def _parse_verify_claims(text: str, *, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return []

    claims: list[str] = []
    remaining = max_chars
    for raw in (text or "").splitlines():
        claim = _strip_bullet(raw)
        if not claim:
            continue
        if claim.lower().rstrip(".") in {"none", "none flagged", "no weak claims found"}:
            continue

        claim = _clip_text(claim, min(160, remaining))
        if not claim:
            break
        claims.append(claim)
        remaining -= len(claim)
        if remaining <= 0 or len(claims) >= 5:
            break

    return claims


def _verify_claims(
    llm: LLMCallable | None,
    draft_text: str,
    anchors: AnchorResult,
    brief: BriefResult,
    *,
    settings: BlogModeSettings,
) -> tuple[list[str], dict[str, Any]]:
    if not settings.verify_claims_enabled:
        return [], {"enabled": False}

    if settings.verify_claims_max_chars <= 0:
        return [], {"enabled": True, "skipped": "verify_claims_max_chars <= 0"}

    caller = llm or _default_llm
    prompt = _build_verify_claims_prompt(draft_text, anchors, brief, settings=settings)
    kwargs = {
        "system": "You are a terse claim-verification assistant.",
        "temperature": settings.verify_claims_temperature,
        "max_tokens": settings.verify_claims_max_tokens,
    }
    try:
        text, raw = caller(prompt, **kwargs)
    except TypeError:
        text, raw = caller(prompt, system=kwargs["system"])

    claims = _parse_verify_claims(text, max_chars=settings.verify_claims_max_chars)
    return claims, {
        "enabled": True,
        "prompt_chars": len(prompt),
        "max_chars": settings.verify_claims_max_chars,
        "temperature": settings.verify_claims_temperature,
        "max_output_tokens": settings.verify_claims_max_tokens,
        "raw": raw,
    }


def _draft_result(
    request: BlogRequest,
    body: str,
    brief: BriefResult,
    verify_claims: list[str],
    extra_sources: list[str] | None = None,
) -> DraftResult:
    sources = list(brief.sources_used)
    for source in extra_sources or []:
        if source and source not in sources:
            sources.append(source)

    return DraftResult(
        title=request.hook_title,
        body=body.strip(),
        sources_used=sources,
        verify_claims=list(verify_claims),
    )


def _detect_rewrite_focus_term(instruction: str) -> str | None:
    lowered = (instruction or "").lower()
    for term, canonical in sorted(DOMAIN_FOCUS_TERMS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return canonical
    return None


def _focused_rewrite_query(term: str) -> str:
    return f"{term} curriculum details rank requirements glossary"


def _build_focused_rewrite_brief(
    request: BlogRequest,
    focus_term: str | None,
    *,
    retriever: RetrieverCallable | None,
    settings: BlogModeSettings,
) -> tuple[BriefResult | None, dict[str, Any]]:
    if not focus_term:
        return None, {"focus_term": None, "retrieval_attempted": False}

    actual_retriever = retriever or _default_retriever
    query = _focused_rewrite_query(focus_term)
    try:
        candidates = list(actual_retriever(query, k=settings.rewrite_focus_top_k_retrieve) or [])
    except Exception as exc:
        return None, {
            "focus_term": focus_term,
            "retrieval_attempted": True,
            "query": query,
            "candidate_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if not candidates:
        return None, {
            "focus_term": focus_term,
            "retrieval_attempted": True,
            "query": query,
            "candidate_count": 0,
            "kept_count": 0,
        }

    def replay_retriever(_query: str, *, k: int) -> list[dict[str, Any]]:
        return candidates[:k]

    focused_request = replace(
        request,
        hook_title=f"Focused rewrite detail: {focus_term}",
        premise=None,
        include_terms=[focus_term],
        avoid_terms=[],
        mode=BlogMode.REWRITE,
    )
    focused_brief = build_brief_result(
        focused_request,
        retriever=replay_retriever,
        top_k=settings.rewrite_focus_top_k_retrieve,
        top_k_keep=settings.rewrite_focus_top_k_keep,
        max_chars=settings.rewrite_focus_budget_chars,
    )
    focused_brief.metadata.update(
        {
            "focus_term": focus_term,
            "retrieval_attempted": True,
            "query": query,
            "candidate_count": len(candidates),
            "kept_count": min(len(candidates), settings.rewrite_focus_top_k_keep),
            "char_count": len(focused_brief.brief_markdown),
        }
    )
    return focused_brief, focused_brief.metadata


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
    verify_claims, verify_metadata = _verify_claims(
        llm,
        text,
        anchors,
        brief,
        settings=cfg,
    )

    return DraftPipelineResult(
        draft=_draft_result(request, text, brief, verify_claims),
        anchors=anchors,
        brief=brief,
        metadata={
            "raw": raw,
            "prompt_chars": len(prompt),
            "temperature": cfg.temperature_for(BlogMode.DRAFT),
            "max_output_tokens": cfg.max_tokens_for(BlogMode.DRAFT),
            "verify_claims": verify_metadata,
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
    verify_claims, verify_metadata = _verify_claims(
        llm,
        text,
        anchors,
        brief,
        settings=cfg,
    )

    return DraftPipelineResult(
        draft=_draft_result(request, text, brief, verify_claims),
        anchors=anchors,
        brief=brief,
        metadata={
            "raw": raw,
            "prompt_chars": len(prompt),
            "temperature": cfg.temperature_for(BlogMode.POLISH),
            "max_output_tokens": cfg.max_tokens_for(BlogMode.POLISH),
            "verify_claims": verify_metadata,
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
    focus_term = _detect_rewrite_focus_term(f"{command.original} {command.instruction}")
    focused_brief, focused_metadata = _build_focused_rewrite_brief(
        rewrite_request,
        focus_term,
        retriever=retriever,
        settings=cfg,
    )
    targeted_draft = f"Edit instruction: {command.instruction}\n\nDraft:\n{draft.strip()}"
    prompt = build_writer_prompt(
        rewrite_request,
        anchors,
        brief,
        draft=targeted_draft,
        focused_brief=focused_brief,
        max_chars=cfg.prompt_char_limit(prompt_max_chars),
    )
    text, raw = _call_llm(llm, prompt, mode=BlogMode.REWRITE, settings=cfg)
    verify_claims, verify_metadata = _verify_claims(
        llm,
        text,
        anchors,
        brief,
        settings=cfg,
    )

    return DraftPipelineResult(
        draft=_draft_result(
            request,
            text,
            brief,
            verify_claims,
            extra_sources=(focused_brief.sources_used if focused_brief else []),
        ),
        anchors=anchors,
        brief=brief,
        metadata={
            "raw": raw,
            "prompt_chars": len(prompt),
            "instruction": command.instruction,
            "command": command,
            "temperature": cfg.temperature_for(BlogMode.REWRITE),
            "max_output_tokens": cfg.max_tokens_for(BlogMode.REWRITE),
            "focused_rewrite": focused_metadata,
            "verify_claims": verify_metadata,
        },
    )
