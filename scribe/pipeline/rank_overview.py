"""Rank-overview grounding for blog mode."""

from __future__ import annotations

import os
import re
from typing import Any

from extractors.rank import (
    _extract_rank_block,
    _find_rank_text_from_passages,
    _norm,
    _rank_key_from_question,
    try_answer_rank_requirements,
)
from scribe.models import AnchorResult, BlogRequest, BriefResult


RANK_OVERVIEW_FIELDS = [
    "Weapon",
    "Weapon Kamae",
    "Weapon Strikes",
    "Cuts",
    "Draws",
    "Evasions",
    "Weapon Spinning",
    "Kamae",
    "Ukemi",
    "Kaiten",
    "Taihenjutsu",
    "Blocking",
    "Striking",
    "Grappling and escapes",
    "Kihon Happo",
    "San Shin no Kata",
    "Nage waza",
    "Jime waza",
    "Kyusho",
    "Other",
]

_RANK_MENTION_RE = re.compile(
    r"\b(?P<num>\d+)\s*(?:st|nd|rd|th)?\s+kyu\b|\b(?P<dan>shodan)\b",
    re.IGNORECASE,
)

_FIELD_RE = re.compile(
    rf"^(?P<label>{'|'.join(re.escape(label) for label in RANK_OVERVIEW_FIELDS)}):\s*(?P<value>.*)$",
    re.IGNORECASE,
)
_GLOSSARY_LINE_RE = re.compile(
    r"^(?P<term>[^:-][^:-]{1,90}?)\s*(?:-|:)\s*(?P<definition>.+)$"
)
_CONTINUATION_FIELDS = {
    "Weapon Kamae",
    "Weapon Strikes",
    "Cuts",
    "Draws",
    "Evasions",
    "Weapon Spinning",
    "Kamae",
    "Ukemi",
    "Kaiten",
    "Taihenjutsu",
    "Blocking",
    "Striking",
    "Grappling and escapes",
    "Kihon Happo",
    "San Shin no Kata",
    "Nage waza",
    "Jime waza",
    "Kyusho",
    "Other",
}
_GEAR_ALIASES = {
    "rokushakubo": {"rokushakubo", "rokushaku bo", "rokushaku-bo"},
    "hanbo": {"hanbo", "short staff", "three-foot staff", "3-foot staff"},
    "katana": {"katana", "daito", "long sword"},
    "knife": {"knife", "tanto", "dagger"},
    "shoto": {"shoto", "short sword"},
    "kusari fundo": {"kusari fundo", "manriki-gusari", "weighted chain"},
    "jutte": {"jutte"},
    "tessen": {"tessen", "iron fan"},
    "kunai": {"kunai"},
    "shuriken": {"shuriken", "bo shuriken", "senban shuriken", "throwing knives"},
    "shuko": {"shuko", "hand claws"},
    "naginata": {"naginata"},
    "kyoketsu shoge": {"kyoketsu shoge", "kyoketsu-shoge"},
    "firearms": {"firearms"},
}


def detect_rank_overview_request(text: str) -> str | None:
    """Return a normalized rank key when the request asks for a single-rank overview."""
    rank_key = _rank_key_from_question(text)
    if not rank_key:
        return None

    low = _norm(text).lower()
    overview_intent = (
        "overview" in low
        or "skills" in low
        or "what do you learn" in low
        or "what do i learn" in low
        or "what do we learn" in low
        or "what is learned" in low
        or "what do you need to know" in low
        or "what do i need to know" in low
    )
    specific_only = any(
        term in low
        for term in [
            "weapon",
            "weapons",
            "kata",
            "sanshin",
            "san shin",
            "kihon happo",
            "ukemi",
            "taihenjutsu",
            "throw",
            "throws",
            "nage",
            "strike",
            "striking",
            "kick",
            "kicks",
        ]
    )
    if overview_intent and not specific_only:
        return rank_key
    if "overview" in low:
        return rank_key
    return None


def _ordinal_kyu_rank(value: str) -> str:
    if value == "1":
        return "1st kyu"
    if value == "2":
        return "2nd kyu"
    if value == "3":
        return "3rd kyu"
    return f"{value}th kyu"


def _rank_mentions(text: str) -> list[str]:
    ranks: list[str] = []
    seen: set[str] = set()
    for match in _RANK_MENTION_RE.finditer(text or ""):
        rank = "shodan" if match.group("dan") else _ordinal_kyu_rank(match.group("num"))
        if rank not in seen:
            ranks.append(rank)
            seen.add(rank)
    return ranks


def _asks_for_rank_comparison(text: str) -> bool:
    low = _norm(text).lower()
    return any(
        phrase in low
        for phrase in [
            "compare",
            "comparison",
            "versus",
            " vs ",
            "difference between",
            "differences between",
            "between",
            "all ranks",
            "across ranks",
            "neighboring ranks",
            "previous ranks",
            "next ranks",
            "up to",
            "up through",
            "by the time",
        ]
    ) or re.search(r"\bby\s+\d+(?:st|nd|rd|th)?\s+kyu\b", low) is not None


def detect_rank_scoped_request(text: str) -> str | None:
    """Return a rank key when anchors should be limited to one named rank."""
    ranks = _rank_mentions(text)
    if len(ranks) != 1:
        return None
    if _asks_for_rank_comparison(text):
        return None
    return ranks[0]


def _request_text(request: BlogRequest) -> str:
    parts = [request.hook_title]
    if request.premise:
        parts.append(request.premise)
    parts.extend(request.include_terms)
    return " ".join(part for part in parts if part)


def _contains_term(text: str, term: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9-]){re.escape(term.lower())}(?![a-z0-9-])",
        text.lower(),
    ) is not None


def _split_gear_items(value: str) -> list[str]:
    return [
        _norm(item).lower()
        for item in re.split(r"[;,]", value or "")
        if _norm(item)
    ]


def _canonical_gear_for_text(text: str) -> set[str]:
    found: set[str] = set()
    for canonical, aliases in _GEAR_ALIASES.items():
        if any(_contains_term(text, alias) for alias in aliases):
            found.add(canonical)
    return found


def _allowed_gear_from_grounding(text: str) -> set[str]:
    allowed: set[str] = set()
    for line in (text or "").splitlines():
        match = re.match(r"^\s*-?\s*Weapon:\s*(?P<value>.+)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        for item in _split_gear_items(match.group("value")):
            allowed.update(_canonical_gear_for_text(item))
    return allowed


def validate_rank_overview_grounding(
    request: BlogRequest,
    anchors: AnchorResult,
    brief: BriefResult,
    *,
    rank_key: str,
) -> dict[str, Any]:
    """Validate that rank-overview grounding did not drift into adjacent ranks."""
    inspected = "\n".join(
        part
        for part in [anchors.anchor_block, brief.brief_markdown]
        if part
    )
    request_context = _request_text(request)
    allow_rank_neighbors = _asks_for_rank_comparison(request_context)
    ranks = _rank_mentions(inspected)
    unrelated_ranks = [
        rank for rank in ranks if rank != rank_key and not allow_rank_neighbors
    ]

    mentioned_gear = _canonical_gear_for_text(inspected)
    allowed_gear = _allowed_gear_from_grounding(inspected)
    allowed_gear.update(_canonical_gear_for_text(request_context))
    unrelated_gear = sorted(mentioned_gear - allowed_gear)

    warnings: list[str] = []
    if unrelated_ranks:
        warnings.append(
            "Unexpected rank references: " + ", ".join(unrelated_ranks)
        )
    if unrelated_gear:
        warnings.append(
            "Unexpected weapon/gear references: " + ", ".join(unrelated_gear)
        )

    return {
        "ok": not warnings,
        "rank": rank_key,
        "checked": True,
        "allowed_rank_neighbors": allow_rank_neighbors,
        "unrelated_ranks": unrelated_ranks,
        "unrelated_gear": unrelated_gear,
        "warnings": warnings,
    }


def rank_scoped_passages(
    passages: list[dict[str, Any]],
    *,
    rank_key: str,
) -> list[dict[str, Any]]:
    """Return a synthetic passage containing only the requested rank block."""
    rank_match = _rank_block_from_passages(passages, rank_key)
    if not rank_match:
        return []

    block, source_name = rank_match
    return [
        {
            "text": block,
            "source": source_name,
            "meta": {
                "source": source_name,
                "rank": rank_key,
                "rank_scoped": True,
            },
        }
    ]


def rank_overview_retrieval_query(request: BlogRequest, rank_key: str) -> str:
    parts = [rank_key, "nttv rank requirements", "rank overview", "skills learned"]
    if request.premise:
        parts.append(request.premise)
    return " | ".join(parts)


def _title_rank(rank_key: str) -> str:
    match = re.match(r"(\d+)(st|nd|rd|th)\s+kyu", rank_key, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}{match.group(2).lower()} Kyu"
    return "Shodan" if rank_key.lower() == "shodan" else rank_key.title()


def _rank_source_name(passages: list[dict[str, Any]]) -> str:
    for passage in passages:
        source = passage.get("source") or (passage.get("meta") or {}).get("source") or ""
        if "nttv rank requirements" in source.lower():
            return os.path.basename(source)
    return "nttv rank requirements.txt"


def _rank_block_from_passages(
    passages: list[dict[str, Any]],
    rank_key: str,
) -> tuple[str, str] | None:
    for passage in passages:
        text = passage.get("text") or ""
        if not text:
            continue

        block = _extract_rank_block(text, rank_key)
        if not block:
            continue

        source = passage.get("source") or (passage.get("meta") or {}).get("source") or ""
        if "nttv rank requirements" in source.lower():
            return block, os.path.basename(source) or "nttv rank requirements.txt"

    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return None

    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return None
    return block, _rank_source_name(passages)


def _field_map(block: str) -> dict[str, str]:
    canonical = {label.lower(): label for label in RANK_OVERVIEW_FIELDS}
    fields: dict[str, list[str]] = {label: [] for label in RANK_OVERVIEW_FIELDS}
    current: str | None = None

    for raw in (block or "").splitlines()[1:]:
        line = raw.strip()
        if not line:
            current = None
            continue

        match = _FIELD_RE.match(line)
        if match:
            label = canonical[match.group("label").lower()]
            current = label
            value = _norm(match.group("value"))
            if value:
                fields[label].append(value)
            continue

        if current in _CONTINUATION_FIELDS:
            fields[current].append(_norm(line))

    return {
        label: _norm(" ".join(values))
        for label, values in fields.items()
        if _norm(" ".join(values))
    }


def _clip_text(text: str, max_chars: int) -> str:
    text = _norm(text)
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _line_fits(lines: list[str], line: str, max_chars: int) -> bool:
    projected = len("\n".join([*lines, line]))
    return projected <= max_chars


def _add_bounded_line(lines: list[str], line: str, max_chars: int) -> None:
    line = _norm(line) if not line.startswith("###") else line.strip()
    if not line:
        return
    if _line_fits(lines, line, max_chars):
        lines.append(line)


def _term_in_rank_block(term: str, rank_block: str) -> bool:
    term = _norm(term).lower()
    block = _norm(rank_block).lower()
    if not term:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", block) is not None


def _glossary_source_name(passage: dict[str, Any]) -> str:
    source = passage.get("source") or (passage.get("meta") or {}).get("source") or ""
    return os.path.basename(source) or "Glossary"


def _supporting_glossary_definitions(
    passages: list[dict[str, Any]],
    rank_block: str,
    *,
    max_items: int = 6,
) -> tuple[list[str], list[str]]:
    definitions: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()

    for passage in passages:
        source = passage.get("source") or (passage.get("meta") or {}).get("source") or ""
        if "glossary" not in source.lower():
            continue

        source_name = _glossary_source_name(passage)
        for raw in (passage.get("text") or "").splitlines():
            line = _norm(raw).replace("\u2013", "-").replace("\u2014", "-")
            if not line or line.lower() == "glossary":
                continue

            match = _GLOSSARY_LINE_RE.match(line)
            if not match:
                continue

            term = _norm(match.group("term"))
            definition = _norm(match.group("definition"))
            if not term or not definition or not _term_in_rank_block(term, rank_block):
                continue

            key = term.lower()
            if key in seen:
                continue

            definitions.append(f"- {term}: {_clip_text(definition, 150)}")
            seen.add(key)
            if source_name not in sources:
                sources.append(source_name)
            if len(definitions) >= max_items:
                return definitions, sources

    return definitions, sources


def build_rank_overview_context(
    request: BlogRequest,
    passages: list[dict[str, Any]],
    *,
    rank_key: str,
    max_chars: int = 5000,
) -> tuple[BriefResult, AnchorResult] | None:
    """Build a single-rank fact sheet from nttv rank requirements passages."""
    rank_match = _rank_block_from_passages(passages, rank_key)
    if not rank_match:
        return None

    block, source_name = rank_match
    fields = _field_map(block)
    if not fields:
        return None

    title_rank = _title_rank(rank_key)
    glossary_definitions, glossary_sources = _supporting_glossary_definitions(passages, block)
    det_answer = try_answer_rank_requirements(
        f"What are the rank requirements for {rank_key}?",
        [{"text": block, "source": source_name, "meta": {"source": source_name}}],
    )

    lines = [
        "### Blog Brief",
        "### Rank Overview Fact Sheet",
        "### Exact Rank Requirements",
        f"- Hook: {request.hook_title}",
        f"- Rank: {title_rank}",
        f"- Source: {source_name}",
        "- Scope: this fact sheet is limited to this rank block only.",
    ]
    for label in RANK_OVERVIEW_FIELDS:
        value = fields.get(label)
        if value:
            lines.append(f"- {label}: {_clip_text(value, 320)}")

    lines.append("### Optional Supporting Definitions")
    if glossary_definitions:
        lines.extend(glossary_definitions)
    else:
        lines.append("- None from retrieved glossary passages.")

    sources_used = [source_name, *glossary_sources]
    lines.append("### Sources Used")
    for source in sources_used:
        lines.append(f"- {source}")

    bounded_lines: list[str] = []
    for line in lines:
        _add_bounded_line(bounded_lines, line, max_chars)

    fact_sheet = "\n".join(bounded_lines)

    anchors = [line for line in fact_sheet.splitlines()[1:] if line.startswith("- ")]
    metadata = {
        "rank_overview": True,
        "rank_brief": True,
        "rank": rank_key,
        "source": source_name,
        "field_count": len(fields),
        "glossary_definition_count": len(glossary_definitions),
        "deterministic_answer": det_answer,
        "char_count": len(fact_sheet),
    }

    return (
        BriefResult(
            title=f"{title_rank} overview",
            sections=anchors,
            brief_markdown=fact_sheet,
            sources_used=sources_used,
            metadata=metadata,
        ),
        AnchorResult(
            anchor_block=fact_sheet,
            anchors=anchors,
            metadata=metadata,
        ),
    )
