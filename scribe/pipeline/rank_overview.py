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


def rank_scoped_passages(
    passages: list[dict[str, Any]],
    *,
    rank_key: str,
) -> list[dict[str, Any]]:
    """Return a synthetic passage containing only the requested rank block."""
    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return []

    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return []

    source_name = _rank_source_name(passages)
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

        if current:
            fields[current].append(_norm(line))

    return {
        label: _norm(" ".join(values))
        for label, values in fields.items()
        if _norm(" ".join(values))
    }


def build_rank_overview_context(
    request: BlogRequest,
    passages: list[dict[str, Any]],
    *,
    rank_key: str,
    max_chars: int = 5000,
) -> tuple[BriefResult, AnchorResult] | None:
    """Build a single-rank fact sheet from nttv rank requirements passages."""
    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return None

    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return None

    fields = _field_map(block)
    if not fields:
        return None

    source_name = _rank_source_name(passages)
    title_rank = _title_rank(rank_key)
    det_answer = try_answer_rank_requirements(
        f"What are the rank requirements for {rank_key}?",
        [{"text": block, "source": source_name, "meta": {"source": source_name}}],
    )

    lines = [
        "### Rank Overview Fact Sheet",
        f"- Rank: {title_rank}",
        f"- Source: {source_name}",
        "- Scope: this fact sheet is limited to this rank block only.",
    ]
    for label in RANK_OVERVIEW_FIELDS:
        value = fields.get(label)
        if value:
            lines.append(f"- {label}: {value}")

    fact_sheet = "\n".join(lines)
    if len(fact_sheet) > max_chars:
        fact_sheet = fact_sheet[: max_chars - 3].rstrip() + "..."

    anchors = [line for line in fact_sheet.splitlines()[1:] if line.startswith("- ")]
    metadata = {
        "rank_overview": True,
        "rank": rank_key,
        "source": source_name,
        "field_count": len(fields),
        "deterministic_answer": det_answer,
        "char_count": len(fact_sheet),
    }

    return (
        BriefResult(
            title=f"{title_rank} overview",
            sections=anchors,
            brief_markdown=fact_sheet,
            sources_used=[source_name],
            metadata=metadata,
        ),
        AnchorResult(
            anchor_block=fact_sheet,
            anchors=anchors,
            metadata=metadata,
        ),
    )
