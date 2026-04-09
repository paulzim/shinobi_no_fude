"""Pure helpers for chunk enrichment and prompt grounding."""

import csv
import os
import re
from typing import Any, Dict, Iterable, List, Tuple


RANK_HEADER_RE = re.compile(
    r"^(?:\d{1,2}(?:st|nd|rd|th)\s+kyu|shodan|nidan|sandan|yondan|godan|rokudan|nanadan|hachidan|kudan|judan)$",
    re.IGNORECASE,
)
TRAINING_RANK_RE = re.compile(r"^\d{1,2}(?:st|nd|rd|th)?\s*kyu$", re.IGNORECASE)

RANK_FIELDS = {
    "weapon",
    "weapon kamae",
    "weapon strikes",
    "cuts",
    "draws",
    "evasions",
    "weapon spinning",
    "kamae",
    "ukemi",
    "kaiten",
    "taihenjutsu",
    "blocking",
    "striking",
    "grappling and escapes",
    "kihon happo",
    "san shin no kata",
    "nage waza",
    "jime waza",
    "kyusho",
    "other",
}

SOURCE_KIND_LABELS = {
    "rank": "Rank",
    "weapons": "Weapon",
    "schools": "School",
    "techniques": "Technique",
    "leadership": "Leadership",
    "glossary": "Glossary",
    "training": "Training",
    "general": "Source",
}

GLOSSARY_SKIP_FIELDS = {
    "aliases",
    "type",
    "translation",
    "focus",
    "weapons",
    "key points",
    "notes",
    "ranks",
    "kamae",
    "core actions",
    "modes",
    "throws",
    "range",
    "safety/drill",
    "school",
    "weapon",
}


def infer_source_kind(source: str) -> str:
    """Map a source path to a coarse source kind."""
    source_low = os.path.basename(source or "").lower()
    if "rank requirements" in source_low:
        return "rank"
    if "weapons reference" in source_low:
        return "weapons"
    if "schools of the bujinkan summaries" in source_low:
        return "schools"
    if "technique descriptions" in source_low:
        return "techniques"
    if "leadership" in source_low:
        return "leadership"
    if "glossary" in source_low:
        return "glossary"
    if "training reference" in source_low:
        return "training"
    return "general"


def _clean_line(line: str) -> str:
    return " ".join((line or "").strip().split())


def _clip(text: str, max_len: int = 140) -> str:
    text = _clean_line(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _append_unique(items: List[str], seen: set[str], value: str) -> None:
    value = _clean_line(value)
    if not value:
        return
    folded = value.lower()
    if folded in seen:
        return
    items.append(value)
    seen.add(folded)


def _looks_like_rank_cell(value: str) -> bool:
    low = (value or "").strip().lower()
    return ("kyu" in low) or ("dan" in low) or ("ranked" in low)


def _normalize_rank_label(line: str) -> str:
    line = _clean_line(line)
    if "kyu" in line.lower():
        parts = line.split()
        if len(parts) == 2 and parts[1].lower() == "kyu":
            return f"{parts[0]} Kyu"
    return line.title()


def _extract_rank_extractions(text: str) -> Tuple[List[str], List[str]]:
    titles: List[str] = []
    title_seen: set[str] = set()
    anchors: List[str] = []
    anchor_seen: set[str] = set()

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        if RANK_HEADER_RE.match(line):
            _append_unique(titles, title_seen, _normalize_rank_label(line))
            continue
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        if label.strip().lower() not in RANK_FIELDS:
            continue
        value = _clean_line(value)
        if not value:
            continue
        _append_unique(anchors, anchor_seen, f"{label.strip()}: {_clip(value)}")

    return titles[:3], anchors[:6]


def _extract_weapon_extractions(text: str) -> Tuple[List[str], List[str]]:
    titles: List[str] = []
    title_seen: set[str] = set()
    anchors: List[str] = []
    anchor_seen: set[str] = set()
    prefixes = ("ALIASES:", "TYPE:", "RANKS:", "CORE ACTIONS:", "MODES:", "THROWS:", "KAMAE:")

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        if line.startswith("[WEAPON]"):
            name = _clean_line(line.replace("[WEAPON]", "", 1))
            _append_unique(titles, title_seen, name)
            _append_unique(anchors, anchor_seen, f"Weapon: {name}")
            continue
        if any(line.startswith(prefix) for prefix in prefixes):
            _append_unique(anchors, anchor_seen, _clip(line))

    return titles[:4], anchors[:6]


def _extract_school_extractions(text: str) -> Tuple[List[str], List[str]]:
    titles: List[str] = []
    title_seen: set[str] = set()
    anchors: List[str] = []
    anchor_seen: set[str] = set()
    prefixes = ("TRANSLATION:", "TYPE:", "FOCUS:", "WEAPONS:", "ALIASES:")

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        if line.startswith("SCHOOL:"):
            school = _clean_line(line.split(":", 1)[1])
            _append_unique(titles, title_seen, school)
            _append_unique(anchors, anchor_seen, f"School: {school}")
            continue
        if any(line.startswith(prefix) for prefix in prefixes):
            _append_unique(anchors, anchor_seen, _clip(line))

    return titles[:4], anchors[:6]


def _extract_technique_extractions(text: str) -> Tuple[List[str], List[str]]:
    titles: List[str] = []
    title_seen: set[str] = set()
    anchors: List[str] = []
    anchor_seen: set[str] = set()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(">") or "FORMAT:" in line:
            continue
        try:
            row = next(csv.reader([line], skipinitialspace=True))
        except Exception:
            continue
        if len(row) < 5:
            continue
        name = _clean_line(row[0])
        translation = _clean_line(row[2])
        kind = _clean_line(row[3])
        rank = _clean_line(row[4])
        if not name or len(name) > 80:
            continue
        if name.lower() in {"name", "csv schema"}:
            continue
        if not translation or not kind or not _looks_like_rank_cell(rank):
            continue
        _append_unique(titles, title_seen, name)
        _append_unique(
            anchors,
            anchor_seen,
            _clip(f"{name} | {translation} | {kind} | {rank}"),
        )

    return titles[:4], anchors[:5]


def _extract_leadership_extractions(text: str) -> Tuple[List[str], List[str]]:
    titles: List[str] = []
    title_seen: set[str] = set()
    anchors: List[str] = []
    anchor_seen: set[str] = set()

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        if line.startswith("[SOKESHIP]"):
            _append_unique(anchors, anchor_seen, "Section: Sokeship")
            continue
        if "Soke =" in line:
            _append_unique(anchors, anchor_seen, _clip(line))
            continue
        if "|" not in line:
            continue
        parts = [_clean_line(part) for part in line.split("|")]
        if len(parts) < 3 or not parts[0] or not parts[1]:
            continue
        _append_unique(titles, title_seen, parts[0])
        _append_unique(
            anchors,
            anchor_seen,
            _clip(f"Sokeship: {parts[0]} -> {parts[1]} ({parts[2]})"),
        )

    return titles[:4], anchors[:6]


def _extract_glossary_extractions(text: str) -> Tuple[List[str], List[str]]:
    titles: List[str] = []
    title_seen: set[str] = set()
    anchors: List[str] = []
    anchor_seen: set[str] = set()

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line or ":" not in line:
            continue
        term, value = line.split(":", 1)
        term = _clean_line(term)
        value = _clean_line(value)
        if not term or not value or len(term) > 50:
            continue
        if term.lower() in GLOSSARY_SKIP_FIELDS:
            continue
        _append_unique(titles, title_seen, term)
        _append_unique(anchors, anchor_seen, _clip(f"{term}: {value}"))

    return titles[:4], anchors[:5]


def _extract_training_extractions(text: str) -> Tuple[List[str], List[str]]:
    titles: List[str] = []
    title_seen: set[str] = set()
    anchors: List[str] = []
    anchor_seen: set[str] = set()

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        if TRAINING_RANK_RE.match(line):
            _append_unique(titles, title_seen, _normalize_rank_label(line))
            continue
        if re.match(r"^[A-Za-z][A-Za-z /\(\)]+-\s+[A-Za-z]", line):
            _append_unique(anchors, anchor_seen, _clip(line))

    return titles[:3], anchors[:6]


def extract_chunk_extractions(text: str, source: str) -> Dict[str, Any]:
    """Extract deterministic anchors from a chunk for reuse in prompts."""
    kind = infer_source_kind(source)
    extractor_map = {
        "rank": _extract_rank_extractions,
        "weapons": _extract_weapon_extractions,
        "schools": _extract_school_extractions,
        "techniques": _extract_technique_extractions,
        "leadership": _extract_leadership_extractions,
        "glossary": _extract_glossary_extractions,
        "training": _extract_training_extractions,
    }
    extractor = extractor_map.get(kind)
    if extractor is None:
        return {"source_kind": kind, "titles": [], "anchors": []}

    titles, anchors = extractor(text or "")
    return {
        "source_kind": kind,
        "titles": titles,
        "anchors": anchors,
    }


def _stored_extractions(meta: Dict[str, Any]) -> Dict[str, Any]:
    extractions = meta.get("extractions")
    if isinstance(extractions, dict):
        return extractions
    return {}


def get_passage_extractions(passage: Dict[str, Any]) -> Dict[str, Any]:
    """Return persisted extraction metadata or derive it on the fly."""
    meta = passage.get("meta") or {}
    stored = _stored_extractions(meta)
    if stored:
        return stored
    source = meta.get("source") or passage.get("source") or ""
    return extract_chunk_extractions(passage.get("text") or "", source)


def _iter_extraction_lines(passages: Iterable[Dict[str, Any]]) -> Iterable[str]:
    for passage in passages:
        extractions = get_passage_extractions(passage)
        label = SOURCE_KIND_LABELS.get(extractions.get("source_kind"), "Source")
        for title in extractions.get("titles") or []:
            yield f"[{label}] {title}"
        for anchor in extractions.get("anchors") or []:
            yield f"[{label}] {anchor}"


def build_extraction_context(
    passages: List[Dict[str, Any]],
    *,
    max_chars: int = 1400,
    max_items: int = 12,
) -> str:
    """Format extracted anchors into a compact prompt section."""
    lines: List[str] = []
    seen: set[str] = set()
    total = 0

    for line in _iter_extraction_lines(passages):
        folded = line.lower()
        if folded in seen:
            continue
        block = f"- {line}"
        if len(lines) >= max_items or total + len(block) + 1 > max_chars:
            break
        lines.append(block)
        seen.add(folded)
        total += len(block) + 1

    return "\n".join(lines)


def build_grounded_prompt(context: str, question: str, extraction_context: str = "") -> str:
    """Compose the final grounded prompt, optionally including extracted anchors."""
    parts = [
        "You must answer using ONLY the context below.",
        "Be concise but complete; avoid filler.",
    ]
    if extraction_context:
        parts.append(
            "Deterministic anchors derived from the retrieved sources:\n"
            f"{extraction_context}\n"
            "Treat them as shortcuts into the context, not as extra facts."
        )
    parts.append(f"Context:\n{context}")
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return "\n\n".join(parts)
