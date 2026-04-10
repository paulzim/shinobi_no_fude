"""Typed data contracts for the separate scribe blog workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class CreativityLevel(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


class BlogMode(str, Enum):
    HOOK_EXPANSION = "hook_expansion"
    OUTLINE = "outline"
    DRAFT = "draft"
    POLISH = "polish"
    REWRITE = "rewrite"


def _as_enum(value: str | Enum, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value

    try:
        return enum_type(value)
    except ValueError as exc:  # pragma: no cover - exercised by tests via caller
        valid = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {valid}") from exc


def _as_str_list(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    return [str(value) for value in values]


@dataclass(slots=True)
class BlogRequest:
    hook_title: str
    premise: str | None = None
    length_target_words: int = 2000
    creativity_level: CreativityLevel | str = CreativityLevel.MED
    include_terms: list[str] = field(default_factory=list)
    avoid_terms: list[str] = field(default_factory=list)
    mode: BlogMode | str = BlogMode.DRAFT

    def __post_init__(self) -> None:
        self.hook_title = self.hook_title.strip()
        if not self.hook_title:
            raise ValueError("hook_title is required")

        if self.premise is not None:
            self.premise = self.premise.strip() or None

        if self.length_target_words <= 0:
            raise ValueError("length_target_words must be positive")

        self.creativity_level = _as_enum(
            self.creativity_level,
            CreativityLevel,
            "creativity_level",
        )
        self.mode = _as_enum(self.mode, BlogMode, "mode")
        self.include_terms = _as_str_list(self.include_terms)
        self.avoid_terms = _as_str_list(self.avoid_terms)


@dataclass(slots=True)
class AnchorResult:
    anchors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BriefResult:
    title: str
    sections: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DraftResult:
    title: str
    body: str

