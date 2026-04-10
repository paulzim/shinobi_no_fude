"""Central settings for the separate blog-mode workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from scribe.models import BlogMode


TOKENS_TO_CHARS = 4


@dataclass(frozen=True, slots=True)
class BlogModeSettings:
    context_limit: int = 32000
    deep_context_limit: int = 64000
    deep_mode: bool = False
    verify_claims_enabled: bool = False
    verify_claims_max_chars: int = 600
    verify_claims_max_tokens: int = 180
    verify_claims_temperature: float = 0.1
    rag_top_k_retrieve: int = 18
    rag_top_k_keep: int = 8
    rag_budget_tokens: int = 8000
    max_output_tokens: dict[BlogMode, int] = field(
        default_factory=lambda: {
            BlogMode.HOOK_EXPANSION: 900,
            BlogMode.OUTLINE: 900,
            BlogMode.DRAFT: 2500,
            BlogMode.POLISH: 2000,
            BlogMode.REWRITE: 2000,
        }
    )
    temperatures: dict[BlogMode, float] = field(
        default_factory=lambda: {
            BlogMode.HOOK_EXPANSION: 0.9,
            BlogMode.OUTLINE: 0.9,
            BlogMode.DRAFT: 0.8,
            BlogMode.POLISH: 0.4,
            BlogMode.REWRITE: 0.4,
        }
    )

    @property
    def active_context_limit(self) -> int:
        return self.deep_context_limit if self.deep_mode else self.context_limit

    @property
    def rag_budget_chars(self) -> int:
        return self.rag_budget_tokens * TOKENS_TO_CHARS

    def prompt_char_limit(self, requested: int | None = None) -> int:
        if requested is None:
            return self.active_context_limit
        return min(requested, self.active_context_limit)

    def brief_char_limit(self, requested: int | None = None) -> int:
        if requested is None:
            return self.rag_budget_chars
        return min(requested, self.rag_budget_chars)

    def max_tokens_for(self, mode: BlogMode) -> int:
        return self.max_output_tokens[mode]

    def temperature_for(self, mode: BlogMode) -> float:
        return self.temperatures[mode]


DEFAULT_BLOG_MODE_SETTINGS = BlogModeSettings()
