"""Pipeline scaffolding for the separate scribe blog workflow."""

from .orchestrator import (
    DraftPipelineResult,
    HookBuildResult,
    build_around_hook,
    draft_from_outline,
    polish_draft,
    rewrite_with_instruction,
)

__all__ = [
    "DraftPipelineResult",
    "HookBuildResult",
    "build_around_hook",
    "draft_from_outline",
    "polish_draft",
    "rewrite_with_instruction",
]
