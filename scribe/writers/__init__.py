"""Writer scaffolding for the separate scribe blog workflow."""

from .prompt_builder import build_writer_prompt
from .rewrite_commands import (
    RewriteCommand,
    extract_headings,
    normalize_rewrite_instruction,
    parse_rewrite_command,
)

__all__ = [
    "RewriteCommand",
    "build_writer_prompt",
    "extract_headings",
    "normalize_rewrite_instruction",
    "parse_rewrite_command",
]
