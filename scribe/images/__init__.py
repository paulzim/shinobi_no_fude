"""Image scaffolding for the separate scribe blog workflow."""

from .prompt_composer import (
    ImagePromptPackage,
    build_image_prompt_package,
    render_image_prompt_package,
)

__all__ = [
    "ImagePromptPackage",
    "build_image_prompt_package",
    "render_image_prompt_package",
]
