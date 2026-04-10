"""Deterministic image prompt package composer for blog sections."""

from __future__ import annotations

from dataclasses import dataclass
import re

from scribe.models import DraftResult


DEFAULT_STYLE_PRESET = "editorial_ink_wash"
DEFAULT_POSITIVE_MAX_CHARS = 700
DEFAULT_NEGATIVE_MAX_CHARS = 280
DEFAULT_COMPOSITION_MAX_CHARS = 360
DEFAULT_FILENAME_MAX_CHARS = 96


STYLE_PRESETS = {
    "editorial_ink_wash": {
        "positive": (
            "editorial ink-wash illustration, textured paper, restrained earth tones, "
            "quiet discipline, clear silhouette, modern blog header art"
        ),
        "negative": (
            "photorealistic gore, injury detail, sensational violence, text overlay, "
            "watermark, logo, distorted hands, extra limbs, cluttered background"
        ),
        "composition": (
            "Use one strong focal figure or object, generous negative space for a blog crop, "
            "and a calm directional line that supports the section's central idea."
        ),
    },
    "dojo_diagram": {
        "positive": (
            "clean dojo training diagram, simple linework, parchment background, "
            "measured instructional tone, readable spatial relationships"
        ),
        "negative": (
            "photorealistic violence, blood, comic-book exaggeration, text labels, "
            "watermark, logo, crowded scene, distorted anatomy"
        ),
        "composition": (
            "Frame the action like a teaching plate with two or three clear spatial beats "
            "and enough margin for captions outside the generated image."
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ImagePromptPackage:
    positive_prompt: str
    negative_prompt: str
    composition_notes: str
    style_preset_name: str
    filename_suggestion: str


def _clean(text: str) -> str:
    return " ".join((text or "").strip().split())


def _clip(text: str, max_chars: int) -> str:
    text = _clean(text)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _draft_title(draft: DraftResult | str) -> str:
    if isinstance(draft, DraftResult):
        return draft.title
    return "blog section"


def _draft_body(draft: DraftResult | str) -> str:
    if isinstance(draft, DraftResult):
        return draft.body
    return str(draft)


def _section_heading(section_text: str) -> str:
    for raw in (section_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.strip("#").strip()
        return line
    return "section"


def _slugify(text: str, max_chars: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _clean(text).lower()).strip("-")
    if not slug:
        slug = "blog-image"
    if len(slug) > max_chars:
        slug = slug[:max_chars].rstrip("-")
    return slug or "blog-image"


def _style(name: str) -> dict[str, str]:
    if name not in STYLE_PRESETS:
        valid = ", ".join(sorted(STYLE_PRESETS))
        raise ValueError(f"style_preset_name must be one of: {valid}")
    return STYLE_PRESETS[name]


def build_image_prompt_package(
    draft: DraftResult | str,
    section_text: str,
    *,
    style_preset_name: str = DEFAULT_STYLE_PRESET,
    positive_max_chars: int = DEFAULT_POSITIVE_MAX_CHARS,
    negative_max_chars: int = DEFAULT_NEGATIVE_MAX_CHARS,
    composition_max_chars: int = DEFAULT_COMPOSITION_MAX_CHARS,
    filename_max_chars: int = DEFAULT_FILENAME_MAX_CHARS,
) -> ImagePromptPackage:
    """Create a backend-neutral image prompt package for one selected section."""
    if not _clean(section_text):
        raise ValueError("section_text is required")

    preset = _style(style_preset_name)
    title = _clip(_draft_title(draft), 120)
    heading = _clip(_section_heading(section_text), 100)
    section_focus = _clip(section_text, 260)
    draft_context = _clip(_draft_body(draft), 180)

    positive = (
        f"{preset['positive']}. "
        f"Blog title: {title}. "
        f"Section focus: {heading}. "
        f"Visual subject: {section_focus}. "
        f"Context cue: {draft_context}."
    )
    composition = (
        f"{preset['composition']} "
        f"Prioritize the selected section over the full draft. "
        f"Suggested crop: wide blog feature image; subject: {heading}."
    )
    filename_base = _slugify(f"{title} {heading}", max(1, filename_max_chars - 4))

    return ImagePromptPackage(
        positive_prompt=_clip(positive, positive_max_chars),
        negative_prompt=_clip(preset["negative"], negative_max_chars),
        composition_notes=_clip(composition, composition_max_chars),
        style_preset_name=style_preset_name,
        filename_suggestion=f"{filename_base}.png",
    )


def render_image_prompt_package(package: ImagePromptPackage) -> str:
    """Render a prompt package in stable markdown for inspection or copy-out."""
    return "\n".join(
        [
            "### Image Prompt Package",
            f"Style preset: {package.style_preset_name}",
            "",
            "Positive prompt:",
            package.positive_prompt,
            "",
            "Negative prompt:",
            package.negative_prompt,
            "",
            "Composition notes:",
            package.composition_notes,
            "",
            "Filename suggestion:",
            package.filename_suggestion,
        ]
    )
