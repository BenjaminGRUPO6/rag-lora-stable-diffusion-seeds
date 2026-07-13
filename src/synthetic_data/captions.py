from __future__ import annotations

from collections.abc import Mapping, Sequence


DEFAULT_TRIGGER_WORD = "soyseed"
EXPECTED_CLASSES: tuple[str, ...] = (
    "intact",
    "spotted",
    "immature",
    "broken",
    "skin_damaged",
)
BANNED_DIAGNOSTIC_TERMS: tuple[str, ...] = ("fungus", "disease", "infection")

CAPTION_TEMPLATES: dict[str, str] = {
    "intact": (
        "photo of a {trigger_word} soybean seed, intact condition, clean surface, "
        "centered inspection photography"
    ),
    "spotted": (
        "photo of a {trigger_word} soybean seed, spotted visual surface pattern, "
        "centered inspection photography"
    ),
    "immature": (
        "photo of a {trigger_word} soybean seed, immature condition, pale surface, "
        "centered inspection photography"
    ),
    "broken": (
        "photo of a {trigger_word} soybean seed, broken condition, visible fracture, "
        "centered inspection photography"
    ),
    "skin_damaged": (
        "photo of a {trigger_word} soybean seed, skin_damaged condition, damaged seed coat, "
        "centered inspection photography"
    ),
}


def validate_caption_templates(
    classes: Sequence[str],
    templates: Mapping[str, str] | None = None,
    trigger_word: str = DEFAULT_TRIGGER_WORD,
) -> None:
    """Validate that every class has a safe caption template."""

    caption_templates = templates or CAPTION_TEMPLATES
    missing = [label for label in classes if label not in caption_templates]
    if missing:
        raise ValueError(f"Missing caption templates for classes: {missing}")

    for label in classes:
        caption = build_caption(label, trigger_word=trigger_word, templates=caption_templates)
        validate_caption_text(caption, trigger_word=trigger_word)


def build_caption(
    label: str,
    trigger_word: str = DEFAULT_TRIGGER_WORD,
    templates: Mapping[str, str] | None = None,
) -> str:
    """Build a consistent LoRA caption for a class label."""

    caption_templates = templates or CAPTION_TEMPLATES
    if label not in caption_templates:
        raise ValueError(f"Unsupported class label for LoRA caption: {label}")
    caption = caption_templates[label].format(trigger_word=trigger_word)
    validate_caption_text(caption, trigger_word=trigger_word)
    return caption


def validate_caption_text(caption: str, trigger_word: str = DEFAULT_TRIGGER_WORD) -> None:
    """Reject empty captions, missing trigger words, and diagnostic claims."""

    if not caption.strip():
        raise ValueError("Caption cannot be empty.")
    if trigger_word not in caption:
        raise ValueError(f"Caption must include trigger word: {trigger_word}")

    caption_lower = caption.lower()
    banned = [term for term in BANNED_DIAGNOSTIC_TERMS if term in caption_lower]
    if banned:
        raise ValueError(f"Caption contains prohibited diagnostic terms: {banned}")
