from __future__ import annotations

import pytest

from src.synthetic_data.captions import (
    BANNED_DIAGNOSTIC_TERMS,
    EXPECTED_CLASSES,
    build_caption,
    validate_caption_templates,
    validate_caption_text,
)


def test_build_caption_includes_trigger_and_visual_label() -> None:
    caption = build_caption("intact", trigger_word="soyseed")

    assert caption == (
        "photo of a soyseed soybean seed, intact condition, clean surface, "
        "centered inspection photography"
    )


def test_all_expected_classes_have_safe_captions() -> None:
    validate_caption_templates(EXPECTED_CLASSES, trigger_word="soyseed")

    for label in EXPECTED_CLASSES:
        caption = build_caption(label, trigger_word="soyseed")
        assert "soyseed" in caption
        assert not any(term in caption.lower() for term in BANNED_DIAGNOSTIC_TERMS)


def test_validate_caption_text_rejects_diagnostic_terms() -> None:
    with pytest.raises(ValueError, match="prohibited diagnostic"):
        validate_caption_text("photo of a soyseed soybean seed with fungus", trigger_word="soyseed")


def test_validate_caption_templates_requires_every_class() -> None:
    with pytest.raises(ValueError, match="Missing caption templates"):
        validate_caption_templates(["intact", "broken"], templates={"intact": "{trigger_word} seed"})
