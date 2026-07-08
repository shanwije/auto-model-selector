"""Tests for :class:`RouterConfig`."""

from __future__ import annotations

import pytest

from auto_model_selector.config import RouterConfig


def test_defaults_are_coherent() -> None:
    config = RouterConfig()
    assert config.short_prompt_words < config.long_prompt_words
    assert 0.0 <= config.confidence_threshold <= 1.0


def test_rejects_overlapping_word_thresholds() -> None:
    with pytest.raises(ValueError, match="short_prompt_words"):
        RouterConfig(short_prompt_words=90, long_prompt_words=80)


def test_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        RouterConfig(confidence_threshold=1.5)


def test_with_overrides_is_non_mutating() -> None:
    base = RouterConfig()
    tuned = base.with_overrides(long_prompt_words=120)
    assert tuned.long_prompt_words == 120
    assert base.long_prompt_words == 80
