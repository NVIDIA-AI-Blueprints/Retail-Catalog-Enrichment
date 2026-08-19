"""Prompt rendering.

The system prefix being byte-identical across products is what makes prefix caching
possible at all, and it is the kind of thing that breaks silently.
"""

from __future__ import annotations

import pytest

from catalog_bench.prompts import SCHEMA_MODES, render_system, render_user

PRODUCT = {"product_id": "SKU-1", "title": "Test Item", "description": "d", "metadata": ""}


def test_system_prefix_is_invariant_across_renders(schema):
    assert render_system(schema, "compact") == render_system(schema, "compact")


def test_system_prefix_is_invariant_across_products(schema):
    """Nothing product-specific may leak into the shared prefix."""
    prefix = render_system(schema, "compact")
    for pid in ("SKU-1", "SKU-99999"):
        render_user({**PRODUCT, "product_id": pid})
        assert render_system(schema, "compact") == prefix


@pytest.mark.parametrize("mode", SCHEMA_MODES)
def test_all_schema_modes_render(schema, mode):
    assert render_system(schema, mode)


def test_compact_is_cheaper_than_full(schema):
    assert len(render_system(schema, "compact")) < len(render_system(schema, "full"))


def test_none_is_cheapest(schema):
    assert len(render_system(schema, "none")) < len(render_system(schema, "compact"))


def test_unknown_schema_mode_is_rejected(schema):
    with pytest.raises(ValueError, match="unknown schema_mode"):
        render_system(schema, "verbose")


def test_user_suffix_carries_the_product_id():
    assert "SKU-1" in render_user(PRODUCT)


def test_missing_product_field_fails_loudly():
    """StrictUndefined, so a silently-empty prompt field cannot corrupt input length."""
    from jinja2 import UndefinedError

    with pytest.raises(UndefinedError):
        render_user({"product_id": "SKU-1"})
