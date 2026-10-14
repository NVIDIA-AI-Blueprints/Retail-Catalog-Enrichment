from __future__ import annotations

import copy

import pytest

from catalog_bench.engines.mock import SAMPLE_ENRICHMENT
from catalog_bench.prompts import DEFAULT_SCHEMA, load_schema


@pytest.fixture(scope="session")
def schema() -> dict:
    return load_schema(DEFAULT_SCHEMA)


@pytest.fixture
def instance() -> dict:
    """A fresh deep copy per test, so mutations do not leak between them."""
    return copy.deepcopy(SAMPLE_ENRICHMENT)
