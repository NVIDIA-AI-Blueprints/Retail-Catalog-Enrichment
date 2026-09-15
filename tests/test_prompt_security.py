# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.prompt_security import (
    MAX_BRAND_INSTRUCTIONS_CHARS,
    MAX_PRODUCT_DEPTH,
    MAX_PRODUCT_LIST_ITEMS,
    MAX_PRODUCT_TITLE_CHARS,
    PromptInputError,
    sanitize_brand_instructions,
    sanitize_product_data,
    untrusted_data_message,
)


def test_sanitize_product_data_preserves_supported_metadata_and_bounds_text():
    product_data = {
        "title": "t" * (MAX_PRODUCT_TITLE_CHARS + 25),
        "description": "Élégant catalog copy",
        "categories": ["bags"] * (MAX_PRODUCT_LIST_ITEMS + 5),
        "tags": ["travel"],
        "price": 15.99,
        "sku": "BAG-001",
        "specifications": {"material": "leather", "warranty_years": 2},
    }

    result = sanitize_product_data(product_data)

    assert len(result["title"]) == MAX_PRODUCT_TITLE_CHARS
    assert len(result["categories"]) == MAX_PRODUCT_LIST_ITEMS
    assert result["description"] == "Élégant catalog copy"
    assert result["price"] == 15.99
    assert result["sku"] == "BAG-001"
    assert result["specifications"] == {"material": "leather", "warranty_years": 2}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (["not", "an", "object"], "must be a JSON object"),
        ({"title": ["wrong"]}, "title must be a string"),
        ({"tags": "wrong"}, "tags must be an array of strings"),
        ({"price": "free"}, "price must be a number"),
    ],
)
def test_sanitize_product_data_rejects_unsupported_shapes(value, message):
    with pytest.raises(PromptInputError, match=message):
        sanitize_product_data(value)


def test_sanitize_product_data_rejects_excessive_nesting():
    nested = "value"
    for _ in range(MAX_PRODUCT_DEPTH + 1):
        nested = {"child": nested}

    with pytest.raises(PromptInputError, match="nesting depth"):
        sanitize_product_data({"metadata": nested})


def test_sanitize_product_data_rejects_non_finite_numbers():
    with pytest.raises(PromptInputError, match="non-finite"):
        sanitize_product_data({"price": float("nan")})


def test_sanitize_brand_instructions_removes_controls_and_caps_length():
    result = sanitize_brand_instructions("Playful\x00tone\r\n" + "x" * MAX_BRAND_INSTRUCTIONS_CHARS)

    assert result is not None
    assert "\x00" not in result
    assert "\r" not in result
    assert "\n" in result
    assert len(result) == MAX_BRAND_INSTRUCTIONS_CHARS


def test_untrusted_data_message_removes_surrogates_and_format_controls():
    message = untrusted_data_message({"text": "before\ud800\u202eafter"})

    assert json.loads(message)["untrusted_data"]["text"] == "beforeafter"


def test_vlm_analyze_rejects_non_object_product_data_before_model_calls(sample_image_bytes):
    with patch("backend.main.extract_vlm_observation") as mock_extract:
        response = TestClient(app).post(
            "/vlm/analyze",
            data={"locale": "en-US", "product_data": json.dumps(["not", "an", "object"])},
            files={"image": ("product.png", sample_image_bytes, "image/png")},
        )

    assert response.status_code == 400
    assert "must be a JSON object" in response.json()["detail"]
    mock_extract.assert_not_called()
