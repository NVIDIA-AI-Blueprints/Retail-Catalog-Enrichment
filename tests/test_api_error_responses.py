# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.web_insights import WebInsightsDependencyError


INTERNAL_ERROR_DETAIL = "An internal error occurred. Please try again later."
SENSITIVE_ERROR = "internal path: /srv/catalog/.env"


def assert_sanitized_error(response, status_code):
    assert response.status_code == status_code
    assert response.json() == {"detail": INTERNAL_ERROR_DETAIL}
    assert SENSITIVE_ERROR not in response.text


def test_vlm_analyze_does_not_expose_unexpected_exception(sample_image_bytes):
    with patch(
        "backend.main.extract_vlm_observation",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/vlm/analyze",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )

    assert_sanitized_error(response, 500)


def test_vlm_faqs_does_not_expose_unexpected_exception():
    with patch(
        "backend.main._call_nemotron_generate_faqs",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/vlm/faqs",
            data={"title": "Test product", "categories": "[]", "tags": "[]", "colors": "[]"},
        )

    assert_sanitized_error(response, 500)


def test_generate_3d_does_not_expose_unexpected_exception(sample_image_bytes):
    with patch(
        "backend.main.generate_3d_asset",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/generate/3d",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )

    assert_sanitized_error(response, 500)


@pytest.mark.parametrize(
    ("exception_type", "status_code"),
    [(ValueError, 502), (RuntimeError, 500)],
)
def test_vlm_rich_product_does_not_expose_exception(
    sample_image_bytes, exception_type, status_code
):
    with patch(
        "backend.main.extract_rich_product_json",
        side_effect=exception_type(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/vlm/rich-product",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )

    assert_sanitized_error(response, status_code)


@pytest.mark.parametrize(
    ("exception_type", "status_code"),
    [(ValueError, 400), (RuntimeError, 500)],
)
def test_vlm_manual_extract_does_not_expose_exception(exception_type, status_code):
    with patch(
        "backend.main.process_manual_pdf",
        side_effect=exception_type(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/vlm/manual/extract",
            files={"file": ("manual.pdf", b"%PDF-test", "application/pdf")},
        )

    assert_sanitized_error(response, status_code)


@pytest.mark.parametrize(
    ("exception_type", "status_code"),
    [(WebInsightsDependencyError, 503), (RuntimeError, 500)],
)
def test_product_insights_does_not_expose_exception(exception_type, status_code):
    with patch(
        "backend.main.build_product_web_insights",
        side_effect=exception_type(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/research/product-insights",
            data={"title": "Test product"},
        )

    assert_sanitized_error(response, status_code)


def test_list_policies_does_not_expose_unexpected_exception():
    with patch(
        "backend.main.policy_library.list_documents",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).get("/policies")

    assert_sanitized_error(response, 500)


def test_upload_policies_does_not_expose_unexpected_exception():
    with patch(
        "backend.main.policy_library.ingest_documents",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/policies",
            files={"files": ("policy.pdf", b"%PDF-test", "application/pdf")},
        )

    assert_sanitized_error(response, 500)


def test_clear_policies_does_not_expose_unexpected_exception():
    with patch(
        "backend.main.policy_library.clear",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).delete("/policies")

    assert_sanitized_error(response, 500)


def test_generate_variation_does_not_expose_unexpected_exception(sample_image_bytes):
    with patch(
        "backend.main.generate_image_variation",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/generate/variation",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
            data={
                "title": "Test product",
                "description": "Test description",
                "categories": "[]",
            },
        )

    assert_sanitized_error(response, 500)


def test_protocols_generate_does_not_expose_unexpected_exception():
    with patch(
        "backend.main._call_nemotron_extract_schema_fields",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/protocols/generate",
            data={"title": "Test product"},
        )

    assert_sanitized_error(response, 500)
