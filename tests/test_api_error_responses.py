# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


INTERNAL_ERROR_DETAIL = "An internal error occurred. Please try again later."
SENSITIVE_ERROR = "internal path: /srv/catalog/.env"


def test_vlm_analyze_does_not_expose_unexpected_exception(sample_image_bytes):
    with patch(
        "backend.main.extract_vlm_observation",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/vlm/analyze",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR_DETAIL}
    assert SENSITIVE_ERROR not in response.text


def test_vlm_faqs_does_not_expose_unexpected_exception():
    with patch(
        "backend.main._call_nemotron_generate_faqs",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/vlm/faqs",
            data={"title": "Test product", "categories": "[]", "tags": "[]", "colors": "[]"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR_DETAIL}
    assert SENSITIVE_ERROR not in response.text


def test_generate_3d_does_not_expose_unexpected_exception(sample_image_bytes):
    with patch(
        "backend.main.generate_3d_asset",
        side_effect=RuntimeError(SENSITIVE_ERROR),
    ):
        response = TestClient(app).post(
            "/generate/3d",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR_DETAIL}
    assert SENSITIVE_ERROR not in response.text
