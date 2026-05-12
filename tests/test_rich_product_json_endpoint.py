# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


def test_vlm_rich_product_endpoint_success(sample_image_bytes):
    expected = {
        "product_type": "handbag",
        "visual_attributes": {
            "colors": ["black", "gold"],
            "hardware": "gold-tone clasp",
        },
        "visible_text": [],
    }

    with patch("backend.main.extract_rich_product_json", return_value=expected) as mock_extract:
        client = TestClient(app)
        response = client.post(
            "/vlm/rich-product",
            data={"locale": "en-US"},
            files={"image": ("product.png", sample_image_bytes, "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == expected
    mock_extract.assert_called_once()
    assert mock_extract.call_args.args[1] == "image/png"
    assert mock_extract.call_args.args[2] == "en-US"


def test_vlm_rich_product_endpoint_validation_errors(sample_image_bytes):
    client = TestClient(app)

    bad_locale = client.post(
        "/vlm/rich-product",
        data={"locale": "de-DE"},
        files={"image": ("product.png", sample_image_bytes, "image/png")},
    )
    assert bad_locale.status_code == 400

    non_image = client.post(
        "/vlm/rich-product",
        data={"locale": "en-US"},
        files={"image": ("product.txt", b"not an image", "text/plain")},
    )
    assert non_image.status_code == 400
    assert non_image.json()["detail"] == "File must be an image"


def test_vlm_rich_product_endpoint_preserves_unstructured_response(sample_image_bytes):
    fallback = {
        "parse_status": "unstructured",
        "warning": "VLM returned content that could not be parsed as a JSON object; raw response preserved.",
        "raw_response": "A product description that was not valid JSON.",
    }
    with patch("backend.main.extract_rich_product_json", return_value=fallback):
        client = TestClient(app)
        response = client.post(
            "/vlm/rich-product",
            data={"locale": "en-US"},
            files={"image": ("product.png", sample_image_bytes, "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == fallback
