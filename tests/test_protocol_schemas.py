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

"""
Unit tests for ACP and UCP protocol schema builders.
"""
import pytest
from backend.main import _build_acp_schema, _build_ucp_schema


@pytest.fixture
def sample_extracted_fields():
    """Sample LLM-extracted fields for protocol schemas."""
    return {
        "brand": "Elegance Co",
        "condition": "new",
        "material": "leather",
        "age_group": "adult",
        "gender": "female",
        "short_title": "Elegant Black Handbag",
        "google_product_category": "Apparel & Accessories > Handbags",
        "product_details": [
            {"attribute_name": "Closure Type", "attribute_value": "Zipper"},
            {"attribute_name": "Hardware", "attribute_value": "Gold-tone"},
        ],
        "product_highlights": [
            "Premium leather construction",
            "Gold-tone hardware accents",
            "Perfect for evening occasions",
        ],
    }


class TestBuildAcpSchema:
    """Tests for the ACP schema builder."""

    def test_populates_enriched_fields(self, sample_vlm_response, sample_faqs_response, sample_extracted_fields):
        schema = _build_acp_schema(sample_vlm_response, sample_faqs_response, sample_extracted_fields)

        assert schema["product"]["title"] == sample_vlm_response["title"]
        assert schema["product"]["description"] == sample_vlm_response["description"]
        assert schema["product"]["attributes"]["colors"] == sample_vlm_response["colors"]
        assert schema["product"]["categories"] == sample_vlm_response["categories"]
        assert schema["product"]["tags"] == sample_vlm_response["tags"]

    def test_merges_extracted_fields(self, sample_vlm_response, sample_extracted_fields):
        schema = _build_acp_schema(sample_vlm_response, [], sample_extracted_fields)

        assert schema["product"]["brand"] == "Elegance Co"
        assert schema["product"]["attributes"]["material"] == "leather"
        assert schema["product"]["attributes"]["condition"] == "new"
        assert schema["product"]["attributes"]["age_group"] == "adult"
        assert schema["product"]["attributes"]["gender"] == "female"
        assert schema["campaigns"]["short_title"] == "Elegant Black Handbag"

    def test_merges_product_details_and_highlights(self, sample_vlm_response, sample_extracted_fields):
        schema = _build_acp_schema(sample_vlm_response, [], sample_extracted_fields)

        assert len(schema["product"]["details"]) == 2
        assert schema["product"]["details"][0]["attribute_name"] == "Closure Type"
        assert len(schema["product"]["highlights"]) == 3

    def test_includes_faqs(self, sample_vlm_response, sample_faqs_response, sample_extracted_fields):
        schema = _build_acp_schema(sample_vlm_response, sample_faqs_response, sample_extracted_fields)

        assert len(schema["faqs"]) == len(sample_faqs_response)
        assert schema["faqs"][0]["question"] == sample_faqs_response[0]["question"]

    def test_empty_faqs(self, sample_vlm_response):
        schema = _build_acp_schema(sample_vlm_response, [], {})
        assert schema["faqs"] == []

    def test_agent_actions_defaults(self, sample_vlm_response):
        schema = _build_acp_schema(sample_vlm_response, [], {})

        assert schema["agent_actions"]["discoverable"] is True
        assert schema["agent_actions"]["buyable"] is True
        assert schema["agent_actions"]["returnable"] is True
        assert schema["agent_actions"]["comparable"] is True
        assert schema["agent_actions"]["subscribable"] is False

    def test_deterministic_defaults(self, sample_vlm_response):
        schema = _build_acp_schema(sample_vlm_response, [], {})

        assert schema["pricing"]["availability"] == "in_stock"
        assert schema["bundling"]["is_bundle"] is False

    def test_empty_extracted_falls_back(self, sample_vlm_response):
        schema = _build_acp_schema(sample_vlm_response, [], {})

        assert schema["product"]["brand"] is None
        assert schema["product"]["attributes"]["material"] is None
        assert schema["product"]["attributes"]["condition"] == "new"
        assert schema["campaigns"]["short_title"] == sample_vlm_response["title"]

    def test_null_extracted_values_do_not_suppress_enriched_fallbacks(self, sample_vlm_response):
        extracted = {
            "condition": None,
            "short_title": "",
            "product_details": None,
            "product_highlights": None,
        }

        schema = _build_acp_schema(sample_vlm_response, [], extracted)

        assert schema["product"]["title"] == sample_vlm_response["title"]
        assert schema["product"]["description"] == sample_vlm_response["description"]
        assert schema["product"]["attributes"]["condition"] == "new"
        assert schema["product"]["details"] == []
        assert schema["product"]["highlights"] == sample_vlm_response["tags"]
        assert schema["campaigns"]["short_title"] == sample_vlm_response["title"]

    def test_nested_measurement_structure(self, sample_vlm_response):
        schema = _build_acp_schema(sample_vlm_response, [], {})

        dims = schema["product"]["dimensions"]
        for key in ("length", "width", "height", "weight"):
            assert dims[key] == {"value": None, "unit": None}

    def test_nested_money_structure(self, sample_vlm_response):
        schema = _build_acp_schema(sample_vlm_response, [], {})

        installment = schema["pricing"]["installment"]
        assert installment["amount"] == {"amount": None, "currency": None}
        assert installment["downpayment"] == {"amount": None, "currency": None}

    def test_all_top_level_sections_present(self, sample_vlm_response):
        schema = _build_acp_schema(sample_vlm_response, [], {})

        expected_sections = [
            "product", "pricing", "faqs", "agent_actions",
            "fulfillment", "campaigns", "certifications",
            "energy_efficiency", "bundling", "marketplace", "metadata",
        ]
        for section in expected_sections:
            assert section in schema, f"Missing top-level section: {section}"

    def test_empty_enriched_data(self):
        schema = _build_acp_schema({}, [], {})

        assert schema["product"]["title"] == ""
        assert schema["product"]["description"] == ""
        assert schema["product"]["attributes"]["colors"] == []
        assert schema["product"]["categories"] == []
        assert schema["product"]["tags"] == []


class TestBuildUcpSchema:
    """Tests for the UCP schema builder."""

    def test_structured_title_and_description(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})

        assert schema["title"] == sample_vlm_response["title"]
        assert schema["structured_title"]["digital_source_type"] == "trained_algorithmic_media"
        assert schema["structured_title"]["content"] == sample_vlm_response["title"]
        assert schema["description"] == sample_vlm_response["description"]
        assert schema["structured_description"]["digital_source_type"] == "trained_algorithmic_media"
        assert schema["structured_description"]["content"] == sample_vlm_response["description"]

    def test_merges_extracted_fields(self, sample_vlm_response, sample_extracted_fields):
        schema = _build_ucp_schema(sample_vlm_response, [], sample_extracted_fields)

        assert schema["brand"] == "Elegance Co"
        assert schema["condition"] == "new"
        assert schema["material"] == "leather"
        assert schema["age_group"] == "adult"
        assert schema["gender"] == "female"
        assert schema["short_title"] == "Elegant Black Handbag"
        assert schema["google_product_category"] == "Apparel & Accessories > Handbags"

    def test_merges_product_details(self, sample_vlm_response, sample_extracted_fields):
        schema = _build_ucp_schema(sample_vlm_response, [], sample_extracted_fields)

        assert len(schema["product_detail"]) == 2
        assert schema["product_detail"][0]["attribute_name"] == "Closure Type"

    def test_deterministic_defaults(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})

        assert schema["availability"] == "in_stock"
        assert schema["adult"] is False
        assert schema["is_bundle"] is False
        assert schema["identifier_exists"] is False

    def test_color_joined_with_slash(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})
        assert schema["color"] == "black / gold"

    def test_product_type_joined_with_arrow(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})
        assert schema["product_type"] == "bags"

    def test_product_highlight_from_tags(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})
        assert schema["product_highlight"] == sample_vlm_response["tags"]

    def test_product_highlight_from_extracted(self, sample_vlm_response, sample_extracted_fields):
        schema = _build_ucp_schema(sample_vlm_response, [], sample_extracted_fields)
        assert schema["product_highlight"] == sample_extracted_fields["product_highlights"]

    def test_includes_faqs(self, sample_vlm_response, sample_faqs_response):
        schema = _build_ucp_schema(sample_vlm_response, sample_faqs_response, {})

        assert len(schema["faqs"]) == len(sample_faqs_response)
        assert schema["faqs"][0]["question"] == sample_faqs_response[0]["question"]

    def test_empty_faqs(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})
        assert schema["faqs"] == []

    def test_money_type_structure(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})

        for field in ("price", "sale_price", "cost_of_goods_sold", "auto_pricing_min_price", "maximum_retail_price"):
            assert schema[field] == {"amount": None, "currency": None}, f"{field} should be a money object"

    def test_measurement_type_structure(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})

        for field in ("unit_pricing_measure", "unit_pricing_base_measure",
                       "product_length", "product_width", "product_height", "product_weight",
                       "shipping_weight", "shipping_length", "shipping_width", "shipping_height"):
            assert schema[field] == {"value": None, "unit": None}, f"{field} should be a measurement object"

    def test_array_fields_are_lists(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})

        array_fields = [
            "additional_image_link", "loyalty_program", "certification",
            "size_type", "product_detail", "promotion_id",
            "excluded_destination", "included_destination", "shopping_ads_excluded_country",
            "shipping", "carrier_shipping", "handling_cutoff_time", "minimum_order_value",
            "shipping_transit_business_days", "shipping_handling_business_days", "free_shipping_threshold",
        ]
        for field in array_fields:
            assert isinstance(schema[field], list), f"{field} should be a list"

    def test_installment_nested_structure(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})

        inst = schema["installment"]
        assert inst["months"] is None
        assert inst["amount"] == {"amount": None, "currency": None}
        assert inst["downpayment"] == {"amount": None, "currency": None}
        assert inst["credit_type"] is None

    def test_subscription_cost_nested_structure(self, sample_vlm_response):
        schema = _build_ucp_schema(sample_vlm_response, [], {})

        sub = schema["subscription_cost"]
        assert sub["period"] is None
        assert sub["period_length"] is None
        assert sub["amount"] == {"amount": None, "currency": None}

    def test_empty_colors_returns_null(self):
        schema = _build_ucp_schema({"title": "t", "description": "d", "colors": [], "categories": [], "tags": []}, [], {})
        assert schema["color"] is None

    def test_empty_categories_returns_null(self):
        schema = _build_ucp_schema({"title": "t", "description": "d", "colors": [], "categories": [], "tags": []}, [], {})
        assert schema["product_type"] is None

    def test_empty_tags_returns_empty_list(self):
        schema = _build_ucp_schema({"title": "t", "description": "d", "colors": [], "categories": [], "tags": []}, [], {})
        assert schema["product_highlight"] == []

    def test_null_extracted_values_do_not_suppress_enriched_fallbacks(self, sample_vlm_response):
        extracted = {
            "condition": None,
            "short_title": "",
            "product_details": None,
            "product_highlights": None,
        }

        schema = _build_ucp_schema(sample_vlm_response, [], extracted)

        assert schema["title"] == sample_vlm_response["title"]
        assert schema["description"] == sample_vlm_response["description"]
        assert schema["condition"] == "new"
        assert schema["product_detail"] == []
        assert schema["product_highlight"] == sample_vlm_response["tags"]
        assert schema["short_title"] == sample_vlm_response["title"]

    def test_all_ucp_sections_present(self, sample_vlm_response):
        """Verify all 9 UCP sections have at least one representative field."""
        schema = _build_ucp_schema(sample_vlm_response, [], {})

        section_fields = {
            "Basic product data": "structured_title",
            "Price and availability": "price",
            "Product category": "product_type",
            "Product identifiers": "brand",
            "Detailed product description": "color",
            "Shopping campaigns": "custom_label_0",
            "Marketplaces": "external_seller_id",
            "Destinations": "excluded_destination",
            "Shipping and returns": "shipping",
        }
        for section, field in section_fields.items():
            assert field in schema, f"Missing field '{field}' from section '{section}'"
