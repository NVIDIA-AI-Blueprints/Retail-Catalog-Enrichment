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

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.config import Config
from backend.main import app
from backend.web_insights import (
    EXA_API_KEY_NOT_CONFIGURED_MESSAGE,
    WebInsightsDependencyError,
    _normalize_exa_result,
    _parse_agent_json,
    _search_exa_sources,
    build_product_web_insights,
    normalize_web_insights_output,
)


def test_web_insights_config_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
vlm:
  url: http://test-vlm/v1
  model: test-vlm
llm:
  url: http://test-llm/v1
  model: test-llm
""",
        encoding="utf-8",
    )

    config = Config(str(config_file))

    assert config.get_web_insights_config() == {
        "max_results": 8,
        "max_sources": 8,
        "min_sources": 2,
        "search_type": "auto",
        "highlights_max_characters": 4000,
    }


def test_normalize_exa_result_uses_highlights():
    result = SimpleNamespace(
        title="Review",
        url="https://example.com/review",
        published_date="2026-01-01",
        highlights=["Compact and durable.", "Battery life is a common purchase factor."],
    )

    normalized = _normalize_exa_result(result)

    assert normalized == {
        "title": "Review",
        "url": "https://example.com/review",
        "published_date": "2026-01-01",
        "snippet": "Compact and durable. Battery life is a common purchase factor.",
    }


def test_normalize_exa_result_preserves_existing_snippet():
    result = {
        "title": "Retailer page",
        "url": "https://example.com/product",
        "published_date": None,
        "snippet": "Existing normalized snippet.",
    }

    assert _normalize_exa_result(result)["snippet"] == "Existing normalized snippet."


def test_normalize_exa_result_removes_source_boilerplate():
    result = {
        "title": "Oster page",
        "url": "https://example.com/oster",
        "published_date": None,
        "snippet": (
            "Error Your Browser Is No Longer Supported. Please use an alternative browser to improve your experience and security.\n"
            "Skip to main content\n"
            "Enjoy healthier fried foods with the Oster Digital Air Fryer and 10 preset functions."
        ),
    }

    normalized = _normalize_exa_result(result)

    assert "Browser Is No Longer Supported" not in normalized["snippet"]
    assert "Skip to main content" not in normalized["snippet"]
    assert "Oster Digital Air Fryer" in normalized["snippet"]


def test_normalize_exa_result_does_not_use_exa_summary():
    result = {
        "title": "Retailer page",
        "url": "https://example.com/product",
        "published_date": None,
        "summary": "Exa-generated summary should not be used.",
        "text": "Raw retrieved source text should be used.",
    }

    normalized = _normalize_exa_result(result)

    assert normalized["snippet"] == "Raw retrieved source text should be used."


def test_search_exa_sources_does_not_request_exa_summary():
    exa = Mock()
    exa.search.return_value = SimpleNamespace(results=[])

    _search_exa_sources(
        exa,
        "product review",
        search_type="auto",
        max_results=5,
        highlights_max_characters=1200,
        query_log=[],
    )

    contents = exa.search.call_args.kwargs["contents"]
    assert "summary" not in contents
    assert contents["highlights"] == {"query": "product review", "max_characters": 1200}
    assert contents["text"] == {"max_characters": 1200}


def test_parse_agent_json_unwraps_json_string():
    payload = {"summary": "Nested JSON", "pros": ["Useful"]}
    wrapped = json.dumps(json.dumps(payload))

    assert _parse_agent_json(wrapped) == payload


def test_normalize_output_warns_on_low_source_coverage():
    output = normalize_web_insights_output(
        {
            "summary": "Useful context.",
            "pros": ["Strong durability"],
            "warnings": [],
        },
        collected_sources=[{"title": "Only source", "url": "https://example.com", "snippet": "source"}],
        search_queries=["product review"],
        locale="en-US",
        min_sources=2,
        max_sources=8,
    )

    assert output["summary"] == "Useful context."
    assert output["pros"] == ["Strong durability"]
    assert output["search_queries"] == ["product review"]
    assert len(output["sources"]) == 1
    assert output["report"]["executive_summary"] == "Useful context."
    assert output["report"]["retail_insights"][0] == {
        "type": "positive",
        "title": "Strong durability",
        "detail": "Strong durability",
    }
    assert output["warnings"] == ["Only 1 relevant source found; review coverage before using these insights."]


def test_normalize_output_suppresses_claims_without_sources():
    output = normalize_web_insights_output(
        {
            "summary": "Unsupported summary.",
            "pros": ["Unsupported pro"],
        },
        collected_sources=[],
        search_queries=["product review"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    assert output["summary"] == "No source-backed web insights were generated for this product."
    assert output["pros"] == []
    assert output["sources"] == []
    assert output["report"]["executive_summary"] == ""
    assert output["report"]["retail_insights"] == []
    assert output["report"]["metrics"]["customer_sentiment"]["score"] is None
    assert output["warnings"] == ["Only 0 relevant sources found; review coverage before using these insights."]


def test_normalize_output_suppresses_claims_without_source_snippets():
    output = normalize_web_insights_output(
        {
            "summary": "Unsupported review summary.",
            "pros": ["Customers love it"],
            "cons": ["Some users report failures"],
            "research_scope": "brand_level",
            "identity_confidence": "high",
            "detected_brand": "Oster",
            "scope_note": "Source titles confirm the brand.",
            "report": {
                "metrics": {
                    "customer_sentiment": {"label": "Positive", "score": 92, "rationale": "Unsupported rating claim."},
                },
                "retail_insights": [
                    {"type": "positive", "title": "High ratings", "detail": "Unsupported review detail."},
                ],
            },
        },
        collected_sources=[{"title": "Oster Air Fryer", "url": "https://example.com/oster", "snippet": ""}],
        search_queries=["Oster air fryer"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    assert output["summary"] == "Found matching web sources, but no source snippets were available to generate a source-backed report."
    assert output["pros"] == []
    assert output["cons"] == []
    assert output["sources"][0]["url"] == "https://example.com/oster"
    assert output["research_scope"] == "brand_level"
    assert output["detected_brand"] == "Oster"
    assert output["report"]["retail_insights"] == []
    assert output["report"]["metrics"]["customer_sentiment"]["score"] is None
    assert output["warnings"] == [
        "Found matching source URLs, but Exa returned no snippets or text excerpts; report claims were suppressed."
    ]


def test_normalize_output_replaces_placeholder_metric_labels_from_sources():
    output = normalize_web_insights_output(
        {
            "summary": "Source-backed air fryer report.",
            "research_scope": "brand_level",
            "identity_confidence": "high",
            "detected_brand": "Oster",
            "scope_note": "Sources confirm the brand and model family.",
            "report": {
                "metrics": {
                    "customer_sentiment": {
                        "label": "Not enough data",
                        "score": None,
                        "rationale": "",
                    },
                    "build_quality": {
                        "label": "qualitative label",
                        "score": None,
                        "rationale": "",
                    },
                    "price_segment": {
                        "label": "qualitative label",
                        "score": None,
                        "rationale": "",
                    },
                    "retail_confidence": {
                        "label": "qualitative label",
                        "score": None,
                        "rationale": "",
                    },
                },
            },
        },
        collected_sources=[
            {
                "title": "Oster Digital Air Fryer",
                "url": "https://example.com/oster",
                "snippet": (
                    "Oster Digital Air Fryer is rated 4.8 out of 5 by 96. "
                    "The nonstick coating is scratch resistant and easy to clean. "
                    "Sale price $89.99 from a retailer."
                ),
            },
            {
                "title": "Oster retailer listing",
                "url": "https://example.com/retailer",
                "snippet": "Customer rating 4.9 out of 5 with durable stainless steel styling.",
            },
        ],
        search_queries=["Oster air fryer"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    metrics = output["report"]["metrics"]
    assert metrics["customer_sentiment"]["label"] == "Positive"
    assert metrics["customer_sentiment"]["score"] == 97
    assert metrics["build_quality"]["label"] == "Durable"
    assert metrics["build_quality"]["score"] is not None
    assert metrics["price_segment"]["label"] == "Mid-range"
    assert metrics["retail_confidence"]["label"] == "Moderate"
    assert metrics["retail_confidence"]["score"] is not None
    assert "qualitative label" not in {metric["label"].lower() for metric in metrics.values()}


def test_normalize_output_replaces_varies_by_retailer_price_label():
    output = normalize_web_insights_output(
        {
            "summary": "Source-backed air fryer report.",
            "research_scope": "brand_level",
            "identity_confidence": "high",
            "detected_brand": "Oster",
            "report": {
                "metrics": {
                    "price_segment": {
                        "label": "Varies by retailer",
                        "score": None,
                        "rationale": "Retail prices vary.",
                    },
                },
            },
        },
        collected_sources=[
            {
                "title": "Oster retailer listing",
                "url": "https://example.com/oster",
                "snippet": "Retail listing shows current price $149.95 for the air fryer oven.",
            }
        ],
        search_queries=["Oster air fryer price"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    metric = output["report"]["metrics"]["price_segment"]
    assert metric["label"] == "Mid-range"
    assert metric["rationale"] == "Source snippets include retail prices in the $150 range."


def test_normalize_output_ignores_incomplete_walmart_price_markup():
    output = normalize_web_insights_output(
        {
            "summary": "Source-backed air fryer report.",
            "research_scope": "brand_level",
            "identity_confidence": "high",
            "detected_brand": "Oster",
            "report": {"metrics": {"price_segment": {"label": "Varies by retailer", "score": None}}},
        },
        collected_sources=[
            {
                "title": "Walmart listing",
                "url": "https://example.com/walmart",
                "snippet": "Add $16999 current price $169.99 Oster Extra-Large Air Fryer Oven.",
            }
        ],
        search_queries=["Oster air fryer price"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    metric = output["report"]["metrics"]["price_segment"]
    assert metric["label"] == "Mid-range"
    assert metric["rationale"] == "Source snippets include retail prices in the $170 range."


def test_normalize_output_uses_price_unavailable_without_price_values():
    output = normalize_web_insights_output(
        {
            "summary": "Source-backed air fryer report.",
            "research_scope": "brand_level",
            "identity_confidence": "high",
            "detected_brand": "Oster",
            "report": {
                "metrics": {
                    "price_segment": {
                        "label": "Varies by retailer",
                        "score": None,
                        "rationale": "Retail prices vary.",
                    },
                },
            },
        },
        collected_sources=[
            {
                "title": "Oster retailer listing",
                "url": "https://example.com/oster",
                "snippet": "Retail listing confirms an Oster air fryer but does not include a price.",
            }
        ],
        search_queries=["Oster air fryer price"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    metric = output["report"]["metrics"]["price_segment"]
    assert metric["label"] == "Price unavailable"
    assert metric["rationale"] == "Source snippets do not include reliable price evidence."


def test_normalize_output_preserves_price_source_when_source_limit_applies():
    collected_sources = [
        {"title": f"Source {index}", "url": f"https://example.com/{index}", "snippet": "No price here."}
        for index in range(8)
    ]
    collected_sources.append(
        {
            "title": "Price source",
            "url": "https://example.com/price",
            "snippet": "Current price $169.00 for the Oster air fryer oven.",
        }
    )

    output = normalize_web_insights_output(
        {
            "summary": "Source-backed report.",
            "research_scope": "brand_level",
            "identity_confidence": "high",
            "detected_brand": "Oster",
            "report": {"metrics": {"price_segment": {"label": "Price unavailable", "score": None}}},
        },
        collected_sources=collected_sources,
        search_queries=["Oster air fryer price"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    assert any(source["url"] == "https://example.com/price" for source in output["sources"])
    assert output["report"]["metrics"]["price_segment"]["label"] == "Mid-range"


def test_normalize_output_suppresses_source_boilerplate_insights():
    output = normalize_web_insights_output(
        {
            "summary": "Source-backed air fryer report.",
            "pros": ["Viewing window helps monitor cooking."],
            "cons": [
                "Browser compatibility error: Several pages return a browser warning.",
                "Some models vary by capacity.",
            ],
            "warnings": ["Unsupported browser message limited access to full product details."],
            "research_scope": "brand_level",
            "identity_confidence": "high",
            "detected_brand": "Oster",
            "report": {
                "retail_insights": [
                    {
                        "type": "negative",
                        "title": "Browser compatibility error",
                        "detail": "Several Oster pages return a 'Your Browser Is No Longer Supported' message.",
                    },
                    {
                        "type": "positive",
                        "title": "Multi-function cooking",
                        "detail": "Sources mention 10 preset cooking functions.",
                    },
                ],
            },
        },
        collected_sources=[
            {
                "title": "Oster Digital Air Fryer",
                "url": "https://example.com/oster",
                "snippet": "Oster Digital Air Fryer is rated 4.8 out of 5 and has 10 preset cooking functions.",
            }
        ],
        search_queries=["Oster air fryer"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    assert output["cons"] == ["Some models vary by capacity."]
    assert output["warnings"] == []
    assert output["report"]["retail_insights"] == [
        {
            "type": "positive",
            "title": "Multi-function cooking",
            "detail": "Sources mention 10 preset cooking functions.",
        }
    ]


def test_normalize_output_adds_report_and_clamps_scores():
    output = normalize_web_insights_output(
        {
            "summary": "Summary",
            "research_scope": "product_specific",
            "identity_confidence": "high",
            "detected_brand": "TestBrand",
            "detected_model": "Model 1",
            "scope_note": "Sources confirm the exact product identity.",
            "identity_evidence": ["Official and retailer pages match the product title."],
            "report": {
                "executive_summary": "Executive summary",
                "positioning_tags": ["Premium", "Durable"],
                "metrics": {
                    "customer_sentiment": {
                        "label": "Very positive",
                        "score": 124,
                        "rationale": "Reviews skew positive.",
                    },
                    "build_quality": {
                        "label": "Premium",
                        "score": 12,
                        "rationale": "Materials are repeatedly praised.",
                    },
                    "price_segment": {
                        "label": "High-end",
                        "score": 5,
                        "rationale": "Retail listings price it above alternatives.",
                    },
                    "retail_confidence": {
                        "label": "Strong",
                        "score": -2,
                        "rationale": "Source coverage is thin.",
                    },
                },
                "retail_insights": [
                    {"type": "positive", "title": "Premium construction", "detail": "Sources mention durable materials."},
                    {"type": "negative", "title": "Higher price", "detail": "Pricing may limit entry-level conversion."},
                ],
                "primary_use_cases": [
                    {"title": "Specialty retail", "detail": "Fits premium product positioning."},
                ],
                "customer_sentiment_summary": "Customers associate it with quality.",
            },
        },
        collected_sources=[{"title": "Review", "url": "https://example.com/review", "snippet": "source"}],
        search_queries=["product review"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    report = output["report"]
    assert report["executive_summary"] == "Executive summary"
    assert report["positioning_tags"] == ["Premium", "Durable"]
    assert report["metrics"]["customer_sentiment"]["score"] == 100
    assert report["metrics"]["build_quality"]["score"] == 10.0
    assert report["metrics"]["price_segment"]["score"] is None
    assert report["metrics"]["retail_confidence"]["score"] == 0.0
    assert report["retail_insights"][1]["type"] == "negative"
    assert report["primary_use_cases"][0]["title"] == "Specialty retail"
    assert output["research_scope"] == "product_specific"
    assert output["detected_brand"] == "TestBrand"
    assert output["identity_evidence"] == ["Official and retailer pages match the product title."]


def test_normalize_output_accepts_source_backed_brand_scope():
    output = normalize_web_insights_output(
        {
            "summary": "Oster air fryer research.",
            "research_scope": "brand_level",
            "identity_confidence": "medium",
            "detected_brand": "Oster",
            "detected_model": None,
            "scope_note": "Sources confirm Oster as the brand, but no exact model was confirmed.",
            "identity_evidence": ["Retailer pages list Oster air fryers matching the title and product type."],
            "report": {
                "metrics": {
                    "customer_sentiment": {"label": "Positive", "score": 83, "rationale": "Reviews skew positive."},
                    "retail_confidence": {"label": "Moderate", "score": 7.1, "rationale": "Brand evidence is present."},
                },
            },
        },
        collected_sources=[{"title": "Oster Air Fryer", "url": "https://example.com/oster-air-fryer", "snippet": "source"}],
        search_queries=["Oster Air Fryer with Digital Display and Stainless Steel Finish"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    assert output["research_scope"] == "brand_level"
    assert output["identity_confidence"] == "medium"
    assert output["detected_brand"] == "Oster"
    assert output["detected_model"] is None
    assert output["report"]["metrics"]["customer_sentiment"]["score"] == 83
    assert output["report"]["metrics"]["retail_confidence"]["score"] == 7.1


def test_normalize_output_suppresses_product_scores_for_category_scope():
    output = normalize_web_insights_output(
        {
            "summary": "Category-level gas grill patterns.",
            "research_scope": "category_level",
            "identity_confidence": "low",
            "detected_brand": None,
            "detected_model": None,
            "scope_note": "No source confirms a brand or model; research is category-level.",
            "report": {
                "metrics": {
                    "customer_sentiment": {"label": "Positive", "score": 91, "rationale": "Review scores are high."},
                    "build_quality": {"label": "Durable", "score": 8.8, "rationale": "Materials are praised."},
                    "price_segment": {"label": "Mid-market", "score": None, "rationale": "Retail prices vary."},
                    "retail_confidence": {"label": "Strong", "score": 8.6, "rationale": "Sources are consistent."},
                },
                "retail_insights": [
                    {"type": "positive", "title": "Cooking area", "detail": "Category sources emphasize grill size."},
                ],
            },
        },
        collected_sources=[{"title": "Guide", "url": "https://example.com/grill-guide", "snippet": "source"}],
        search_queries=["black gas grill buying guide"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    assert output["research_scope"] == "category_level"
    assert output["identity_confidence"] == "low"
    assert output["detected_brand"] is None
    assert output["detected_model"] is None
    assert output["report"]["metrics"]["customer_sentiment"]["label"] == "Category signal"
    assert output["report"]["metrics"]["customer_sentiment"]["score"] is None
    assert output["report"]["metrics"]["retail_confidence"]["label"] == "Limited"
    assert output["report"]["metrics"]["retail_confidence"]["score"] is None


def test_normalize_output_missing_identity_scope_defaults_to_category_level():
    output = normalize_web_insights_output(
        {
            "summary": "Legacy report without identity fields.",
            "report": {
                "metrics": {
                    "customer_sentiment": {"label": "Positive", "score": 92, "rationale": "Legacy score."},
                    "retail_confidence": {"label": "Strong", "score": 9, "rationale": "Legacy score."},
                },
            },
        },
        collected_sources=[{"title": "Guide", "url": "https://example.com/guide", "snippet": "source"}],
        search_queries=["legacy product review"],
        locale="en-US",
        min_sources=1,
        max_sources=8,
    )

    assert output["research_scope"] == "category_level"
    assert output["identity_confidence"] == "low"
    assert output["detected_brand"] is None
    assert output["detected_model"] is None
    assert output["scope_note"] == "The research agent did not return a supported identity scope; insights are treated as category-level context."
    assert output["report"]["metrics"]["customer_sentiment"]["score"] is None
    assert output["report"]["metrics"]["retail_confidence"]["score"] is None


def test_build_product_web_insights_returns_disabled_when_exa_api_key_missing(monkeypatch):
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    output = build_product_web_insights(title="Test product", locale="en-US")

    assert output["status"] == "disabled"
    assert output["disabled_reason"] == EXA_API_KEY_NOT_CONFIGURED_MESSAGE
    assert output["warnings"] == [EXA_API_KEY_NOT_CONFIGURED_MESSAGE]
    assert output["sources"] == []
    assert output["report"]["retail_insights"] == []


@patch("backend.web_insights.create_deep_agent")
@patch("backend.web_insights.ChatOpenAI")
@patch("backend.web_insights.Exa")
@patch("backend.web_insights.get_config")
def test_build_product_web_insights_returns_schema(
    mock_get_config,
    mock_exa_class,
    mock_chat_openai,
    mock_create_deep_agent,
    monkeypatch,
):
    monkeypatch.setenv("NGC_API_KEY", "test-ngc")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")

    mock_config = Mock()
    mock_config.get_llm_config.return_value = {"url": "http://test-llm/v1", "model": "test-llm"}
    mock_config.get_web_insights_config.return_value = {
        "max_results": 8,
        "max_sources": 8,
        "min_sources": 1,
        "search_type": "auto",
        "highlights_max_characters": 4000,
    }
    mock_get_config.return_value = mock_config

    exa = Mock()
    exa.search.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(
                title="Product review",
                url="https://example.com/review",
                published_date=None,
                highlights=["Reviewers mention durability and everyday use. Retail price is $89.99."],
            )
        ]
    )
    mock_exa_class.return_value = exa

    agent_result = {
        "messages": [
            SimpleNamespace(
                content=json.dumps({
                    "summary": "Source-backed summary.",
                    "pros": ["Durable"],
                    "cons": ["Limited color options"],
                    "use_cases": ["Everyday use"],
                    "customer_insights": ["Customers compare durability."],
                    "purchase_considerations": ["Clarify compatibility."],
                    "search_queries": ["product review"],
                    "warnings": [],
                    "research_scope": "product_specific",
                    "identity_confidence": "high",
                    "detected_brand": "Test",
                    "detected_model": "product",
                    "scope_note": "Sources appear to match a specific product identity.",
                    "identity_evidence": ["Search results matched the test product."],
                    "report": {
                        "executive_summary": "Source-backed dashboard summary.",
                        "positioning_tags": ["Durability"],
                        "metrics": {
                            "customer_sentiment": {
                                "label": "Positive",
                                "score": 82,
                                "rationale": "Review snippets mention durability.",
                            },
                            "build_quality": {
                                "label": "Durable",
                                "score": 8,
                                "rationale": "Sources mention durability.",
                            },
                            "price_segment": {
                                "label": "Mid-market",
                                "score": None,
                                "rationale": "Pricing context is limited.",
                            },
                            "retail_confidence": {
                                "label": "Moderate",
                                "score": 7.2,
                                "rationale": "One relevant source was found.",
                            },
                        },
                        "retail_insights": [
                            {"type": "positive", "title": "Durable", "detail": "Reviewers mention durable use."},
                        ],
                        "primary_use_cases": [
                            {"title": "Everyday use", "detail": "Sources mention everyday use."},
                        ],
                        "customer_sentiment_summary": "Customers compare durability.",
                    },
                })
            )
        ]
    }

    agent = Mock()

    def _invoke(_payload):
        web_search_tool = mock_create_deep_agent.call_args.kwargs["tools"][0]
        web_search_tool("product review")
        return agent_result

    agent.invoke.side_effect = _invoke
    mock_create_deep_agent.return_value = agent

    output = build_product_web_insights(title="Test product", locale="en-US")

    assert output["summary"] == "Source-backed summary."
    assert output["pros"] == ["Durable"]
    assert output["research_scope"] == "product_specific"
    assert output["identity_confidence"] == "high"
    assert output["identity_evidence"] == ["Search results matched the test product."]
    assert output["report"]["executive_summary"] == "Source-backed dashboard summary."
    assert output["report"]["metrics"]["customer_sentiment"]["score"] == 82
    assert output["sources"][0]["url"] == "https://example.com/review"
    mock_exa_class.assert_called_once_with(api_key="test-exa")
    exa.search.assert_called_once()
    mock_chat_openai.assert_called_once_with(
        model="test-llm",
        base_url="http://test-llm/v1",
        api_key="test-ngc",
        temperature=0.1,
        max_tokens=4096,
        timeout=60,
        max_retries=1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    mock_create_deep_agent.assert_called_once()


@patch("backend.web_insights.create_deep_agent")
@patch("backend.web_insights.ChatOpenAI")
@patch("backend.web_insights.Exa")
@patch("backend.web_insights.get_config")
def test_build_product_web_insights_bootstraps_sources_when_agent_skips_tool(
    mock_get_config,
    mock_exa_class,
    _mock_chat_openai,
    mock_create_deep_agent,
    monkeypatch,
):
    monkeypatch.setenv("NGC_API_KEY", "test-ngc")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")

    mock_config = Mock()
    mock_config.get_llm_config.return_value = {"url": "http://test-llm/v1", "model": "test-llm"}
    mock_config.get_web_insights_config.return_value = {
        "max_results": 8,
        "max_sources": 8,
        "min_sources": 1,
        "search_type": "auto",
        "highlights_max_characters": 4000,
    }
    mock_get_config.return_value = mock_config

    exa = Mock()
    exa.search.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(
                title="Oster Air Fryer retailer page",
                url="https://example.com/oster-air-fryer",
                published_date=None,
                highlights=["Retailer page identifies Oster as the air fryer brand."],
            )
        ]
    )
    mock_exa_class.return_value = exa

    first_agent_result = {
        "messages": [
            SimpleNamespace(
                content=json.dumps({
                    "summary": "",
                    "search_queries": ["Oster Air Fryer with Digital Display and Cooking Functions"],
                    "warnings": [],
                    "research_scope": "brand_level",
                    "identity_confidence": "medium",
                    "detected_brand": "Oster",
                    "detected_model": None,
                    "scope_note": "Brand found, model unclear.",
                    "identity_evidence": ["Oster appears in matching retailer titles."],
                    "report": {},
                })
            )
        ]
    }
    grounded_agent_result = {
        "messages": [
            SimpleNamespace(
                content=json.dumps({
                    "summary": "Oster source-backed air fryer summary.",
                    "pros": ["Brand-recognized small appliance line"],
                    "cons": [],
                    "use_cases": ["Countertop air frying"],
                    "customer_insights": ["Shoppers compare functions and controls."],
                    "purchase_considerations": ["Verify capacity and exact model."],
                    "search_queries": ["Oster Air Fryer with Digital Display and Cooking Functions"],
                    "warnings": [],
                    "research_scope": "brand_level",
                    "identity_confidence": "medium",
                    "detected_brand": "Oster",
                    "detected_model": None,
                    "scope_note": "Sources confirm Oster as the brand, but no exact model was confirmed.",
                    "identity_evidence": ["Retailer page identifies Oster as the air fryer brand."],
                    "report": {
                        "executive_summary": "Oster source-backed dashboard summary.",
                        "metrics": {
                            "customer_sentiment": {"label": "Not enough data", "score": None, "rationale": ""},
                            "retail_confidence": {"label": "Moderate", "score": 7, "rationale": "One retailer source matched."},
                        },
                        "retail_insights": [
                            {"type": "positive", "title": "Brand match", "detail": "Retailer source identifies Oster."}
                        ],
                        "primary_use_cases": [
                            {"title": "Countertop cooking", "detail": "Source context supports air fryer use."}
                        ],
                        "customer_sentiment_summary": "Not enough customer sentiment sources were found.",
                    },
                })
            )
        ]
    }

    agent = Mock()
    agent.invoke.side_effect = [first_agent_result, grounded_agent_result]
    mock_create_deep_agent.return_value = agent

    output = build_product_web_insights(
        title="Oster Air Fryer with Digital Display and Cooking Functions",
        categories=["appliances"],
        tags=["air fryer", "digital display"],
        locale="en-US",
    )

    assert agent.invoke.call_count == 2
    assert exa.search.called
    assert output["summary"] == "Oster source-backed air fryer summary."
    assert output["sources"][0]["url"] == "https://example.com/oster-air-fryer"
    assert output["research_scope"] == "brand_level"
    assert output["detected_brand"] == "Oster"
    assert output["report"]["executive_summary"] == "Oster source-backed dashboard summary."


def test_product_insights_endpoint_success():
    expected = {
        "summary": "Summary",
        "pros": [],
        "cons": [],
        "use_cases": [],
        "customer_insights": [],
        "purchase_considerations": [],
        "search_queries": [],
        "sources": [],
        "warnings": [],
        "locale": "en-US",
    }
    with patch("backend.main.build_product_web_insights", return_value=expected) as mock_build:
        client = TestClient(app)
        response = client.post(
            "/research/product-insights",
            data={
                "title": "Test product",
                "description": "Description",
                "categories": '["electronics"]',
                "tags": '["portable"]',
                "locale": "en-US",
            },
        )

    assert response.status_code == 200
    assert response.json() == expected
    mock_build.assert_called_once()


def test_product_insights_endpoint_returns_disabled_when_exa_api_key_missing(monkeypatch):
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    client = TestClient(app)
    response = client.post("/research/product-insights", data={"title": "Product", "locale": "en-US"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "disabled"
    assert data["disabled_reason"] == EXA_API_KEY_NOT_CONFIGURED_MESSAGE
    assert data["warnings"] == [EXA_API_KEY_NOT_CONFIGURED_MESSAGE]
    assert data["sources"] == []


def test_product_insights_endpoint_validation_errors():
    client = TestClient(app)

    missing_title = client.post("/research/product-insights", data={"title": ""})
    assert missing_title.status_code == 400
    assert missing_title.json()["detail"] == "title is required"

    bad_locale = client.post("/research/product-insights", data={"title": "Product", "locale": "de-DE"})
    assert bad_locale.status_code == 400

    malformed_json = client.post(
        "/research/product-insights",
        data={"title": "Product", "categories": "{bad json"},
    )
    assert malformed_json.status_code == 400
    assert malformed_json.json()["detail"] == "Invalid JSON in request fields."

    wrong_json_shape = client.post(
        "/research/product-insights",
        data={"title": "Product", "categories": '{"not":"array"}'},
    )
    assert wrong_json_shape.status_code == 400
    assert wrong_json_shape.json()["detail"] == "categories and tags must be JSON arrays"


def test_product_insights_endpoint_dependency_error():
    with patch(
        "backend.main.build_product_web_insights",
        side_effect=WebInsightsDependencyError("EXA_API_KEY is not set"),
    ):
        client = TestClient(app)
        response = client.post("/research/product-insights", data={"title": "Product"})

    assert response.status_code == 503
    assert response.json()["detail"] == "EXA_API_KEY is not set"
