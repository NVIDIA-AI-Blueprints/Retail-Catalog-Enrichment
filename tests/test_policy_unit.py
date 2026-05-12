# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for policy PDF extraction, summarization, and evaluation helpers.
"""
import json
from unittest.mock import Mock, patch

from backend.policy import extract_text_from_pdf_bytes, summarize_policy_document, evaluate_policy_compliance


class TestExtractTextFromPdfBytes:
    """Tests for extracting text from PDF bytes."""

    @patch("backend.policy.PdfReader")
    def test_extract_text_from_pdf_bytes_collects_non_empty_pages(self, mock_pdf_reader):
        mock_reader = Mock()
        page_one = Mock()
        page_one.extract_text.return_value = "First page"
        page_two = Mock()
        page_two.extract_text.return_value = ""
        page_three = Mock()
        page_three.extract_text.return_value = "Third page"
        mock_reader.pages = [page_one, page_two, page_three]
        mock_pdf_reader.return_value = mock_reader

        result = extract_text_from_pdf_bytes(b"%PDF-test")

        assert result == "First page\n\nThird page"


class TestPolicyModelCalls:
    """Tests for summarization and evaluation model helpers."""

    @patch("backend.policy.OpenAI")
    @patch("backend.policy.get_config")
    def test_summarize_policy_document_returns_structured_json(
        self,
        mock_get_config,
        mock_openai_class,
        mock_env_vars,
        sample_policy_summary,
    ):
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test-llm-model"}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(sample_policy_summary)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = summarize_policy_document("policy.pdf", "Example marketplace policy text.")

        assert result["document_name"] == "policy-a.pdf"
        assert result["blocking_rules"][0]["title"] == "Missing required compliance marker"

    @patch("backend.policy.OpenAI")
    @patch("backend.policy.get_config")
    def test_evaluate_policy_compliance_returns_structured_json(
        self,
        mock_get_config,
        mock_openai_class,
        mock_env_vars,
        sample_policy_decision,
        sample_policy_summary,
    ):
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test-llm-model"}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(sample_policy_decision)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = evaluate_policy_compliance(
            {"title": "Catalog Item", "description": "Generic product listing", "categories": ["bags"]},
            [sample_policy_summary],
        )

        assert result["status"] == "fail"
        assert result["matched_policies"][0]["document_name"] == "policy-a.pdf"

    @patch("backend.policy.OpenAI")
    @patch("backend.policy.get_config")
    def test_evaluate_policy_compliance_accepts_structured_fail_result(
        self,
        mock_get_config,
        mock_openai_class,
        mock_env_vars,
        sample_policy_summary,
    ):
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test-llm-model"}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        unsupported_fail = {
            "status": "fail",
            "label": "Policy Check Failed",
            "summary": "The product does not comply with the retrieved policy.",
            "matched_policies": [
                {
                    "document_name": "policy-a.pdf",
                    "policy_title": "Marketplace Policy A",
                    "rule_title": "Missing required compliance marker",
                    "reason": "Unsupported rationale.",
                    "evidence": ["invented phrase", "another invented phrase"]
                }
            ],
            "warnings": [],
            "evidence_note": "Unsupported fail."
        }

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(unsupported_fail)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = evaluate_policy_compliance(
            {"title": "Catalog Item", "description": "Generic product listing", "categories": ["bags"]},
            [sample_policy_summary],
        )

        assert result["status"] == "fail"
        assert result["matched_policies"][0]["rule_title"] == "Missing required compliance marker"

    @patch("backend.policy.OpenAI")
    @patch("backend.policy.get_config")
    def test_evaluate_policy_compliance_repairs_inconsistent_fail_without_matches(
        self,
        mock_get_config,
        mock_openai_class,
        mock_env_vars,
        sample_policy_summary,
    ):
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test-llm-model"}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        inconsistent_decision = {
            "status": "fail",
            "label": "Policy Check Failed",
            "summary": "The product does not match the retrieved policy.",
            "matched_policies": [],
            "warnings": ["Low-confidence policy evidence."],
            "evidence_note": "Candidate decision was inconsistent.",
        }
        repaired_decision = {
            "status": "pass",
            "label": "Policy Check Passed",
            "summary": "No retrieved policy blocks this product.",
            "matched_policies": [],
            "warnings": [],
            "evidence_note": "Decision based on the retrieved policy records.",
        }

        first_chunk = Mock()
        first_delta = Mock()
        first_delta.content = json.dumps(inconsistent_decision)
        first_choice = Mock()
        first_choice.delta = first_delta
        first_chunk.choices = [first_choice]

        second_chunk = Mock()
        second_delta = Mock()
        second_delta.content = json.dumps(repaired_decision)
        second_choice = Mock()
        second_choice.delta = second_delta
        second_chunk.choices = [second_choice]

        mock_client.chat.completions.create.side_effect = [[first_chunk], [second_chunk]]

        result = evaluate_policy_compliance(
            {"title": "Catalog Item", "description": "Generic product listing", "categories": ["bags"]},
            [sample_policy_summary],
        )

        assert result["status"] == "pass"
        assert result["matched_policies"] == []
        assert mock_client.chat.completions.create.call_count == 2
