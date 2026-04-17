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
Unit tests for the product manual module.

Tests text chunking, embedding-based retrieval, dynamic query generation,
and knowledge extraction with mocked external dependencies.
"""

import json
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from backend.product_manual import (
    _chunk_text,
    _embed_texts,
    _get_api_key,
    ProductManualContext,
    process_manual_pdf,
    generate_manual_queries,
    extract_manual_knowledge,
)


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    """Tests for the _chunk_text helper."""

    def test_empty_text_returns_empty_list(self):
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world this is a test"
        chunks = _chunk_text(text, chunk_size_words=100, overlap_words=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_splits_into_multiple_chunks(self):
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size_words=30, overlap_words=5)
        assert len(chunks) > 1
        # First chunk should have exactly 30 words
        assert len(chunks[0].split()) == 30

    def test_overlap_creates_shared_words(self):
        words = [f"w{i}" for i in range(50)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size_words=20, overlap_words=5)
        # The end of chunk 0 and the start of chunk 1 should share words
        c0_words = chunks[0].split()
        c1_words = chunks[1].split()
        overlap = set(c0_words[-5:]) & set(c1_words[:5])
        assert len(overlap) == 5

    def test_default_params(self):
        words = [f"w{i}" for i in range(600)]
        text = " ".join(words)
        chunks = _chunk_text(text)
        assert len(chunks) >= 2
        assert len(chunks[0].split()) == 250


# ---------------------------------------------------------------------------
# ProductManualContext
# ---------------------------------------------------------------------------

class TestEmbedTextsBatching:
    """Tests for embedding batching logic."""

    @patch("backend.product_manual.OpenAI")
    @patch("backend.product_manual.get_config")
    @patch("backend.product_manual._get_api_key", return_value="test-key")
    def test_batches_large_inputs(self, mock_key, mock_config, mock_openai):
        from backend.product_manual import _embed_texts, EMBEDDING_BATCH_SIZE

        mock_cfg = Mock()
        mock_cfg.get_embeddings_config.return_value = {"url": "http://test:8005/v1", "model": "test-embed"}
        mock_config.return_value = mock_cfg

        # Create a response mock that returns embeddings matching the input batch size
        def create_response(input_texts, **kwargs):
            mock_resp = Mock()
            mock_resp.data = [Mock(embedding=[0.1] * 4) for _ in input_texts]
            return mock_resp

        mock_openai.return_value.embeddings.create.side_effect = (
            lambda input, **kwargs: create_response(input, **kwargs)
        )

        # Send more texts than EMBEDDING_BATCH_SIZE
        n_texts = EMBEDDING_BATCH_SIZE + 10
        texts = [f"text {i}" for i in range(n_texts)]
        result = _embed_texts(texts, input_type="passage")

        assert len(result) == n_texts
        # Should have made 2 API calls (one full batch + one partial)
        assert mock_openai.return_value.embeddings.create.call_count == 2

    @patch("backend.product_manual.OpenAI")
    @patch("backend.product_manual.get_config")
    @patch("backend.product_manual._get_api_key", return_value="test-key")
    def test_single_batch_for_small_inputs(self, mock_key, mock_config, mock_openai):
        from backend.product_manual import _embed_texts

        mock_cfg = Mock()
        mock_cfg.get_embeddings_config.return_value = {"url": "http://test:8005/v1", "model": "test-embed"}
        mock_config.return_value = mock_cfg

        mock_resp = Mock()
        mock_resp.data = [Mock(embedding=[0.1] * 4) for _ in range(3)]
        mock_openai.return_value.embeddings.create.return_value = mock_resp

        result = _embed_texts(["a", "b", "c"], input_type="query")
        assert len(result) == 3
        assert mock_openai.return_value.embeddings.create.call_count == 1


class TestProductManualContext:
    """Tests for the ProductManualContext class."""

    def _make_context(self, n_chunks=5, dim=4):
        chunks = [f"chunk {i} content" for i in range(n_chunks)]
        # Deterministic vectors
        rng = np.random.RandomState(42)
        vectors = rng.randn(n_chunks, dim).tolist()
        return ProductManualContext(filename="test.pdf", chunks=chunks, vectors=vectors)

    def test_chunk_count(self):
        ctx = self._make_context(n_chunks=7)
        assert ctx.chunk_count == 7

    def test_retrieve_returns_strings(self):
        ctx = self._make_context()
        query_vec = [1.0, 0.0, 0.0, 0.0]
        results = ctx.retrieve(query_vec, top_k=2, min_score=-1.0)
        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)
        assert len(results) <= 2

    def test_retrieve_respects_top_k(self):
        ctx = self._make_context(n_chunks=10)
        query_vec = [1.0, 0.0, 0.0, 0.0]
        results = ctx.retrieve(query_vec, top_k=3, min_score=-1.0)
        assert len(results) <= 3

    def test_retrieve_filters_by_min_score(self):
        chunks = ["relevant", "irrelevant"]
        # Make the second vector orthogonal / low similarity
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        ctx = ProductManualContext(filename="t.pdf", chunks=chunks, vectors=vectors)
        results = ctx.retrieve([1.0, 0.0], top_k=2, min_score=0.9)
        assert len(results) == 1
        assert results[0] == "relevant"

    def test_retrieve_empty_context(self):
        ctx = ProductManualContext(filename="empty.pdf", chunks=[], vectors=[])
        results = ctx.retrieve([1.0, 0.0], top_k=3)
        assert results == []

    def test_filename_stored(self):
        ctx = self._make_context()
        assert ctx.filename == "test.pdf"


# ---------------------------------------------------------------------------
# process_manual_pdf
# ---------------------------------------------------------------------------

class TestProcessManualPdf:
    """Tests for process_manual_pdf with mocked dependencies."""

    @patch("backend.product_manual._embed_texts")
    @patch("backend.product_manual.extract_text_from_pdf_bytes")
    @patch("backend.product_manual.get_config")
    def test_returns_context_with_chunks_and_vectors(
        self, mock_config, mock_extract, mock_embed
    ):
        mock_cfg = Mock()
        mock_cfg.get_product_manual_config.return_value = {
            "chunk_size_words": 10,
            "chunk_overlap_words": 2,
            "top_k_per_query": 3,
            "min_relevance_score": 0.25,
        }
        mock_config.return_value = mock_cfg
        mock_extract.return_value = " ".join(f"word{i}" for i in range(30))
        mock_embed.return_value = [[0.1, 0.2, 0.3]] * 3  # 3 chunks

        ctx = process_manual_pdf(b"fake-pdf", "manual.pdf")

        assert isinstance(ctx, ProductManualContext)
        assert ctx.filename == "manual.pdf"
        assert ctx.chunk_count > 0
        mock_embed.assert_called_once()

    @patch("backend.product_manual.extract_text_from_pdf_bytes")
    @patch("backend.product_manual.get_config")
    def test_raises_on_empty_text(self, mock_config, mock_extract):
        mock_cfg = Mock()
        mock_cfg.get_product_manual_config.return_value = {
            "chunk_size_words": 250,
            "chunk_overlap_words": 50,
            "top_k_per_query": 3,
            "min_relevance_score": 0.25,
        }
        mock_config.return_value = mock_cfg
        mock_extract.return_value = ""

        with pytest.raises(ValueError, match="Unable to extract"):
            process_manual_pdf(b"fake-pdf", "empty.pdf")

    @patch("backend.product_manual.extract_text_from_pdf_bytes")
    @patch("backend.product_manual.get_config")
    def test_raises_on_short_text(self, mock_config, mock_extract):
        mock_cfg = Mock()
        mock_cfg.get_product_manual_config.return_value = {
            "chunk_size_words": 250,
            "chunk_overlap_words": 50,
            "top_k_per_query": 3,
            "min_relevance_score": 0.25,
        }
        mock_config.return_value = mock_cfg
        mock_extract.return_value = "short"

        with pytest.raises(ValueError, match="Unable to extract"):
            process_manual_pdf(b"fake-pdf", "short.pdf")


# ---------------------------------------------------------------------------
# generate_manual_queries
# ---------------------------------------------------------------------------

class TestGenerateManualQueries:
    """Tests for dynamic query generation via LLM."""

    @patch("backend.product_manual.OpenAI")
    @patch("backend.product_manual.get_config")
    @patch("backend.product_manual._get_api_key", return_value="test-key")
    def test_returns_list_of_query_dicts(self, mock_key, mock_config, mock_openai):
        mock_cfg = Mock()
        mock_cfg.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test"}
        mock_config.return_value = mock_cfg

        response_json = json.dumps([
            {"topic": "specs", "query": "What are the specifications?"},
            {"topic": "care", "query": "How to maintain?"},
        ])
        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content=response_json))]
        mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]

        queries = generate_manual_queries("Speaker", ["electronics"], "en-US")
        assert len(queries) == 2
        assert queries[0]["topic"] == "specs"
        assert "specifications" in queries[0]["query"]

    @patch("backend.product_manual.OpenAI")
    @patch("backend.product_manual.get_config")
    @patch("backend.product_manual._get_api_key", return_value="test-key")
    def test_returns_empty_on_bad_json(self, mock_key, mock_config, mock_openai):
        mock_cfg = Mock()
        mock_cfg.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test"}
        mock_config.return_value = mock_cfg

        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content="not valid json"))]
        mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]

        queries = generate_manual_queries("Speaker", ["electronics"])
        assert queries == []

    @patch("backend.product_manual.OpenAI")
    @patch("backend.product_manual.get_config")
    @patch("backend.product_manual._get_api_key", return_value="test-key")
    def test_empty_categories_uses_general_fallback(self, mock_key, mock_config, mock_openai):
        mock_cfg = Mock()
        mock_cfg.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test"}
        mock_config.return_value = mock_cfg

        response_json = json.dumps([{"topic": "specs", "query": "What specs?"}])
        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content=response_json))]
        mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]

        queries = generate_manual_queries("Widget", [], "en-US")
        assert len(queries) == 1

        # Verify "general" was used as category fallback in the prompt
        call_args = mock_openai.return_value.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert "general" in prompt

    @patch("backend.product_manual.OpenAI")
    @patch("backend.product_manual.get_config")
    @patch("backend.product_manual._get_api_key", return_value="test-key")
    def test_unsupported_locale_falls_back_to_english(self, mock_key, mock_config, mock_openai):
        mock_cfg = Mock()
        mock_cfg.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test"}
        mock_config.return_value = mock_cfg

        response_json = json.dumps([{"topic": "specs", "query": "What are the specs?"}])
        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content=response_json))]
        mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]

        queries = generate_manual_queries("Widget", ["toys"], "xx-XX")
        assert len(queries) == 1

        call_args = mock_openai.return_value.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert "English" in prompt

    @patch("backend.product_manual.OpenAI")
    @patch("backend.product_manual.get_config")
    @patch("backend.product_manual._get_api_key", return_value="test-key")
    def test_filters_malformed_entries(self, mock_key, mock_config, mock_openai):
        mock_cfg = Mock()
        mock_cfg.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test"}
        mock_config.return_value = mock_cfg

        response_json = json.dumps([
            {"topic": "specs", "query": "Valid query"},
            {"bad_key": "no topic or query"},
            "just a string",
        ])
        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content=response_json))]
        mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]

        queries = generate_manual_queries("Dress", ["clothing"])
        assert len(queries) == 1
        assert queries[0]["topic"] == "specs"


# ---------------------------------------------------------------------------
# extract_manual_knowledge
# ---------------------------------------------------------------------------

class TestExtractManualKnowledge:
    """Tests for targeted knowledge extraction."""

    @patch("backend.product_manual._embed_texts")
    @patch("backend.product_manual.get_config")
    def test_returns_dict_keyed_by_topic(self, mock_config, mock_embed):
        mock_cfg = Mock()
        mock_cfg.get_product_manual_config.return_value = {
            "chunk_size_words": 250,
            "chunk_overlap_words": 50,
            "top_k_per_query": 2,
            "min_relevance_score": -1.0,  # accept all
        }
        mock_config.return_value = mock_cfg

        # Build a context with known vectors
        chunks = ["battery info here", "safety warnings here", "care instructions"]
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        ctx = ProductManualContext(filename="t.pdf", chunks=chunks, vectors=vectors)

        # Mock embedding for queries: first query vector aligns with first chunk
        mock_embed.side_effect = [
            [[1.0, 0.0]],  # battery query
            [[0.0, 1.0]],  # safety query
        ]

        queries = [
            {"topic": "battery", "query": "Battery life?"},
            {"topic": "safety", "query": "Safety warnings?"},
        ]

        knowledge = extract_manual_knowledge(ctx, queries)

        assert "battery" in knowledge
        assert "safety" in knowledge
        assert "battery info" in knowledge["battery"]
        assert "safety warnings" in knowledge["safety"]

    @patch("backend.product_manual._embed_texts")
    @patch("backend.product_manual.get_config")
    def test_deduplicates_chunks_across_queries(self, mock_config, mock_embed):
        mock_cfg = Mock()
        mock_cfg.get_product_manual_config.return_value = {
            "chunk_size_words": 250,
            "chunk_overlap_words": 50,
            "top_k_per_query": 3,
            "min_relevance_score": -1.0,
        }
        mock_config.return_value = mock_cfg

        chunks = ["shared chunk", "unique chunk"]
        vectors = [[1.0, 0.0], [0.5, 0.5]]
        ctx = ProductManualContext(filename="t.pdf", chunks=chunks, vectors=vectors)

        # Both queries get the same top vector
        mock_embed.side_effect = [
            [[1.0, 0.0]],
            [[0.9, 0.1]],
        ]

        queries = [
            {"topic": "a", "query": "First query"},
            {"topic": "b", "query": "Second query"},
        ]

        knowledge = extract_manual_knowledge(ctx, queries)

        # "shared chunk" should appear in topic "a" but not duplicated in "b"
        assert "shared chunk" in knowledge["a"]

    @patch("backend.product_manual._embed_texts")
    @patch("backend.product_manual.get_config")
    def test_empty_knowledge_when_no_embedding(self, mock_config, mock_embed):
        mock_cfg = Mock()
        mock_cfg.get_product_manual_config.return_value = {
            "chunk_size_words": 250,
            "chunk_overlap_words": 50,
            "top_k_per_query": 3,
            "min_relevance_score": 0.25,
        }
        mock_config.return_value = mock_cfg

        ctx = ProductManualContext(filename="t.pdf", chunks=["some text"], vectors=[[1.0, 0.0]])
        mock_embed.return_value = []  # embedding fails

        queries = [{"topic": "specs", "query": "What specs?"}]
        knowledge = extract_manual_knowledge(ctx, queries)

        assert knowledge["specs"] == ""


# ---------------------------------------------------------------------------
# _format_manual_knowledge (tested via vlm.py)
# ---------------------------------------------------------------------------

class TestFormatManualKnowledge:
    """Tests for _format_manual_knowledge in vlm.py."""

    def test_formats_non_empty_topics(self):
        from backend.vlm import _format_manual_knowledge
        knowledge = {
            "battery_life": "Lasts up to 12 hours on a single charge.",
            "safety": "Do not immerse in water above 1 meter.",
            "empty_topic": "",
        }
        result = _format_manual_knowledge(knowledge)
        assert "PRODUCT MANUAL KNOWLEDGE:" in result
        assert "[Battery Life]" in result
        assert "12 hours" in result
        assert "[Safety]" in result
        assert "[Empty Topic]" not in result

    def test_all_empty_topics(self):
        from backend.vlm import _format_manual_knowledge
        knowledge = {"a": "", "b": "   "}
        result = _format_manual_knowledge(knowledge)
        assert "PRODUCT MANUAL KNOWLEDGE:" in result
        # No topic sections should appear
        assert "[A]" not in result
        assert "[B]" not in result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestProductManualConfig:
    """Tests for product manual config loading."""

    def test_get_product_manual_config_defaults(self, tmp_path):
        import yaml
        from backend.config import Config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"vlm": {"url": "http://x", "model": "m"}}))
        config = Config(config_path=str(config_file))
        pm_config = config.get_product_manual_config()

        assert pm_config["chunk_size_words"] == 250
        assert pm_config["chunk_overlap_words"] == 50
        assert pm_config["top_k_per_query"] == 3
        assert pm_config["min_relevance_score"] == 0.25

    def test_get_product_manual_config_custom(self, tmp_path):
        import yaml
        from backend.config import Config

        data = {
            "vlm": {"url": "http://x", "model": "m"},
            "product_manual": {
                "chunk_size_words": 300,
                "chunk_overlap_words": 75,
                "top_k_per_query": 5,
                "min_relevance_score": 0.4,
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(data))
        config = Config(config_path=str(config_file))
        pm_config = config.get_product_manual_config()

        assert pm_config["chunk_size_words"] == 300
        assert pm_config["chunk_overlap_words"] == 75
        assert pm_config["top_k_per_query"] == 5
        assert pm_config["min_relevance_score"] == 0.4


# ---------------------------------------------------------------------------
# FAQ generation with manual knowledge (via vlm.py)
# ---------------------------------------------------------------------------

class TestFaqGenerationWithManual:
    """Test that _call_nemotron_generate_faqs correctly uses manual knowledge."""

    @patch("backend.vlm.OpenAI")
    @patch("backend.vlm.get_config")
    def test_prompt_includes_manual_section_when_provided(
        self, mock_config, mock_openai, mock_env_vars
    ):
        mock_cfg = Mock()
        mock_cfg.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test"}
        mock_config.return_value = mock_cfg

        faqs_json = json.dumps([
            {"question": "What is the battery life?", "answer": "Up to 12 hours."},
        ])
        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content=faqs_json))]
        mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]

        manual_knowledge = {
            "battery_life": "The speaker lasts up to 12 hours on a single charge.",
        }

        from backend.vlm import _call_nemotron_generate_faqs
        result = _call_nemotron_generate_faqs(
            {"title": "Bluetooth Speaker", "description": "Portable speaker"},
            locale="en-US",
            manual_knowledge=manual_knowledge,
        )

        assert len(result) == 1
        assert "battery" in result[0]["question"].lower()

        # Verify the prompt contained manual knowledge
        call_args = mock_openai.return_value.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert "PRODUCT MANUAL KNOWLEDGE:" in prompt
        assert "Battery Life" in prompt
        assert "12 hours" in prompt

    @patch("backend.vlm.OpenAI")
    @patch("backend.vlm.get_config")
    def test_no_manual_section_when_not_provided(
        self, mock_config, mock_openai, mock_env_vars
    ):
        mock_cfg = Mock()
        mock_cfg.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test"}
        mock_config.return_value = mock_cfg

        faqs_json = json.dumps([
            {"question": "What material?", "answer": "Leather."},
        ])
        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content=faqs_json))]
        mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]

        from backend.vlm import _call_nemotron_generate_faqs
        result = _call_nemotron_generate_faqs(
            {"title": "Handbag", "description": "Leather bag"},
            locale="en-US",
        )

        call_args = mock_openai.return_value.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert "PRODUCT MANUAL KNOWLEDGE:" not in prompt
        assert "3 to 5" in prompt

    @patch("backend.vlm.OpenAI")
    @patch("backend.vlm.get_config")
    def test_empty_manual_knowledge_uses_basic_prompt(
        self, mock_config, mock_openai, mock_env_vars
    ):
        """manual_knowledge with all-empty values should use the basic (no-manual) prompt."""
        mock_cfg = Mock()
        mock_cfg.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test"}
        mock_config.return_value = mock_cfg

        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content='[{"question":"Q","answer":"A"}]'))]
        mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]

        from backend.vlm import _call_nemotron_generate_faqs
        _call_nemotron_generate_faqs(
            {"title": "X", "description": "Y"},
            manual_knowledge={"specs": "", "care": "   "},
        )
        call_args = mock_openai.return_value.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert "PRODUCT MANUAL KNOWLEDGE:" not in prompt
        assert "3 to 5" in prompt
        assert call_args.kwargs["max_tokens"] == 2048

    @patch("backend.vlm.OpenAI")
    @patch("backend.vlm.get_config")
    def test_max_tokens_higher_with_manual(
        self, mock_config, mock_openai, mock_env_vars
    ):
        mock_cfg = Mock()
        mock_cfg.get_llm_config.return_value = {"url": "http://test:8000/v1", "model": "test"}
        mock_config.return_value = mock_cfg

        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content="[]"))]
        mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]

        from backend.vlm import _call_nemotron_generate_faqs

        # With manual
        _call_nemotron_generate_faqs(
            {"title": "X", "description": "Y"},
            manual_knowledge={"specs": "Some specs here"},
        )
        call_with = mock_openai.return_value.chat.completions.create.call_args
        assert call_with.kwargs["max_tokens"] == 4096

        # Without manual
        _call_nemotron_generate_faqs(
            {"title": "X", "description": "Y"},
        )
        call_without = mock_openai.return_value.chat.completions.create.call_args
        assert call_without.kwargs["max_tokens"] == 2048
