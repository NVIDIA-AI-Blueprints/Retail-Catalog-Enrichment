# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Unit tests for VLM module with mocked OpenAI API calls.

Tests VLM analysis, enhancement, and branding functions without external dependencies.
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.vlm import (
    _call_vlm,
    _call_nemotron_structure_vlm,
    _call_nemotron_filter_user_data,
    _call_nemotron_enhance_vlm,
    _call_nemotron_resolve_merge_conflicts,
    _call_nemotron_apply_branding,
    _call_nemotron_generate_faqs,
    _call_nemotron_enhance,
    _repair_visual_identity_regression,
    extract_vlm_observation,
    build_enriched_vlm_result,
    run_vlm_analysis
)


class TestCallVLM:
    """Tests for _call_vlm function with mocked VLM + structuring."""

    @patch('backend.vlm._call_nemotron_structure_vlm')
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_call_vlm_passes_free_text_to_structuring(self, mock_get_config, mock_openai_class, mock_structure, sample_image_bytes, sample_vlm_response, mock_env_vars):
        """Test that VLM free text is passed to the structuring LLM call."""
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        vlm_free_text = "A black and red Craftsman lawn mower with 2XV20 printed on the deck."
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = vlm_free_text
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        mock_structure.return_value = sample_vlm_response

        result = _call_vlm(sample_image_bytes, "image/png", "en-US")

        mock_structure.assert_called_once_with(vlm_free_text, "en-US")
        assert result == sample_vlm_response

    @patch('backend.vlm._call_nemotron_structure_vlm')
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_call_vlm_uses_short_prompt(self, mock_get_config, mock_openai_class, mock_structure, sample_image_bytes, sample_vlm_response, mock_env_vars):
        """Test that the VLM prompt is short (not the old ~35 line prompt)."""
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = "A product description"
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]
        mock_structure.return_value = sample_vlm_response

        _call_vlm(sample_image_bytes, "image/png")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["content"] == "/no_think"
        prompt_text = messages[1]["content"][1]["text"]
        assert len(prompt_text) < 300
        assert "Describe only visible facts" in prompt_text
        assert "Include numbers/specs only if clearly readable as printed text" in prompt_text
        assert "never infer capacity, size, model, power, weight, or volume" in prompt_text

    @patch('backend.vlm._call_nemotron_structure_vlm')
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_call_vlm_with_different_image_types(self, mock_get_config, mock_openai_class, mock_structure, sample_jpeg_bytes, sample_vlm_response, mock_env_vars):
        """Test VLM call with different image content types."""
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = "A product"
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]
        mock_structure.return_value = sample_vlm_response

        result = _call_vlm(sample_jpeg_bytes, "image/jpeg")
        assert isinstance(result, dict)


class TestCallNemotronStructureVlm:
    """Tests for _call_nemotron_structure_vlm function."""

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_structure_success(self, mock_get_config, mock_openai_class, sample_vlm_response, mock_env_vars):
        """Test successful structuring of free text into JSON."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-llm-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(sample_vlm_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = _call_nemotron_structure_vlm("A black handbag with gold accents.")

        assert isinstance(result, dict)
        assert result["title"] == sample_vlm_response["title"]
        assert "description" in result

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "Do NOT state capacity, dimensions, volume, weight, power rating" in prompt
        assert "readable printed text" in prompt
        assert "If the visual description mentions a number/spec but does not say it is readable printed text, omit it" in prompt
        assert "Clear catalog title, not creative copy" in prompt
        assert "Do NOT use size/weight claims like compact" in prompt
        assert "ALLOWED COLORS" in prompt
        assert "Do not output materials, finishes, textures, or product types as colors" in prompt
        assert "Do not include packaging/container appearance such as cap color" in prompt
        assert "official product variant or necessary retail differentiator" in prompt
        assert "Use established retail terminology for the target locale" not in prompt
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.0
        assert call_args.kwargs["top_p"] == 1

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_structure_non_english_prompt_adds_terminology_rule(self, mock_get_config, mock_openai_class, sample_vlm_response, mock_env_vars):
        """Test localized terminology guard is added only for non-English output."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-llm-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(sample_vlm_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        _call_nemotron_structure_vlm("A black air fryer.", "es-AR")

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "Use established retail terminology for the target locale" in prompt
        assert "The visual analysis may be in English" in prompt
        assert "translate generic product-type nouns from the visual analysis" in prompt
        assert "English generic product-type nouns are not allowed" in prompt
        assert "Do not keep English generic product-type nouns just because they appear in the visual analysis" in prompt
        assert "readable label text" in prompt
        assert "Do not invent new compound words, calques, or phonetic translations" in prompt
        assert "never coin or merge words to translate a product type" in prompt
        assert "use a common generic product term in the target language instead of inventing one" in prompt
        assert "readable English label text does not override the localized generic product type" in prompt
        assert "self-check title and description" in prompt
        assert "Do not copy visible English generic product-type label text as the localized product type" in prompt
        assert "LOCALIZATION CHECK" in prompt
        assert "Title and description are invalid if they keep English generic product-type nouns" in prompt
        assert "rewrite any remaining English generic product-type noun into Spanish" in prompt

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_structure_fallback_on_parse_failure(self, mock_get_config, mock_openai_class, mock_env_vars):
        """Test fallback to raw text when LLM returns unparseable output."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-llm-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = "Not valid JSON at all"
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        vlm_text = "A red lawn mower with Craftsman branding."
        result = _call_nemotron_structure_vlm(vlm_text)

        assert result["title"] == ""
        assert result["description"] == vlm_text
        assert result["categories"] == ["uncategorized"]

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_structure_extracts_from_markdown(self, mock_get_config, mock_openai_class, sample_vlm_response, mock_env_vars):
        """Test extraction from markdown-fenced JSON."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-llm-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        wrapped = f"```json\n{json.dumps(sample_vlm_response)}\n```"
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = wrapped
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = _call_nemotron_structure_vlm("A handbag.")

        assert result["title"] == sample_vlm_response["title"]

    def test_structure_raises_without_api_key(self, monkeypatch):
        """Test RuntimeError when NGC_API_KEY is not set."""
        monkeypatch.delenv("NGC_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="NGC_API_KEY is not set"):
            _call_nemotron_structure_vlm("Some text")


class TestCallNemotronFilterUserData:
    """Tests for contradiction-aware user data filtering before merge."""

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_filter_user_data_allows_term_level_cleanup_for_label_conflicts(self, mock_get_config, mock_openai_class, mock_env_vars):
        """Test prompt supports removing only conflicting user terms when label text disagrees."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        cleaned_product_data = {
            "title": "Example Brand",
            "description": "A supplement from Example Brand.",
            "price": 12.99,
            "sku": "SUP-001",
        }

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(cleaned_product_data)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        vlm_output = {
            "title": "Example Brand Omega Softgels",
            "description": "Bottle label reads Example Brand Omega softgels.",
            "categories": ["skincare"],
            "tags": ["omega", "softgels"],
            "colors": ["yellow"],
        }
        product_data = {
            "title": "Example Brand Mineral",
            "description": "A mineral supplement from Example Brand.",
            "price": 12.99,
            "sku": "SUP-001",
        }

        result = _call_nemotron_filter_user_data(vlm_output, product_data)

        assert result == cleaned_product_data
        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "partially correct, edit that field minimally" in prompt
        assert "remove only the conflicting terms" in prompt
        assert "Readable label text is authoritative for product names, active ingredients" in prompt
        assert "differs from readable label text or the visually identified product type" in prompt
        assert "Do not combine two conflicting product identities" in prompt
        assert "For non-text fields (price, SKU, numeric values): always keep unchanged" in prompt
        assert "This is a binary decision per field" not in prompt
        assert "Never partially edit" not in prompt


class TestCallNemotronResolveMergeConflicts:
    """Tests for final merge QA validation."""

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_resolve_merge_conflicts_removes_surviving_identity_conflicts(self, mock_get_config, mock_openai_class, mock_env_vars):
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        repaired_content = {
            "title": "Example Brand Omega Oil Softgels",
            "description": "Example Brand omega oil softgels with readable dosage and count details.",
            "categories": ["uncategorized"],
            "tags": ["example brand", "omega oil", "softgels"],
            "colors": ["yellow", "brown"],
        }

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(repaired_content)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        vlm_output = {
            "title": "Example Brand Omega Oil Softgels",
            "description": "Readable label text says Example Brand Omega Oil.",
            "categories": ["uncategorized"],
            "tags": ["omega oil", "softgels"],
            "colors": ["yellow", "brown"],
        }
        product_data = {"title": "Example Brand"}
        merged_content = {
            "title": "Example Brand Mineral Softgels",
            "description": "Example Brand mineral softgels.",
            "categories": ["uncategorized"],
            "tags": ["example brand", "mineral", "softgels"],
            "colors": ["yellow", "brown"],
        }

        result = _call_nemotron_resolve_merge_conflicts(vlm_output, product_data, merged_content, "en-US")

        assert result == repaired_content
        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "product catalog merge QA validator" in prompt
        assert "MERGED CATALOG CONTENT TO VALIDATE" in prompt
        assert "Readable label text is authoritative for product names, active ingredients" in prompt
        assert "remove it or replace it with the exact supported visual/readable-label term" in prompt
        assert "Do not combine two conflicting product identities" in prompt
        assert "Do not remove a term merely because it is absent from the image" in prompt
        assert "Remove packaging/container appearance from title" in prompt
        assert "cap color, bottle color, box color, label color" in prompt


class TestVisualIdentityRegressionRepair:
    """Tests for deterministic fallback when merge QA keeps stale identity terms."""

    def test_repair_restores_visual_identity_when_title_regresses_to_user_only_terms(self):
        vlm_output = {
            "title": "Example Brand Omega Oil Softgels 300 Count",
            "description": "Example Brand Omega Oil softgels with 300 count visible on the label.",
            "categories": ["uncategorized"],
            "tags": ["omega oil", "softgels", "300 count", "dietary supplement"],
            "colors": ["yellow", "brown"],
        }
        product_data = {
            "title": "Example Brand Mineral",
            "description": "Example Brand mineral supplement for immune support.",
            "tags": ["mineral", "immune support"],
            "price": 12.99,
        }
        merged_content = {
            "title": "Example Brand Mineral Softgel Supplement",
            "description": "Example Brand mineral supplement in softgel form.",
            "categories": ["uncategorized"],
            "tags": ["mineral", "immune support", "softgel", "dietary supplement"],
            "colors": ["yellow", "brown"],
        }

        result = _repair_visual_identity_regression(vlm_output, product_data, merged_content)

        assert result["title"] == vlm_output["title"]
        assert result["description"] == vlm_output["description"]
        assert "mineral" not in result["tags"]
        assert "immune support" not in result["tags"]
        assert "omega oil" in result["tags"]
        assert result["categories"] == merged_content["categories"]
        assert result["colors"] == merged_content["colors"]

    def test_repair_leaves_non_conflicting_hidden_user_specs_when_visual_identity_is_present(self):
        vlm_output = {
            "title": "Example Brand Trail Running Shoes",
            "description": "Example Brand trail running shoes with a textured outsole.",
            "tags": ["trail running", "shoes", "textured outsole"],
        }
        product_data = {
            "title": "Example Brand Waterproof Shoes",
            "description": "Waterproof trail footwear.",
            "tags": ["waterproof"],
        }
        merged_content = {
            "title": "Example Brand Waterproof Trail Running Shoes",
            "description": "Waterproof trail running shoes with a textured outsole.",
            "tags": ["waterproof", "trail running", "shoes"],
        }

        result = _repair_visual_identity_regression(vlm_output, product_data, merged_content)

        assert result == merged_content


class TestCallNemotronEnhanceVLM:
    """Tests for _call_nemotron_enhance_vlm function."""
    
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_enhance_vlm_output_without_product_data(self, mock_get_config, mock_openai_class, sample_vlm_response, mock_env_vars):
        """Test enhancement without existing product data."""
        # Mock config
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        enhanced_response = {
            "title": "Enhanced Title",
            "description": "Enhanced Description",
            "categories": ["bags"],
            "tags": ["enhanced", "tags"],
            "colors": ["black", "gold"]
        }
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(enhanced_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Call function
        result = _call_nemotron_enhance_vlm(sample_vlm_response, None, "en-US")
        
        # Assertions
        assert isinstance(result, dict)
        assert result["title"] == "Enhanced Title"
        assert result["description"] == "Enhanced Description"
    
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_enhance_vlm_with_product_data(self, mock_get_config, mock_openai_class, sample_vlm_response, sample_product_data, mock_env_vars):
        """Test enhancement with existing product data (augmentation mode)."""
        # Mock config
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        enhanced_response = {
            "title": "Enhanced Augmented Title",
            "description": "Enhanced augmented description",
            "price": 15.99,  # Preserved from original
            "categories": ["bags"],
            "tags": ["enhanced", "augmented"],
            "colors": ["black", "gold"],
            "sku": "BAG-001"  # Preserved from original
        }
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(enhanced_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Call function
        result = _call_nemotron_enhance_vlm(sample_vlm_response, sample_product_data, "en-US")
        
        # Assertions
        assert isinstance(result, dict)
        assert "price" in result  # Should preserve original fields
        assert "sku" in result

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_enhance_vlm_prompt_requires_richer_augmented_title(self, mock_get_config, mock_openai_class, sample_vlm_response, mock_env_vars):
        """Test augmentation prompt tells the LLM to enrich, not copy, user titles."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        enhanced_response = {
            "title": "Sport Sneakers with White Finish and Black Accents",
            "description": "Enhanced product description",
            "categories": ["footwear"],
            "tags": ["sneakers", "sport"],
            "colors": ["white", "black"]
        }

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(enhanced_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        product_data = {
            "title": "Sport sneakers",
            "description": "Comfortable shoes",
            "categories": ["footwear"],
            "tags": ["sneakers"]
        }

        _call_nemotron_enhance_vlm(sample_vlm_response, product_data, "en-US")

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "Add only title-worthy facts from the VISUAL ANALYSIS" in prompt
        assert "not identical to, the user-provided title" in prompt
        assert "Treat the remaining user title terms as validated anchors" in prompt
        assert "If readable label text identifies a product name, active ingredient" in prompt
        assert "it overrides any remaining conflicting user title term" in prompt
        assert "Do not combine conflicting product identities in the final title" in prompt
        assert "filtered user-provided title words are validated anchors" in prompt
        assert "Do not add packaging/container appearance such as cap color" in prompt
        assert "official product variant or necessary to distinguish the sold product" in prompt
        assert "Do not replace user title words with unrelated synonyms" in prompt
        assert "Do not state measurable specs such as capacity, dimensions" in prompt
        assert "Do not use size/weight claims such as compact" in prompt
        assert "ALLOWED COLORS" in prompt
        assert "Do not output materials, finishes, textures, or product types as colors" in prompt
        assert "Use established retail terminology for the target locale" not in prompt
    
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_enhance_vlm_with_different_locales(self, mock_get_config, mock_openai_class, sample_vlm_response, mock_env_vars):
        """Test enhancement with different locales."""
        # Mock config
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Spanish response
        spanish_response = {
            "title": "Bolso Negro Elegante con Detalles Dorados",
            "description": "Un bolso sofisticado de cuero...",
            "categories": ["bags"],
            "tags": ["cuero negro", "herrajes dorados"],
            "colors": ["black", "gold"]
        }
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(spanish_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Call function with Spanish locale
        result = _call_nemotron_enhance_vlm(sample_vlm_response, None, "es-ES")
        
        # Should contain localized content
        assert isinstance(result, dict)
        assert result["title"] == spanish_response["title"]
        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "Use established retail terminology for the target locale" in prompt
        assert "English generic product-type nouns are not allowed" in prompt
        assert "Do not invent new compound words, calques, or phonetic translations" in prompt
    
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_enhance_vlm_json_extraction_from_markdown(self, mock_get_config, mock_openai_class, sample_vlm_response, mock_env_vars):
        """Test JSON extraction when wrapped in markdown code blocks."""
        # Mock config
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        enhanced_response = {
            "title": "Test Title",
            "description": "Test Description",
            "categories": ["test"],
            "tags": ["test"],
            "colors": ["test"]
        }
        
        # Wrap JSON in markdown
        markdown_response = f"```json\n{json.dumps(enhanced_response)}\n```"
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = markdown_response
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Call function
        result = _call_nemotron_enhance_vlm(sample_vlm_response, None, "en-US")
        
        # Should extract JSON from markdown
        assert isinstance(result, dict)
        assert result["title"] == "Test Title"


class TestCallNemotronApplyBranding:
    """Tests for _call_nemotron_apply_branding function."""
    
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_apply_branding_success(self, mock_get_config, mock_openai_class, sample_enhanced_product, mock_env_vars):
        """Test successful brand application."""
        # Mock config
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        branded_response = {
            "title": "Brand-Aligned Title",
            "description": "Brand-aligned description with brand voice",
            "price": 15.99,
            "categories": ["bags"],
            "tags": ["brand", "aligned"],
            "colors": ["black", "gold"],
            "sku": "BAG-001"
        }
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(branded_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        brand_instructions = "Use playful and empowering tone. Focus on self-expression."
        
        # Call function
        result = _call_nemotron_apply_branding(sample_enhanced_product, brand_instructions, "en-US")
        
        # Assertions
        assert isinstance(result, dict)
        assert result["title"] == "Brand-Aligned Title"
        assert "price" in result  # Should preserve structure
    
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_apply_branding_preserves_structure(self, mock_get_config, mock_openai_class, sample_enhanced_product, mock_env_vars):
        """Test that branding preserves exact JSON structure."""
        # Mock config
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Return same structure with modified values
        branded_response = sample_enhanced_product.copy()
        branded_response["title"] = "Branded Title"
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(branded_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        brand_instructions = "Professional tone"
        
        # Call function
        result = _call_nemotron_apply_branding(sample_enhanced_product, brand_instructions, "en-US")
        
        # Should have same keys as input
        assert set(result.keys()) == set(sample_enhanced_product.keys())

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_apply_branding_locks_output_language_for_spanish_brand_instructions(self, mock_get_config, mock_openai_class, sample_enhanced_product, mock_env_vars):
        """Test brand instructions cannot override the selected output locale."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-llm-model'
        }
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        branded_response = sample_enhanced_product.copy()
        branded_response["description"] = "Descripción de lujo en español argentino."

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(branded_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        _call_nemotron_apply_branding(
            sample_enhanced_product,
            "utiliza palabras de lujo para describir el producto",
            "es-AR",
        )

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "OUTPUT LANGUAGE LOCK" in prompt
        assert "Title and description must remain in Spanish for Argentina" in prompt
        assert "Brand instructions may be written in any language" in prompt
        assert "do not infer the output language from them" in prompt
        assert "richer, longer, more detailed" in prompt
        assert "Add 1-3 additional sentences" in prompt
        assert "safely expand only what is already there" in prompt
        assert "Use established retail terminology for the target locale" in prompt
        assert "English generic product-type nouns are not allowed" in prompt
        assert "Do not invent new compound words, calques, or phonetic translations" in prompt
        assert "readable English label text does not override the localized generic product type" in prompt
        assert "Do NOT add new measurable specs such as capacity, dimensions" in prompt
        assert "Do NOT add size/weight claims such as compact" in prompt


class TestCallNemotronGenerateFaqs:
    """Tests for _call_nemotron_generate_faqs function."""

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_generate_faqs_success(self, mock_get_config, mock_openai_class, sample_vlm_response, sample_faqs_response, mock_env_vars):
        """Test successful FAQ generation with valid JSON array."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-llm-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(sample_faqs_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = _call_nemotron_generate_faqs(sample_vlm_response, "en-US")

        assert isinstance(result, list)
        assert len(result) == 3
        assert all("question" in faq and "answer" in faq for faq in result)
        mock_client.chat.completions.create.assert_called_once()
        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "Use established retail terminology for the target locale" not in prompt

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_generate_faqs_empty_on_parse_failure(self, mock_get_config, mock_openai_class, sample_vlm_response, mock_env_vars):
        """Test that non-JSON response returns empty list."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-llm-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = "This is not valid JSON at all"
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = _call_nemotron_generate_faqs(sample_vlm_response, "en-US")

        assert result == []

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_generate_faqs_with_markdown_wrapped_response(self, mock_get_config, mock_openai_class, sample_vlm_response, sample_faqs_response, mock_env_vars):
        """Test extraction of FAQs from markdown-fenced JSON."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-llm-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        wrapped_content = f"```json\n{json.dumps(sample_faqs_response)}\n```"
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = wrapped_content
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = _call_nemotron_generate_faqs(sample_vlm_response, "en-US")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["question"] == sample_faqs_response[0]["question"]

    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_generate_faqs_with_locale(self, mock_get_config, mock_openai_class, sample_vlm_response, sample_faqs_response, mock_env_vars):
        """Test FAQ generation with non-English locale."""
        mock_config = Mock()
        mock_config.get_llm_config.return_value = {'url': 'http://test:8000/v1', 'model': 'test-llm-model'}
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(sample_faqs_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = _call_nemotron_generate_faqs(sample_vlm_response, "es-ES")

        assert isinstance(result, list)
        assert len(result) == 3
        # Verify the prompt included Spanish locale context
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert "Spanish" in prompt
        assert "Spain" in prompt
        assert "Use established retail terminology for the target locale" in prompt
        assert "English generic product-type nouns are not allowed" in prompt
        assert "Do not invent new compound words, calques, or phonetic translations" in prompt

    def test_generate_faqs_raises_without_api_key(self, sample_vlm_response, monkeypatch):
        """Test RuntimeError when NGC_API_KEY is not set."""
        monkeypatch.delenv("NGC_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="NGC_API_KEY is not set"):
            _call_nemotron_generate_faqs(sample_vlm_response, "en-US")


class TestCallNemotronEnhance:
    """Tests for _call_nemotron_enhance orchestration function."""

    @patch('backend.vlm._call_nemotron_apply_branding')
    @patch('backend.vlm._call_nemotron_enhance_vlm')
    def test_enhance_skips_step1_without_product_data(self, mock_enhance_vlm, mock_apply_branding, sample_vlm_response):
        """Test that Step 1 is skipped when no product_data — VLM output used directly."""
        result = _call_nemotron_enhance(sample_vlm_response, None, "en-US", None)

        # Step 1 should be SKIPPED (no product data to merge)
        mock_enhance_vlm.assert_not_called()
        # Step 2 should NOT be called
        mock_apply_branding.assert_not_called()
        assert result == sample_vlm_response

    @patch('backend.vlm._call_nemotron_apply_branding')
    @patch('backend.vlm._call_nemotron_enhance_vlm')
    def test_enhance_with_brand_instructions_skips_step1(self, mock_enhance_vlm, mock_apply_branding, sample_vlm_response):
        """Test that Step 1 is skipped but Step 2 runs on raw VLM output when only brand instructions provided."""
        branded_data = {"title": "Branded", "description": "Branded"}
        mock_apply_branding.return_value = branded_data

        brand_instructions = "Use playful tone"
        result = _call_nemotron_enhance(sample_vlm_response, None, "en-US", brand_instructions)

        # Step 1 should be SKIPPED (no product data)
        mock_enhance_vlm.assert_not_called()
        # Step 2 should run on raw VLM output
        mock_apply_branding.assert_called_once_with(sample_vlm_response, brand_instructions, "en-US")
        assert result == branded_data

    @patch('backend.vlm._call_nemotron_resolve_merge_conflicts')
    @patch('backend.vlm._call_nemotron_filter_user_data')
    @patch('backend.vlm._call_nemotron_apply_branding')
    @patch('backend.vlm._call_nemotron_enhance_vlm')
    def test_enhance_runs_step1_with_product_data(self, mock_enhance_vlm, mock_apply_branding, mock_filter, mock_repair, sample_vlm_response, sample_product_data):
        """Test that Step 1 runs when product_data is provided."""
        enhanced_data = {"title": "Enhanced", "description": "Enhanced"}
        mock_filter.return_value = sample_product_data
        mock_enhance_vlm.return_value = enhanced_data
        mock_repair.return_value = enhanced_data

        result = _call_nemotron_enhance(sample_vlm_response, sample_product_data, "en-US", None)

        # Pre-filter and Step 1 should run
        mock_filter.assert_called_once()
        mock_enhance_vlm.assert_called_once()
        # Step 2 should NOT run
        mock_apply_branding.assert_not_called()
        mock_repair.assert_called_once_with(sample_vlm_response, sample_product_data, enhanced_data, "en-US")
        assert result == enhanced_data

    @patch('backend.vlm._call_nemotron_resolve_merge_conflicts')
    @patch('backend.vlm._call_nemotron_filter_user_data')
    @patch('backend.vlm._call_nemotron_apply_branding')
    @patch('backend.vlm._call_nemotron_enhance_vlm')
    def test_enhance_runs_full_pipeline_with_product_data_and_brand(self, mock_enhance_vlm, mock_apply_branding, mock_filter, mock_repair, sample_vlm_response, sample_product_data):
        """Test full pipeline (Step 1 + Step 2) when both product_data and brand_instructions provided."""
        enhanced_data = {"title": "Enhanced", "description": "Enhanced"}
        branded_data = {"title": "Branded", "description": "Branded"}
        mock_filter.return_value = sample_product_data
        mock_enhance_vlm.return_value = enhanced_data
        mock_apply_branding.return_value = branded_data
        mock_repair.return_value = branded_data

        brand_instructions = "Use playful tone"
        result = _call_nemotron_enhance(sample_vlm_response, sample_product_data, "en-US", brand_instructions)

        # All steps should run
        mock_filter.assert_called_once()
        mock_enhance_vlm.assert_called_once()
        mock_apply_branding.assert_called_once_with(enhanced_data, brand_instructions, "en-US")
        mock_repair.assert_called_once_with(sample_vlm_response, sample_product_data, branded_data, "en-US")
        assert result == branded_data


class TestRunVLMAnalysis:
    """Tests for run_vlm_analysis orchestration function."""
    
    @patch('backend.vlm._call_nemotron_enhance')
    @patch('backend.vlm._call_vlm')
    def test_run_vlm_analysis_generation_mode(self, mock_call_vlm, mock_enhance, sample_image_bytes, sample_vlm_response):
        """Test VLM analysis in generation mode (no product_data)."""
        mock_call_vlm.return_value = sample_vlm_response
        
        enhanced_response = sample_vlm_response.copy()
        enhanced_response["title"] = "Enhanced Title"
        mock_enhance.return_value = enhanced_response
        
        result = run_vlm_analysis(sample_image_bytes, "image/png", "en-US", None, None)
        
        # Should call VLM and enhance
        mock_call_vlm.assert_called_once()
        mock_enhance.assert_called_once()
        
        # Should NOT have enhanced_product in result
        assert "enhanced_product" not in result
        assert result["title"] == "Enhanced Title"
    
    @patch('backend.vlm._call_nemotron_enhance')
    @patch('backend.vlm._call_vlm')
    def test_run_vlm_analysis_augmentation_mode(self, mock_call_vlm, mock_enhance, sample_image_bytes, sample_vlm_response, sample_product_data):
        """Test VLM analysis in augmentation mode (with product_data)."""
        mock_call_vlm.return_value = sample_vlm_response
        
        enhanced_response = {
            "title": "Enhanced Title",
            "description": "Enhanced Description",
            "price": 15.99,
            "categories": ["bags"],
            "tags": ["test"],
            "colors": ["black"],
            "sku": "BAG-001"
        }
        mock_enhance.return_value = enhanced_response
        
        result = run_vlm_analysis(sample_image_bytes, "image/png", "en-US", sample_product_data, None)
        
        # Should have enhanced_product in result
        assert "enhanced_product" in result
        assert isinstance(result["enhanced_product"], dict)
        assert result["enhanced_product"]["price"] == 15.99
        assert result["enhanced_product"]["sku"] == "BAG-001"
    
    @patch('backend.vlm._call_nemotron_enhance')
    @patch('backend.vlm._call_vlm')
    def test_run_vlm_analysis_with_brand_instructions(self, mock_call_vlm, mock_enhance, sample_image_bytes, sample_vlm_response):
        """Test VLM analysis with brand instructions."""
        mock_call_vlm.return_value = sample_vlm_response
        mock_enhance.return_value = sample_vlm_response
        
        brand_instructions = "Use premium luxury tone"
        result = run_vlm_analysis(sample_image_bytes, "image/png", "en-US", None, brand_instructions)
        
        # Should pass brand_instructions to enhance
        mock_enhance.assert_called_once()
        # Check positional args or kwargs
        call_args = mock_enhance.call_args
        # brand_instructions is the 4th argument (index 3) in _call_nemotron_enhance
        assert call_args[0][3] == brand_instructions or call_args.kwargs.get('brand_instructions') == brand_instructions
    
    def test_run_vlm_analysis_validates_image_bytes(self):
        """Test that function validates required parameters."""
        with pytest.raises(ValueError) as exc_info:
            run_vlm_analysis(None, "image/png", "en-US", None, None)
        
        assert "image_bytes is required" in str(exc_info.value)
    
    def test_run_vlm_analysis_validates_content_type(self, sample_image_bytes):
        """Test that function validates content type."""
        with pytest.raises(ValueError) as exc_info:
            run_vlm_analysis(sample_image_bytes, "text/plain", "en-US", None, None)
        
        assert "content_type must be an image" in str(exc_info.value)

    @patch('backend.vlm._call_nemotron_enhance')
    @patch('backend.vlm._call_vlm')
    def test_run_vlm_analysis_returns_enriched_fields_without_policy_evaluation(
        self,
        mock_call_vlm,
        mock_enhance,
        sample_image_bytes,
        sample_vlm_response,
    ):
        """Test VLM analysis returns enriched fields and leaves policy checks to the API layer."""
        mock_call_vlm.return_value = sample_vlm_response
        mock_enhance.return_value = sample_vlm_response

        result = run_vlm_analysis(
            sample_image_bytes,
            "image/png",
            "en-US",
            None,
            None,
        )

        assert result["title"] == sample_vlm_response["title"]
        assert "policy_decision" not in result


class TestSplitVLMFlow:
    @patch('backend.vlm._call_vlm')
    def test_extract_vlm_observation_returns_raw_vlm_output(self, mock_call_vlm, sample_image_bytes, sample_vlm_response):
        mock_call_vlm.return_value = sample_vlm_response

        result = extract_vlm_observation(sample_image_bytes, "image/png", "en-US")

        assert result == sample_vlm_response
        mock_call_vlm.assert_called_once_with(sample_image_bytes, "image/png", "en-US")

    @patch('backend.vlm._call_nemotron_enhance')
    def test_build_enriched_vlm_result_uses_existing_vlm_observation(self, mock_enhance, sample_vlm_response):
        enhanced_response = sample_vlm_response.copy()
        enhanced_response["title"] = "Enhanced Title"
        mock_enhance.return_value = enhanced_response

        result = build_enriched_vlm_result(sample_vlm_response, "en-US", None, None)

        assert result["title"] == "Enhanced Title"
        assert "enhanced_product" not in result

    @patch('backend.vlm._call_nemotron_enhance')
    def test_build_enriched_vlm_result_normalizes_categories_and_colors(self, mock_enhance, sample_vlm_response, sample_product_data):
        enhanced_response = sample_vlm_response.copy()
        enhanced_response["categories"] = ["accessories", "bags", "unknown", "uncategorized"]
        enhanced_response["colors"] = ["acero inoxidable", "black leather", "grey fabric", "gold-tone"]
        mock_enhance.return_value = enhanced_response

        result = build_enriched_vlm_result(sample_vlm_response, "en-US", sample_product_data, None)

        assert result["categories"] == ["bags"]
        assert result["colors"] == ["black", "gray", "gold"]
        assert result["enhanced_product"]["categories"] == ["bags"]
        assert result["enhanced_product"]["colors"] == ["black", "gray", "gold"]
