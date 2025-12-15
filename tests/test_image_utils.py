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
Unit tests for pure functions in image.py module.

Tests image generation utility functions without external dependencies.
"""
import pytest
from backend.image import _render_flux_prompt, _extract_base64_image_from_flux_response


class TestRenderFluxPrompt:
    """Tests for _render_flux_prompt function."""
    
    def test_render_complete_plan(self, sample_flux_plan):
        """Test rendering a complete FLUX plan."""
        prompt = _render_flux_prompt(sample_flux_plan, locale="en-US")
        
        assert "elegant black handbag with gold hardware" in prompt.lower()
        assert "marble bistro table" in prompt.lower()
        assert "overhead" in prompt.lower()
        assert "natural window light" in prompt.lower()
        assert "do not alter the subject" in prompt.lower()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_render_minimal_plan(self):
        """Test rendering with minimal plan data (missing optional fields)."""
        minimal_plan = {
            "preserve_subject": "test product",
            "background_style": "neutral background"
        }
        
        prompt = _render_flux_prompt(minimal_plan)
        
        assert "test product" in prompt.lower()
        assert "neutral background" in prompt.lower()
        assert "hyperrealistic" in prompt.lower()
        assert isinstance(prompt, str)
    
    def test_render_with_negatives_list(self):
        """Test rendering with negatives as a list."""
        plan = {
            "preserve_subject": "product",
            "background_style": "studio",
            "negatives": ["no text", "no logos", "no duplicates"]
        }
        
        prompt = _render_flux_prompt(plan)
        
        assert "avoid:" in prompt.lower()
        assert "no text" in prompt.lower()
        assert "no logos" in prompt.lower()
    
    def test_render_with_negatives_string(self):
        """Test rendering with negatives as a string."""
        plan = {
            "preserve_subject": "product",
            "background_style": "studio",
            "negatives": "no text; no logos"
        }
        
        prompt = _render_flux_prompt(plan)
        
        assert "avoid:" in prompt.lower()
        assert "no text" in prompt.lower()
    
    def test_render_empty_plan(self):
        """Test rendering with empty plan dict."""
        prompt = _render_flux_prompt({})
        
        # Should use defaults
        assert "the product" in prompt.lower()
        assert "hyperrealistic" in prompt.lower()
        assert isinstance(prompt, str)
    
    def test_render_different_locales(self):
        """Test that prompt is in English regardless of locale."""
        plan = {
            "preserve_subject": "handbag",
            "background_style": "Parisian café"
        }
        
        prompt_us = _render_flux_prompt(plan, locale="en-US")
        prompt_es = _render_flux_prompt(plan, locale="es-ES")
        prompt_fr = _render_flux_prompt(plan, locale="fr-FR")
        
        # All prompts should be in English (FLUX requirement)
        assert "handbag" in prompt_us.lower()
        assert "handbag" in prompt_es.lower()
        assert "handbag" in prompt_fr.lower()


class TestExtractBase64ImageFromFluxResponse:
    """Tests for _extract_base64_image_from_flux_response function."""
    
    def test_extract_from_image_key(self):
        """Test extraction from 'image' key."""
        response = {"image": "base64imagedata123"}
        result = _extract_base64_image_from_flux_response(response)
        assert result == "base64imagedata123"
    
    def test_extract_from_output_key(self):
        """Test extraction from 'output' key."""
        response = {"output": "base64imagedata456"}
        result = _extract_base64_image_from_flux_response(response)
        assert result == "base64imagedata456"
    
    def test_extract_from_data_key(self):
        """Test extraction from 'data' key."""
        response = {"data": "base64imagedata789"}
        result = _extract_base64_image_from_flux_response(response)
        assert result == "base64imagedata789"
    
    def test_extract_from_images_array_string(self):
        """Test extraction from 'images' array with string."""
        response = {"images": ["base64imageXYZ"]}
        result = _extract_base64_image_from_flux_response(response)
        assert result == "base64imageXYZ"
    
    def test_extract_from_images_array_dict(self):
        """Test extraction from 'images' array with dict containing 'b64'."""
        response = {"images": [{"b64": "base64imageABC"}]}
        result = _extract_base64_image_from_flux_response(response)
        assert result == "base64imageABC"
    
    def test_extract_from_images_array_dict_base64_key(self):
        """Test extraction from 'images' array with dict containing 'base64'."""
        response = {"images": [{"base64": "base64imageDEF"}]}
        result = _extract_base64_image_from_flux_response(response)
        assert result == "base64imageDEF"
    
    def test_extract_from_images_array_dict_image_key(self):
        """Test extraction from 'images' array with dict containing 'image'."""
        response = {"images": [{"image": "base64imageGHI"}]}
        result = _extract_base64_image_from_flux_response(response)
        assert result == "base64imageGHI"
    
    def test_extract_from_artifacts_array(self):
        """Test extraction from 'artifacts' array."""
        response = {"artifacts": [{"base64": "base64imageJKL"}]}
        result = _extract_base64_image_from_flux_response(response)
        assert result == "base64imageJKL"
    
    def test_extract_returns_none_for_empty_response(self):
        """Test that None is returned for empty response."""
        response = {}
        result = _extract_base64_image_from_flux_response(response)
        assert result is None
    
    def test_extract_returns_none_for_empty_string(self):
        """Test that None is returned for empty string values."""
        response = {"image": ""}
        result = _extract_base64_image_from_flux_response(response)
        assert result is None
    
    def test_extract_returns_none_for_empty_array(self):
        """Test that None is returned for empty arrays."""
        response = {"images": []}
        result = _extract_base64_image_from_flux_response(response)
        assert result is None
    
    def test_extract_returns_none_for_none_values(self):
        """Test that None is returned for None values."""
        response = {"image": None}
        result = _extract_base64_image_from_flux_response(response)
        assert result is None
    
    def test_extract_priority_order(self):
        """Test that extraction follows priority order (image > output > data)."""
        response = {
            "image": "priority1",
            "output": "priority2",
            "data": "priority3"
        }
        result = _extract_base64_image_from_flux_response(response)
        assert result == "priority1"
    
    def test_extract_complex_response(self):
        """Test extraction from complex realistic response."""
        response = {
            "status": "success",
            "metadata": {"model": "flux"},
            "images": [
                {"base64": "actualImageData", "format": "png"}
            ]
        }
        result = _extract_base64_image_from_flux_response(response)
        assert result == "actualImageData"

