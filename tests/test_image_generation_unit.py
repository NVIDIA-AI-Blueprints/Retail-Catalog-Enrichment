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
Unit tests for image generation module with mocked external APIs.

Tests image variation generation pipeline with mocked OpenAI and HTTPX calls.
"""
import json
import pytest
import base64
from unittest.mock import Mock, patch, AsyncMock
from backend.image import (
    _call_planner_llm,
    _call_flux_edit,
    persist_generated_image,
    generate_image_variation
)


class TestCallPlannerLLM:
    """Tests for _call_planner_llm function."""
    
    @patch('backend.image.OpenAI')
    @patch('backend.image.get_config')
    def test_planner_success_with_valid_json(self, mock_get_config, mock_openai_class, sample_flux_plan, mock_env_vars):
        """Test successful planner call with valid JSON plan."""
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
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(sample_flux_plan)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Call function
        result = _call_planner_llm("Test Product", "Test description", ["accessories"], "en-US")
        
        # Assertions
        assert isinstance(result, dict)
        assert "preserve_subject" in result
        assert "background_style" in result
        assert "camera_angle" in result
        assert "lighting" in result
        assert "cfg_scale" in result
        assert "steps" in result
    
    @patch('backend.image.OpenAI')
    @patch('backend.image.get_config')
    def test_planner_extracts_json_from_markdown(self, mock_get_config, mock_openai_class, sample_flux_plan, mock_env_vars):
        """Test JSON extraction from markdown code blocks."""
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
        
        # Wrap JSON in markdown
        markdown_response = f"```json\n{json.dumps(sample_flux_plan)}\n```"
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = markdown_response
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Call function
        result = _call_planner_llm("Test", "Test", ["test"], "en-US")
        
        # Should extract JSON successfully
        assert isinstance(result, dict)
        assert result["preserve_subject"] == sample_flux_plan["preserve_subject"]
    
    @patch('backend.image.OpenAI')
    @patch('backend.image.get_config')
    def test_planner_fallback_for_invalid_json(self, mock_get_config, mock_openai_class, mock_env_vars):
        """Test fallback plan generation for invalid JSON response."""
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
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = "This is not valid JSON"
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Call function
        result = _call_planner_llm("Test Product", "Test", ["test"], "en-US")
        
        # Should return fallback plan
        assert isinstance(result, dict)
        assert "preserve_subject" in result
        assert "background_style" in result
        assert "cfg_scale" in result
        assert isinstance(result["cfg_scale"], (int, float))
    
    @patch('backend.image.OpenAI')
    @patch('backend.image.get_config')
    def test_planner_with_different_locales(self, mock_get_config, mock_openai_class, sample_flux_plan, mock_env_vars):
        """Test planner with different locale contexts."""
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
        
        french_plan = sample_flux_plan.copy()
        french_plan["background_style"] = "marble bistro table at a Parisian café"
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(french_plan)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Call with French locale
        result = _call_planner_llm("Sac à main", "Description", ["accessories"], "fr-FR")
        
        # Plan should still be in English (FLUX requirement)
        assert isinstance(result, dict)
        assert "Parisian" in result["background_style"]


class TestCallFluxEdit:
    """Tests for _call_flux_edit async function."""
    
    @pytest.mark.asyncio
    @patch('backend.image.httpx.AsyncClient')
    @patch('backend.image.get_config')
    async def test_flux_edit_success(self, mock_get_config, mock_async_client_class, sample_image_bytes, mock_env_vars):
        """Test successful FLUX edit call."""
        # Mock config
        mock_config = Mock()
        mock_config.get_flux_config.return_value = {
            'url': 'http://test-flux:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock httpx AsyncClient
        mock_response = Mock()
        mock_response.json.return_value = {
            "image": "base64encodedimagedata"
        }
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Call function
        result = await _call_flux_edit(sample_image_bytes, "image/png", "test prompt", 30, 3.5, 42)
        
        # Assertions
        assert isinstance(result, dict)
        mock_client_instance.post.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('backend.image.httpx.AsyncClient')
    @patch('backend.image.get_config')
    async def test_flux_edit_with_various_response_formats(self, mock_get_config, mock_async_client_class, sample_image_bytes, mock_env_vars):
        """Test FLUX edit handles various response formats."""
        # Mock config
        mock_config = Mock()
        mock_config.get_flux_config.return_value = {
            'url': 'http://test-flux:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Test with 'output' key
        mock_response = Mock()
        mock_response.json.return_value = {
            "output": "base64imagedata"
        }
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        result = await _call_flux_edit(sample_image_bytes, "image/png", "prompt", 30, 3.5)
        
        assert isinstance(result, dict)
        assert "output" in result


class TestPersistGeneratedImage:
    """Tests for persist_generated_image function."""
    
    def test_persist_image_without_enhanced_product(self, tmp_path, sample_base64_image, monkeypatch):
        """Test persisting image without enhanced product data."""
        # Set OUTPUT_DIR to temp directory
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
        
        result = persist_generated_image(
            image_b64=sample_base64_image,
            title="Test Product",
            description="Test description",
            categories=["accessories"],
            tags=["test", "tag"],
            colors=["black"],
            locale="en-US",
            content_type="image/png",
            enhanced_product=None
        )
        
        # Assertions
        assert "artifact_id" in result
        assert "image_path" in result
        assert "metadata_path" in result
        
        # Check files exist
        from pathlib import Path
        assert Path(result["image_path"]).exists()
        assert Path(result["metadata_path"]).exists()
        
        # Check metadata content
        with open(result["metadata_path"], 'r') as f:
            metadata = json.load(f)
        
        assert metadata["title"] == "Test Product"
        assert metadata["locale"] == "en-US"
        assert "created_at" in metadata
    
    def test_persist_image_with_enhanced_product(self, tmp_path, sample_base64_image, sample_enhanced_product, monkeypatch):
        """Test persisting image with enhanced product data."""
        # Set OUTPUT_DIR to temp directory
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
        
        result = persist_generated_image(
            image_b64=sample_base64_image,
            title="Test Product",
            description="Test description",
            categories=["accessories"],
            tags=["test"],
            colors=["black"],
            locale="en-US",
            content_type="image/png",
            enhanced_product=sample_enhanced_product
        )
        
        # Check metadata includes enhanced product
        with open(result["metadata_path"], 'r') as f:
            metadata = json.load(f)
        
        assert metadata["price"] == sample_enhanced_product["price"]
        assert metadata["sku"] == sample_enhanced_product["sku"]
        assert "id" in metadata
        assert "locale" in metadata
    
    def test_persist_image_generates_unique_ids(self, tmp_path, sample_base64_image, monkeypatch):
        """Test that multiple persists generate unique artifact IDs."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
        
        result1 = persist_generated_image(
            image_b64=sample_base64_image,
            title="Test 1",
            description="Test",
            categories=["test"],
            tags=[],
            colors=[],
            locale="en-US",
            content_type="image/png"
        )
        
        result2 = persist_generated_image(
            image_b64=sample_base64_image,
            title="Test 2",
            description="Test",
            categories=["test"],
            tags=[],
            colors=[],
            locale="en-US",
            content_type="image/png"
        )
        
        # IDs should be different
        assert result1["artifact_id"] != result2["artifact_id"]


class TestGenerateImageVariation:
    """Tests for generate_image_variation orchestration function."""
    
    @pytest.mark.asyncio
    @patch('backend.image.evaluate_image_quality')
    @patch('backend.image.persist_generated_image')
    @patch('backend.image._call_flux_edit')
    @patch('backend.image._call_planner_llm')
    async def test_generate_variation_complete_pipeline(self, mock_planner, mock_flux, mock_persist, mock_reflection, sample_image_bytes, sample_flux_plan):
        """Test complete image generation pipeline with reflection."""
        # Mock planner
        mock_planner.return_value = sample_flux_plan
        
        # Mock FLUX
        mock_flux.return_value = {
            "image": "generatedbase64image"
        }
        
        # Mock reflection
        mock_reflection.return_value = {
            "score": 85.5,
            "issues": ["Minor background blur"]
        }
        
        # Mock persist
        mock_persist.return_value = {
            "artifact_id": "test123",
            "image_path": "/tmp/test.png",
            "metadata_path": "/tmp/test.json"
        }
        
        # Call function
        result = await generate_image_variation(
            image_bytes=sample_image_bytes,
            content_type="image/png",
            title="Test Product",
            description="Test description",
            categories=["accessories"],
            tags=["test"],
            colors=["black"],
            locale="en-US"
        )
        
        # Assertions
        assert "generated_image_b64" in result
        assert "artifact_id" in result
        assert "image_path" in result
        assert "metadata_path" in result
        assert "variation_plan" in result
        assert "quality_score" in result
        assert "quality_issues" in result
        
        # Verify new reflection fields
        assert result["quality_score"] == 85.5
        assert isinstance(result["quality_issues"], list)
        assert len(result["quality_issues"]) == 1
        
        # Verify pipeline calls (now includes reflection)
        mock_planner.assert_called_once()
        mock_flux.assert_called_once()
        mock_reflection.assert_called_once()
        mock_persist.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('backend.image.evaluate_image_quality')
    @patch('backend.image.persist_generated_image')
    @patch('backend.image._call_flux_edit')
    @patch('backend.image._call_planner_llm')
    async def test_generate_variation_with_enhanced_product(self, mock_planner, mock_flux, mock_persist, mock_reflection, sample_image_bytes, sample_flux_plan, sample_enhanced_product):
        """Test image generation with enhanced product data."""
        # Mock planner
        mock_planner.return_value = sample_flux_plan
        
        # Mock FLUX
        mock_flux.return_value = {
            "image": "generatedbase64image"
        }
        
        # Mock reflection
        mock_reflection.return_value = {
            "score": 92.0,
            "issues": []
        }
        
        # Mock persist
        mock_persist.return_value = {
            "artifact_id": "test456",
            "image_path": "/tmp/test2.png",
            "metadata_path": "/tmp/test2.json"
        }
        
        # Call function with enhanced_product
        result = await generate_image_variation(
            image_bytes=sample_image_bytes,
            content_type="image/png",
            title="Test Product",
            description="Test description",
            categories=["accessories"],
            tags=["test"],
            colors=["black"],
            locale="en-US",
            enhanced_product=sample_enhanced_product
        )
        
        # Verify enhanced_product was passed to persist
        persist_call_kwargs = mock_persist.call_args[1]
        assert persist_call_kwargs["enhanced_product"] == sample_enhanced_product
        
        # Verify reflection ran
        assert result["quality_score"] == 92.0
        assert result["quality_issues"] == []
    
    @pytest.mark.asyncio
    @patch('backend.image._call_flux_edit')
    @patch('backend.image._call_planner_llm')
    async def test_generate_variation_flux_returns_no_image(self, mock_planner, mock_flux, sample_image_bytes, sample_flux_plan):
        """Test error handling when FLUX returns no image."""
        # Mock planner
        mock_planner.return_value = sample_flux_plan
        
        # Mock FLUX with no image
        mock_flux.return_value = {
            "status": "success",
            # No image field
        }
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await generate_image_variation(
                image_bytes=sample_image_bytes,
                content_type="image/png",
                title="Test",
                description="Test",
                categories=["test"],
                tags=[],
                colors=[],
                locale="en-US"
            )
        
        assert "did not include an image" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('backend.image.evaluate_image_quality')
    @patch('backend.image.persist_generated_image')
    @patch('backend.image._call_flux_edit')
    @patch('backend.image._call_planner_llm')
    async def test_generate_variation_with_different_locales(self, mock_planner, mock_flux, mock_persist, mock_reflection, sample_image_bytes, sample_base64_image, sample_flux_plan):
        """Test image generation with different locales."""
        # Mock planner
        mock_planner.return_value = sample_flux_plan
        
        # Mock FLUX with valid base64
        mock_flux.return_value = {"image": sample_base64_image}
        
        # Mock reflection
        mock_reflection.return_value = {
            "score": 88.0,
            "issues": []
        }
        
        # Mock persist
        mock_persist.return_value = {
            "artifact_id": "test789",
            "image_path": "/tmp/test3.png",
            "metadata_path": "/tmp/test3.json"
        }
        
        # Test with Spanish locale
        result = await generate_image_variation(
            image_bytes=sample_image_bytes,
            content_type="image/png",
            title="Producto de Prueba",
            description="Descripción de prueba",
            categories=["accessories"],
            tags=["prueba"],
            colors=["negro"],
            locale="es-ES"
        )
        
        # Verify planner was called with Spanish locale
        planner_call_args = mock_planner.call_args[0]
        planner_call_kwargs = mock_planner.call_args[1] if len(mock_planner.call_args) > 1 else {}
        # Locale should be passed to planner
        assert planner_call_kwargs.get("locale") == "es-ES" or planner_call_args[-1] == "es-ES"
        
        # Verify reflection fields in result
        assert "quality_score" in result
        assert "quality_issues" in result
    
    @pytest.mark.asyncio
    @patch('backend.image.evaluate_image_quality')
    @patch('backend.image.persist_generated_image')
    @patch('backend.image._call_flux_edit')
    @patch('backend.image._call_planner_llm')
    async def test_generate_variation_handles_reflection_failure(self, mock_planner, mock_flux, mock_persist, mock_reflection, sample_image_bytes, sample_base64_image, sample_flux_plan):
        """Test image generation handles reflection failure gracefully."""
        # Mock planner
        mock_planner.return_value = sample_flux_plan
        
        # Mock FLUX with valid base64
        mock_flux.return_value = {"image": sample_base64_image}
        
        # Mock reflection to return None (failure)
        mock_reflection.return_value = None
        
        # Mock persist
        mock_persist.return_value = {
            "artifact_id": "test999",
            "image_path": "/tmp/test4.png",
            "metadata_path": "/tmp/test4.json"
        }
        
        # Call function
        result = await generate_image_variation(
            image_bytes=sample_image_bytes,
            content_type="image/png",
            title="Test Product",
            description="Test description",
            categories=["accessories"],
            tags=["test"],
            colors=["black"],
            locale="en-US"
        )
        
        # Should still complete successfully
        assert "generated_image_b64" in result
        assert "artifact_id" in result
        
        # Reflection fields should be None and empty list
        assert result["quality_score"] is None
        assert result["quality_issues"] == []
        
        # Pipeline should have completed despite reflection failure
        mock_persist.assert_called_once()

