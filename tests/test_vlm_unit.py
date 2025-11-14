"""
Unit tests for VLM module with mocked OpenAI API calls.

Tests VLM analysis, enhancement, and branding functions without external dependencies.
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.vlm import (
    _call_vlm,
    _call_nemotron_enhance_vlm,
    _call_nemotron_apply_branding,
    _call_nemotron_enhance,
    run_vlm_analysis
)


class TestCallVLM:
    """Tests for _call_vlm function with mocked OpenAI client."""
    
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_call_vlm_success_with_valid_json(self, mock_get_config, mock_openai_class, sample_image_bytes, sample_vlm_response, mock_env_vars):
        """Test successful VLM call with valid JSON response."""
        # Mock config
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock streaming response
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(sample_vlm_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Call function
        result = _call_vlm(sample_image_bytes, "image/png")
        
        # Assertions
        assert isinstance(result, dict)
        assert result["title"] == sample_vlm_response["title"]
        assert result["description"] == sample_vlm_response["description"]
        assert result["categories"] == sample_vlm_response["categories"]
        assert "tags" in result
        assert "colors" in result
    
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_call_vlm_with_invalid_json_fallback(self, mock_get_config, mock_openai_class, sample_image_bytes, mock_env_vars):
        """Test VLM call with non-JSON response uses fallback."""
        # Mock config
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client with non-JSON response
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
        result = _call_vlm(sample_image_bytes, "image/png")
        
        # Should return fallback structure
        assert isinstance(result, dict)
        assert result["title"] == ""
        assert result["description"] == "This is not valid JSON"
        assert result["categories"] == ["uncategorized"]
        assert result["tags"] == []
        assert result["colors"] == []
    
    @patch('backend.vlm.OpenAI')
    @patch('backend.vlm.get_config')
    def test_call_vlm_with_different_image_types(self, mock_get_config, mock_openai_class, sample_jpeg_bytes, sample_vlm_response, mock_env_vars):
        """Test VLM call with different image content types."""
        # Mock config
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = json.dumps(sample_vlm_response)
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        # Test with JPEG
        result = _call_vlm(sample_jpeg_bytes, "image/jpeg")
        assert isinstance(result, dict)


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
            "categories": ["accessories"],
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
            "categories": ["accessories", "bags"],
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
            "categories": ["accessories"],
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
            "categories": ["accessories"],
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


class TestCallNemotronEnhance:
    """Tests for _call_nemotron_enhance orchestration function."""
    
    @patch('backend.vlm._call_nemotron_apply_branding')
    @patch('backend.vlm._call_nemotron_enhance_vlm')
    def test_enhance_without_brand_instructions(self, mock_enhance_vlm, mock_apply_branding, sample_vlm_response):
        """Test enhancement pipeline without brand instructions (Step 2 skipped)."""
        enhanced_data = {"title": "Enhanced", "description": "Enhanced"}
        mock_enhance_vlm.return_value = enhanced_data
        
        result = _call_nemotron_enhance(sample_vlm_response, None, "en-US", None)
        
        # Step 1 should be called
        mock_enhance_vlm.assert_called_once()
        # Step 2 should NOT be called
        mock_apply_branding.assert_not_called()
        assert result == enhanced_data
    
    @patch('backend.vlm._call_nemotron_apply_branding')
    @patch('backend.vlm._call_nemotron_enhance_vlm')
    def test_enhance_with_brand_instructions(self, mock_enhance_vlm, mock_apply_branding, sample_vlm_response):
        """Test enhancement pipeline with brand instructions (both steps)."""
        enhanced_data = {"title": "Enhanced", "description": "Enhanced"}
        branded_data = {"title": "Branded", "description": "Branded"}
        
        mock_enhance_vlm.return_value = enhanced_data
        mock_apply_branding.return_value = branded_data
        
        brand_instructions = "Use playful tone"
        result = _call_nemotron_enhance(sample_vlm_response, None, "en-US", brand_instructions)
        
        # Both steps should be called
        mock_enhance_vlm.assert_called_once()
        mock_apply_branding.assert_called_once_with(enhanced_data, brand_instructions, "en-US")
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
            "categories": ["accessories"],
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

