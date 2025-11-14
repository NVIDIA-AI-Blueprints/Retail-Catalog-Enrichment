"""
Pytest configuration and shared fixtures for unit tests.

This module provides reusable fixtures and mocks for testing
the catalog enrichment backend without external dependencies.
"""
import os
import json
import base64
import pytest
from io import BytesIO
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create a minimal valid PNG image (1x1 pixel)."""
    img = Image.new('RGB', (1, 1), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Create a minimal valid JPEG image (1x1 pixel)."""
    img = Image.new('RGB', (1, 1), color='blue')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def sample_base64_image() -> str:
    """Return a base64-encoded sample image."""
    img = Image.new('RGB', (1, 1), color='green')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("NVIDIA_API_BASE_URL", "https://test.nvidia.com/v1")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/test_outputs")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def sample_vlm_response() -> Dict[str, Any]:
    """Sample VLM analysis response."""
    return {
        "title": "Elegant Black Handbag with Gold Accents",
        "description": "A sophisticated leather handbag featuring gold-tone hardware and a structured design perfect for evening occasions.",
        "categories": ["accessories", "bags"],
        "tags": ["black leather", "gold hardware", "evening bag", "structured", "luxury", "handbag", "elegant", "formal", "premium", "classic"],
        "colors": ["black", "gold"]
    }


@pytest.fixture
def sample_product_data() -> Dict[str, Any]:
    """Sample existing product data for augmentation."""
    return {
        "title": "Black Purse",
        "description": "Nice bag",
        "price": 15.99,
        "categories": ["accessories"],
        "tags": ["bag", "purse"],
        "sku": "BAG-001"
    }


@pytest.fixture
def sample_enhanced_product() -> Dict[str, Any]:
    """Sample enhanced product data."""
    return {
        "title": "Elegant Black Evening Handbag with Gold Hardware",
        "description": "This exquisite leather handbag combines timeless elegance with modern sophistication. Features premium black leather construction with lustrous gold-tone hardware.",
        "price": 15.99,
        "categories": ["accessories", "bags"],
        "tags": ["black leather", "gold hardware", "evening bag", "luxury", "elegant"],
        "colors": ["black", "gold"],
        "sku": "BAG-001"
    }


@pytest.fixture
def sample_flux_plan() -> Dict[str, Any]:
    """Sample FLUX variation plan."""
    return {
        "preserve_subject": "elegant black handbag with gold hardware",
        "background_style": "marble bistro table at a Parisian café with Eiffel Tower softly blurred in background",
        "camera_angle": "overhead",
        "lighting": "natural window light",
        "color_palette": "warm neutrals with gold accents",
        "negatives": ["do not alter the subject", "no text, no logos, no duplicates"],
        "cfg_scale": 3.5,
        "steps": 30,
        "variants": 1
    }


@pytest.fixture
def mock_openai_completion():
    """Create a mock OpenAI completion response with streaming."""
    def create_mock_completion(content: str):
        """Factory to create mock completion with specific content."""
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = content
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        return [mock_chunk]
    
    return create_mock_completion


@pytest.fixture
def mock_openai_client(mock_openai_completion):
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_chat = Mock()
    mock_completions = Mock()
    
    # Set up the chained mock structure
    mock_client.chat = mock_chat
    mock_chat.completions = mock_completions
    
    # Default successful response
    default_response = {
        "title": "Test Product",
        "description": "Test description",
        "categories": ["accessories"],
        "tags": ["test"],
        "colors": ["black"]
    }
    
    mock_completions.create.return_value = mock_openai_completion(json.dumps(default_response))
    
    return mock_client


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"image": "base64encodedimage"}
    mock_response.content = b"fake_glb_data"
    mock_response.raise_for_status = Mock()
    return mock_response


@pytest.fixture
def mock_httpx_client(mock_httpx_response):
    """Create a mock httpx AsyncClient."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_httpx_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


@pytest.fixture
def valid_locales() -> set:
    """Return set of valid locales."""
    return {
        "en-US", "en-GB", "en-AU", "en-CA",
        "es-ES", "es-MX", "es-AR", "es-CO",
        "fr-FR", "fr-CA"
    }


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary."""
    return {
        "vlm": {
            "url": "http://test-vlm:8000/v1",
            "model": "test-vlm-model"
        },
        "llm": {
            "url": "http://test-llm:8000/v1",
            "model": "test-llm-model"
        },
        "flux": {
            "url": "http://test-flux:8000/v1/infer"
        },
        "trellis": {
            "url": "http://test-trellis:8000/v1/infer"
        }
    }


@pytest.fixture
def invalid_config_dict() -> Dict[str, Any]:
    """Invalid configuration dictionary (missing required fields)."""
    return {
        "vlm": {
            "url": "http://test-vlm:8000/v1"
            # Missing 'model' field
        },
        "llm": {
            "model": "test-llm-model"
            # Missing 'url' field
        }
    }

