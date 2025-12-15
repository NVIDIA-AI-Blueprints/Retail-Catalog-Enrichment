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
Unit tests for TRELLIS 3D generation module with mocked httpx.

Tests 3D asset generation without external dependencies.
"""
import json
import base64
import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx
from backend.trellis import generate_3d_asset


class TestGenerate3DAsset:
    """Tests for generate_3d_asset function."""
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_success_binary_response(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test successful 3D generation with binary GLB response."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock httpx AsyncClient with binary response
        fake_glb_data = b"FAKE_GLB_BINARY_DATA"
        mock_response = Mock()
        mock_response.content = fake_glb_data
        mock_response.json.side_effect = json.JSONDecodeError("test", "test", 0)  # Not JSON
        mock_response.raise_for_status = Mock()
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Call function
        result = await generate_3d_asset(
            image_bytes=sample_image_bytes,
            content_type="image/png",
            slat_cfg_scale=5.0,
            ss_cfg_scale=10.0,
            slat_sampling_steps=50,
            ss_sampling_steps=50,
            seed=42
        )
        
        # Assertions
        assert isinstance(result, dict)
        assert "glb_data" in result
        assert "artifact_id" in result
        assert "metadata" in result
        assert result["glb_data"] == fake_glb_data
        assert result["metadata"]["seed"] == 42
        assert result["metadata"]["size_bytes"] == len(fake_glb_data)
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_success_json_response_with_artifacts(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test successful 3D generation with JSON response containing artifacts."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock httpx AsyncClient with JSON response
        fake_glb_data = b"FAKE_GLB_DATA_FROM_JSON"
        glb_b64 = base64.b64encode(fake_glb_data).decode("ascii")
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "custom_artifact_id_123",
            "artifacts": [
                {"base64": glb_b64, "format": "glb"}
            ]
        }
        mock_response.raise_for_status = Mock()
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Call function
        result = await generate_3d_asset(
            image_bytes=sample_image_bytes,
            content_type="image/png",
            seed=100
        )
        
        # Assertions
        assert result["glb_data"] == fake_glb_data
        assert result["artifact_id"] == "custom_artifact_id_123"
        assert result["metadata"]["size_bytes"] == len(fake_glb_data)
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_with_custom_parameters(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test 3D generation with custom parameters."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock response
        fake_glb_data = b"TEST_GLB"
        mock_response = Mock()
        mock_response.content = fake_glb_data
        mock_response.json.side_effect = json.JSONDecodeError("test", "test", 0)
        mock_response.raise_for_status = Mock()
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Call with custom parameters
        result = await generate_3d_asset(
            image_bytes=sample_image_bytes,
            content_type="image/jpeg",
            slat_cfg_scale=7.5,
            ss_cfg_scale=12.0,
            slat_sampling_steps=60,
            ss_sampling_steps=60,
            seed=999
        )
        
        # Verify metadata includes custom parameters
        assert result["metadata"]["slat_cfg_scale"] == 7.5
        assert result["metadata"]["ss_cfg_scale"] == 12.0
        assert result["metadata"]["slat_sampling_steps"] == 60
        assert result["metadata"]["ss_sampling_steps"] == 60
        assert result["metadata"]["seed"] == 999
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_handles_timeout(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test 3D generation handles timeout errors."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock timeout exception
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=httpx.TimeoutException("Request timeout"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Should raise TimeoutException
        with pytest.raises(httpx.TimeoutException):
            await generate_3d_asset(
                image_bytes=sample_image_bytes,
                content_type="image/png"
            )
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_handles_http_error(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test 3D generation handles HTTP errors."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=Mock(),
            response=mock_response
        )
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Should raise HTTPStatusError
        with pytest.raises(httpx.HTTPStatusError):
            await generate_3d_asset(
                image_bytes=sample_image_bytes,
                content_type="image/png"
            )
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_handles_request_error(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test 3D generation handles request errors."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock request error
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Should raise RequestError
        with pytest.raises(httpx.RequestError):
            await generate_3d_asset(
                image_bytes=sample_image_bytes,
                content_type="image/png"
            )
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_with_different_image_formats(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test 3D generation with different image formats."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock response
        fake_glb_data = b"GLB_DATA"
        mock_response = Mock()
        mock_response.content = fake_glb_data
        mock_response.json.side_effect = json.JSONDecodeError("test", "test", 0)
        mock_response.raise_for_status = Mock()
        
        mock_client_instance = AsyncMock()
        
        # Capture the payload to verify image format
        captured_payload = None
        async def capture_post(*args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get('json', {})
            return mock_response
        
        mock_client_instance.post = capture_post
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Test with JPEG
        await generate_3d_asset(
            image_bytes=sample_image_bytes,
            content_type="image/jpeg"
        )
        
        # Verify image format in payload
        assert captured_payload is not None
        assert "data:image/jpeg;base64," in captured_payload["image"]
        
        # Test with PNG
        await generate_3d_asset(
            image_bytes=sample_image_bytes,
            content_type="image/png"
        )
        
        assert "data:image/png;base64," in captured_payload["image"]
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_default_parameters(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test 3D generation uses correct default parameters."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock response
        fake_glb_data = b"GLB_WITH_DEFAULTS"
        mock_response = Mock()
        mock_response.content = fake_glb_data
        mock_response.json.side_effect = json.JSONDecodeError("test", "test", 0)
        mock_response.raise_for_status = Mock()
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Call with defaults
        result = await generate_3d_asset(
            image_bytes=sample_image_bytes,
            content_type="image/png"
        )
        
        # Verify default values in metadata
        assert result["metadata"]["slat_cfg_scale"] == 5.0
        assert result["metadata"]["ss_cfg_scale"] == 10.0
        assert result["metadata"]["slat_sampling_steps"] == 50
        assert result["metadata"]["ss_sampling_steps"] == 50
        assert result["metadata"]["seed"] == 0
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_fallback_artifact_id(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test 3D generation creates fallback artifact ID when not in response."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock response with no ID
        fake_glb_data = b"GLB_NO_ID"
        mock_response = Mock()
        mock_response.content = fake_glb_data
        mock_response.json.side_effect = json.JSONDecodeError("test", "test", 0)
        mock_response.raise_for_status = Mock()
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Call function
        result = await generate_3d_asset(
            image_bytes=sample_image_bytes,
            content_type="image/png",
            seed=777
        )
        
        # Should create fallback ID using seed
        assert result["artifact_id"] == "trellis_777"
    
    @pytest.mark.asyncio
    @patch('backend.trellis.httpx.AsyncClient')
    @patch('backend.trellis.get_config')
    async def test_generate_3d_empty_artifacts_array(self, mock_get_config, mock_async_client_class, sample_image_bytes):
        """Test 3D generation handles empty artifacts array in JSON response."""
        # Mock config
        mock_config = Mock()
        mock_config.get_trellis_config.return_value = {
            'url': 'http://test-trellis:8000/v1/infer'
        }
        mock_get_config.return_value = mock_config
        
        # Mock response with empty artifacts
        fake_glb_data = b"FALLBACK_GLB"
        mock_response = Mock()
        mock_response.content = fake_glb_data
        mock_response.json.return_value = {
            "artifacts": []  # Empty array
        }
        mock_response.raise_for_status = Mock()
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_async_client_class.return_value = mock_client_instance
        
        # Call function
        result = await generate_3d_asset(
            image_bytes=sample_image_bytes,
            content_type="image/png",
            seed=888
        )
        
        # Should fall back to response.content
        assert result["glb_data"] == fake_glb_data
        assert result["artifact_id"] == "trellis_888"

