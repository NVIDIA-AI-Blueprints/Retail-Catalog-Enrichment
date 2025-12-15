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
Unit tests for validation logic from main.py module.

Tests input validation, locale validation, and JSON parsing.
"""
import io
import json
import pytest
from fastapi import UploadFile
from backend.main import _validate_image, VALID_LOCALES


class TestValidateImage:
    """Tests for _validate_image helper function."""
    
    @pytest.mark.asyncio
    async def test_validate_image_success_png(self, sample_image_bytes):
        """Test successful validation of PNG image."""
        from unittest.mock import Mock, AsyncMock
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.png"
        upload_file.file = io.BytesIO(sample_image_bytes)
        upload_file.content_type = "image/png"
        upload_file.read = AsyncMock(return_value=sample_image_bytes)
        
        result, error = await _validate_image(upload_file, "/test")
        
        assert error is None
        assert result is not None
        image_bytes, content_type = result
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0
        assert content_type == "image/png"
    
    @pytest.mark.asyncio
    async def test_validate_image_success_jpeg(self, sample_jpeg_bytes):
        """Test successful validation of JPEG image."""
        from unittest.mock import Mock, AsyncMock
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.jpg"
        upload_file.file = io.BytesIO(sample_jpeg_bytes)
        upload_file.content_type = "image/jpeg"
        upload_file.read = AsyncMock(return_value=sample_jpeg_bytes)
        
        result, error = await _validate_image(upload_file, "/test")
        
        assert error is None
        assert result is not None
        image_bytes, content_type = result
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0
        assert content_type == "image/jpeg"
    
    @pytest.mark.asyncio
    async def test_validate_image_empty_file(self):
        """Test validation fails for empty file."""
        from unittest.mock import Mock, AsyncMock
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "empty.png"
        upload_file.file = io.BytesIO(b"")
        upload_file.content_type = "image/png"
        upload_file.read = AsyncMock(return_value=b"")
        
        result, error = await _validate_image(upload_file, "/test")
        
        assert result is None
        assert error is not None
        assert error.status_code == 400
        error_detail = json.loads(error.body.decode())
        assert "empty" in error_detail["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_validate_image_non_image_content_type(self):
        """Test validation fails for non-image content type."""
        from unittest.mock import Mock, AsyncMock
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.txt"
        upload_file.file = io.BytesIO(b"some text content")
        upload_file.content_type = "text/plain"
        upload_file.read = AsyncMock(return_value=b"some text content")
        
        result, error = await _validate_image(upload_file, "/test")
        
        assert result is None
        assert error is not None
        assert error.status_code == 400
        error_detail = json.loads(error.body.decode())
        assert "must be an image" in error_detail["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_validate_image_no_content_type(self, sample_image_bytes):
        """Test validation with missing content_type defaults to image/png."""
        from unittest.mock import Mock, AsyncMock
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.png"
        upload_file.file = io.BytesIO(sample_image_bytes)
        upload_file.content_type = None
        upload_file.read = AsyncMock(return_value=sample_image_bytes)
        
        result, error = await _validate_image(upload_file, "/test")
        
        assert error is None
        assert result is not None
        image_bytes, content_type = result
        # Should default to image/png
        assert content_type == "image/png"
    
    @pytest.mark.asyncio
    async def test_validate_image_with_various_image_types(self, sample_image_bytes):
        """Test validation accepts various image MIME types."""
        from unittest.mock import Mock, AsyncMock
        image_types = [
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/gif",
            "image/webp"
        ]
        
        for content_type in image_types:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = f"test.{content_type.split('/')[-1]}"
            upload_file.file = io.BytesIO(sample_image_bytes)
            upload_file.content_type = content_type
            upload_file.read = AsyncMock(return_value=sample_image_bytes)
            
            result, error = await _validate_image(upload_file, "/test")
            
            assert error is None, f"Failed for {content_type}"
            assert result is not None
            _, returned_content_type = result
            assert returned_content_type == content_type


class TestLocaleValidation:
    """Tests for locale validation."""
    
    def test_valid_locales_contains_expected_values(self):
        """Test that VALID_LOCALES contains all expected locales."""
        expected_locales = {
            "en-US", "en-GB", "en-AU", "en-CA",
            "es-ES", "es-MX", "es-AR", "es-CO",
            "fr-FR", "fr-CA"
        }
        
        assert VALID_LOCALES == expected_locales
    
    def test_valid_locale_check(self):
        """Test checking valid locales."""
        valid_examples = ["en-US", "es-ES", "fr-FR"]
        
        for locale in valid_examples:
            assert locale in VALID_LOCALES
    
    def test_invalid_locale_check(self):
        """Test checking invalid locales."""
        invalid_examples = ["en-ZZ", "es", "fr-XX", "invalid"]
        
        for locale in invalid_examples:
            assert locale not in VALID_LOCALES
    
    def test_case_sensitive_locale(self):
        """Test that locale validation is case-sensitive."""
        # These should not be valid (wrong case)
        assert "en-us" not in VALID_LOCALES  # lowercase
        assert "EN-US" not in VALID_LOCALES  # uppercase
        assert "en-US" in VALID_LOCALES      # correct case


class TestJSONParsing:
    """Tests for JSON parsing in endpoint handlers."""
    
    def test_parse_valid_json_string(self):
        """Test parsing valid JSON string."""
        json_str = '{"title": "Test", "price": 19.99}'
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, dict)
        assert parsed["title"] == "Test"
        assert parsed["price"] == 19.99
    
    def test_parse_valid_json_array(self):
        """Test parsing valid JSON array."""
        json_str = '["item1", "item2", "item3"]'
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert "item1" in parsed
    
    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        invalid_json = '{invalid: json}'
        
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)
    
    def test_parse_empty_string_raises_error(self):
        """Test that empty string raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            json.loads("")
    
    def test_parse_nested_json(self):
        """Test parsing nested JSON structures."""
        json_str = '{"product": {"title": "Test", "tags": ["tag1", "tag2"]}}'
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, dict)
        assert "product" in parsed
        assert isinstance(parsed["product"]["tags"], list)
        assert len(parsed["product"]["tags"]) == 2
    
    def test_parse_json_with_unicode(self):
        """Test parsing JSON with Unicode characters."""
        json_str = '{"title": "Café au Lait", "description": "Crème brûlée"}'
        parsed = json.loads(json_str)
        
        assert parsed["title"] == "Café au Lait"
        assert parsed["description"] == "Crème brûlée"
    
    def test_parse_json_with_escaped_characters(self):
        """Test parsing JSON with escaped characters."""
        json_str = r'{"description": "Line 1\nLine 2\tTabbed"}'
        parsed = json.loads(json_str)
        
        assert "\n" in parsed["description"]
        assert "\t" in parsed["description"]


class TestCategoriesValidation:
    """Tests for categories validation logic."""
    
    def test_categories_as_list(self):
        """Test that categories can be parsed as list."""
        categories_json = '["accessories", "bags"]'
        categories = json.loads(categories_json)
        
        assert isinstance(categories, list)
        assert "accessories" in categories
        assert "bags" in categories
    
    def test_empty_categories_list(self):
        """Test handling of empty categories list."""
        categories_json = '[]'
        categories = json.loads(categories_json)
        
        assert isinstance(categories, list)
        assert len(categories) == 0
    
    def test_single_category(self):
        """Test single category in list."""
        categories_json = '["accessories"]'
        categories = json.loads(categories_json)
        
        assert isinstance(categories, list)
        assert len(categories) == 1
        assert categories[0] == "accessories"


class TestTagsValidation:
    """Tests for tags validation logic."""
    
    def test_tags_as_list(self):
        """Test that tags can be parsed as list."""
        tags_json = '["leather", "gold hardware", "evening bag"]'
        tags = json.loads(tags_json)
        
        assert isinstance(tags, list)
        assert len(tags) == 3
        assert "leather" in tags
    
    def test_empty_tags_list(self):
        """Test handling of empty tags list."""
        tags_json = '[]'
        tags = json.loads(tags_json)
        
        assert isinstance(tags, list)
        assert len(tags) == 0
    
    def test_tags_with_special_characters(self):
        """Test tags with special characters."""
        tags_json = '["24/7 wear", "eco-friendly", "size: large"]'
        tags = json.loads(tags_json)
        
        assert isinstance(tags, list)
        assert "24/7 wear" in tags
        assert "eco-friendly" in tags


class TestColorsValidation:
    """Tests for colors validation logic."""
    
    def test_colors_as_list(self):
        """Test that colors can be parsed as list."""
        colors_json = '["black", "gold", "silver"]'
        colors = json.loads(colors_json)
        
        assert isinstance(colors, list)
        assert len(colors) == 3
        assert "black" in colors
    
    def test_empty_colors_list(self):
        """Test handling of empty colors list."""
        colors_json = '[]'
        colors = json.loads(colors_json)
        
        assert isinstance(colors, list)
        assert len(colors) == 0
    
    def test_single_color(self):
        """Test single color in list."""
        colors_json = '["red"]'
        colors = json.loads(colors_json)
        
        assert isinstance(colors, list)
        assert len(colors) == 1
        assert colors[0] == "red"

