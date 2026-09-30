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
Unit tests for image quality reflection module.

Tests the reflection/evaluation system that compares original and generated images.
"""
import json
import pytest
from unittest.mock import Mock, patch
from backend.reflection import (
    REFLECTION_PROMPT_TEMPLATE,
    evaluate_image_quality,
    _encode_image_to_base64,
    _parse_quality_response,
    _build_reflection_messages,
    _count_message_images,
    _format_generation_prompt_section,
)


class TestEvaluateImageQuality:
    """Tests for evaluate_image_quality function."""
    
    @patch('backend.reflection.OpenAI')
    @patch('backend.reflection.get_config')
    def test_evaluation_success_with_perfect_score(
        self, mock_get_config, mock_openai_class, sample_image_bytes, mock_env_vars
    ):
        """Test successful evaluation with perfect score (100%)."""
        # Mock config
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-vlm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock VLM response with perfect score
        mock_message = Mock()
        mock_message.content = json.dumps({
            "value": 100.0,
            "rationale": "The generated image preserves the same product structure, color, labels, and proportions.",
            "issues": []
        })
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Call function
        result = evaluate_image_quality(
            original_image_bytes=sample_image_bytes,
            generated_image_bytes=sample_image_bytes,
            content_type="image/png"
        )
        
        # Assertions
        assert result is not None
        assert isinstance(result, dict)
        assert result["score"] == 100.0
        assert "same product structure" in result["rationale"]
        assert result["issues"] == []
        assert isinstance(result["issues"], list)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert _count_message_images(messages) == 2
        assert call_args.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
        assert call_args.kwargs["temperature"] == 0.0
        assert call_args.kwargs["max_tokens"] == 8192

    @patch('backend.reflection.OpenAI')
    @patch('backend.reflection.get_config')
    def test_evaluation_includes_generation_prompt_context(
        self, mock_get_config, mock_openai_class, sample_image_bytes, mock_env_vars
    ):
        """The judge should know which background/context changes were intended."""
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-vlm-model'
        }
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps({"value": 95.0, "issues": []})
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        generation_prompt = (
            "Keep the floor-standing product unchanged. "
            "Replace only the background with a plausible outdoor usage scene."
        )

        evaluate_image_quality(
            original_image_bytes=sample_image_bytes,
            generated_image_bytes=sample_image_bytes,
            content_type="image/png",
            product_title="Floor-Standing Product",
            generation_prompt=generation_prompt,
        )

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        judge_prompt = messages[0]["content"][-1]["text"]
        assert "EXPECTED GENERATION INTENT" in judge_prompt
        assert generation_prompt in judge_prompt
        assert "Do not list expected contextual/background differences as quality issues" in judge_prompt
        assert "Penalize the background or staging if it corrupts" in judge_prompt
        assert "Functional realism" in judge_prompt
        assert "support surface" in judge_prompt
        assert "make real-world sense" in judge_prompt
        assert "Always include a concise rationale explaining the score" in judge_prompt
        assert '"rationale": "concise explanation of the score"' in judge_prompt
    
    @patch('backend.reflection.OpenAI')
    @patch('backend.reflection.get_config')
    def test_evaluation_with_multiple_issues(
        self, mock_get_config, mock_openai_class, sample_image_bytes, mock_env_vars
    ):
        """Test evaluation that detects multiple quality issues."""
        # Mock config
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-vlm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock VLM response with issues
        mock_message = Mock()
        mock_message.content = json.dumps({
            "value": 65.5,
            "issues": [
                "Product appears oversized relative to background furniture",
                "Slight color shift detected in product",
                "Background context inappropriate for product type"
            ]
        })
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Call function
        result = evaluate_image_quality(
            original_image_bytes=sample_image_bytes,
            generated_image_bytes=sample_image_bytes,
            content_type="image/png"
        )
        
        # Assertions
        assert result is not None
        assert result["score"] == 65.5
        assert len(result["issues"]) == 3
        assert "oversized" in result["issues"][0].lower()
        assert "color shift" in result["issues"][1].lower()
    
    @patch('backend.reflection.OpenAI')
    @patch('backend.reflection.get_config')
    def test_evaluation_extracts_json_from_markdown(
        self, mock_get_config, mock_openai_class, sample_image_bytes, mock_env_vars
    ):
        """Test evaluation handles JSON wrapped in markdown code blocks."""
        # Mock config
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-vlm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock VLM response with markdown formatting
        response_data = {
            "value": 87.0,
            "issues": ["Minor background blur"]
        }
        markdown_response = f"```json\n{json.dumps(response_data)}\n```"
        
        mock_message = Mock()
        mock_message.content = markdown_response
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Call function
        result = evaluate_image_quality(
            original_image_bytes=sample_image_bytes,
            generated_image_bytes=sample_image_bytes,
            content_type="image/png"
        )
        
        # Should extract JSON successfully
        assert result is not None
        assert result["score"] == 87.0
        assert len(result["issues"]) == 1
    
    @patch('backend.reflection.OpenAI')
    @patch('backend.reflection.get_config')
    def test_evaluation_handles_missing_issues_field(
        self, mock_get_config, mock_openai_class, sample_image_bytes, mock_env_vars
    ):
        """Test evaluation handles response missing issues field."""
        # Mock config
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-vlm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock VLM response without issues field
        mock_message = Mock()
        mock_message.content = json.dumps({
            "value": 75.0
            # No "issues" field
        })
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Call function
        result = evaluate_image_quality(
            original_image_bytes=sample_image_bytes,
            generated_image_bytes=sample_image_bytes,
            content_type="image/png"
        )
        
        # Should default to empty issues list
        assert result is not None
        assert result["score"] == 75.0
        assert result["issues"] == []
    
    @patch('backend.reflection.OpenAI')
    @patch('backend.reflection.get_config')
    def test_evaluation_handles_api_error(
        self, mock_get_config, mock_openai_class, sample_image_bytes, mock_env_vars
    ):
        """Test evaluation handles API errors gracefully."""
        # Mock config
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-vlm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client to raise exception
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Call function
        result = evaluate_image_quality(
            original_image_bytes=sample_image_bytes,
            generated_image_bytes=sample_image_bytes,
            content_type="image/png"
        )
        
        # Should return None on error
        assert result is None
    
    @patch('backend.reflection.OpenAI')
    @patch('backend.reflection.get_config')
    def test_evaluation_score_clamped_to_valid_range(
        self, mock_get_config, mock_openai_class, sample_image_bytes, mock_env_vars
    ):
        """Test evaluation clamps scores to 0-100 range."""
        # Mock config
        mock_config = Mock()
        mock_config.get_vlm_config.return_value = {
            'url': 'http://test:8000/v1',
            'model': 'test-vlm-model'
        }
        mock_get_config.return_value = mock_config
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock VLM response with out-of-range score
        mock_message = Mock()
        mock_message.content = json.dumps({
            "value": 150.0,  # Invalid (> 100)
            "issues": []
        })
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Call function
        result = evaluate_image_quality(
            original_image_bytes=sample_image_bytes,
            generated_image_bytes=sample_image_bytes,
            content_type="image/png"
        )
        
        # Score should be clamped to 100
        assert result is not None
        assert result["score"] == 100.0

    def test_reflection_payload_sends_two_labeled_images(self):
        """Reflection payload must send original and generated images to Omni."""
        messages = _build_reflection_messages(
            original_image_b64="original-b64",
            generated_image_b64="generated-b64",
            reflection_prompt="Judge these images.",
        )

        assert len(messages) == 1
        content = messages[0]["content"]
        assert _count_message_images(messages) == 2
        assert content[0] == {"type": "text", "text": "IMAGE 1 - ORIGINAL REFERENCE product image:"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/png;base64,original-b64"
        assert content[2] == {"type": "text", "text": "IMAGE 2 - GENERATED VARIATION to judge against Image 1:"}
        assert content[3]["type"] == "image_url"
        assert content[3]["image_url"]["url"] == "data:image/png;base64,generated-b64"
        assert content[4]["text"] == "Judge these images."

    def test_count_message_images_ignores_non_image_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}},
                    {"type": "input_image", "image_url": {"url": "data:image/png;base64,b"}},
                ],
            },
            {"role": "assistant", "content": "ignored"},
        ]

        assert _count_message_images(messages) == 1

    def test_format_generation_prompt_section_has_fallback(self):
        section = _format_generation_prompt_section("")

        assert "No image-edit prompt was provided" in section
        assert "different background/context" in section


class TestEncodeImageToBase64:
    """Tests for _encode_image_to_base64 helper function."""
    
    def test_encode_png_image(self, sample_image_bytes):
        """Test encoding PNG image to base64."""
        result = _encode_image_to_base64(sample_image_bytes, "png")
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should be valid base64
        import base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    
    def test_encode_jpeg_image(self, sample_jpeg_bytes):
        """Test encoding JPEG image to base64."""
        result = _encode_image_to_base64(sample_jpeg_bytes, "jpeg")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_encode_handles_rgba_to_jpeg_conversion(self, sample_image_bytes):
        """Test RGBA to RGB conversion for JPEG format."""
        from PIL import Image
        from io import BytesIO
        
        # Create RGBA image
        rgba_img = Image.new('RGBA', (10, 10), color=(255, 0, 0, 128))
        buffer = BytesIO()
        rgba_img.save(buffer, format='PNG')
        rgba_bytes = buffer.getvalue()
        
        # Should convert to RGB for JPEG
        result = _encode_image_to_base64(rgba_bytes, "jpeg")
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestParseQualityResponse:
    """Tests for _parse_quality_response parser function."""
    
    def test_parse_valid_json_with_issues(self):
        """Test parsing valid JSON with score and issues."""
        response = json.dumps({
            "value": 85.5,
            "issues": ["Issue 1", "Issue 2"]
        })
        
        result = _parse_quality_response(response)
        
        assert result is not None
        assert result["score"] == 85.5
        assert len(result["issues"]) == 2
        assert result["issues"][0] == "Issue 1"
    
    def test_parse_json_without_issues(self):
        """Test parsing JSON without issues field."""
        response = json.dumps({
            "value": 90.0
        })
        
        result = _parse_quality_response(response)
        
        assert result is not None
        assert result["score"] == 90.0
        assert result["rationale"] == ""
        assert result["issues"] == []
    
    def test_parse_json_in_markdown_block(self):
        """Test parsing JSON wrapped in markdown code block."""
        data = {"value": 75.0, "rationale": "Visible distortion.", "issues": ["Test issue"]}
        response = f"```json\n{json.dumps(data)}\n```"
        
        result = _parse_quality_response(response)
        
        assert result is not None
        assert result["score"] == 75.0
        assert result["rationale"] == "Visible distortion."
        assert len(result["issues"]) == 1
    
    def test_parse_json_in_generic_code_block(self):
        """Test parsing JSON in generic code block."""
        data = {"value": 80.0, "issues": []}
        response = f"```\n{json.dumps(data)}\n```"
        
        result = _parse_quality_response(response)
        
        assert result is not None
        assert result["score"] == 80.0
    
    def test_parse_invalid_json_returns_none(self):
        """Test parsing invalid JSON returns None."""
        response = "This is not valid JSON"
        
        result = _parse_quality_response(response)
        
        assert result is None
    
    def test_parse_json_missing_value_returns_none(self):
        """Test parsing JSON without 'value' field returns None."""
        response = json.dumps({
            "score": 85.0,  # Wrong key name
            "issues": []
        })
        
        result = _parse_quality_response(response)
        
        assert result is None
    
    def test_parse_clamps_score_to_valid_range(self):
        """Test parser clamps score to 0-100 range."""
        # Test upper bound
        response_high = json.dumps({"value": 150.0, "issues": []})
        result_high = _parse_quality_response(response_high)
        assert result_high["score"] == 100.0
        
        # Test lower bound
        response_low = json.dumps({"value": -10.0, "issues": []})
        result_low = _parse_quality_response(response_low)
        assert result_low["score"] == 0.0
    
    def test_parse_handles_non_list_issues(self):
        """Test parser handles non-list issues field."""
        response = json.dumps({
            "value": 70.0,
            "issues": "This should be a list"  # Wrong type
        })
        
        result = _parse_quality_response(response)
        
        # Should default to empty list
        assert result is not None
        assert result["score"] == 70.0
        assert result["issues"] == []
    
    def test_parse_handles_empty_issues_array(self):
        """Test parser handles empty issues array."""
        response = json.dumps({
            "value": 100.0,
            "issues": []
        })
        
        result = _parse_quality_response(response)
        
        assert result is not None
        assert result["score"] == 100.0
        assert result["issues"] == []

    def test_parse_preserves_rationale_text(self):
        """Test parser preserves the judge rationale."""
        response = json.dumps({
            "value": 100.0,
            "rationale": "Product identity, structure, color, labels, and proportions all match.",
            "issues": [],
        })

        result = _parse_quality_response(response)

        assert result is not None
        assert result["score"] == 100.0
        assert result["rationale"] == "Product identity, structure, color, labels, and proportions all match."
        assert result["issues"] == []
    
    def test_parse_preserves_issue_text_exactly(self):
        """Test parser preserves issue text without modification."""
        issues_text = [
            "Product appears oversized relative to background furniture",
            "Slight color shift detected (darker than original)",
            "Left hand has 6 fingers"
        ]
        
        response = json.dumps({
            "value": 55.0,
            "issues": issues_text
        })
        
        result = _parse_quality_response(response)
        
        assert result is not None
        assert result["issues"] == issues_text
        for i, issue in enumerate(result["issues"]):
            assert issue == issues_text[i]


class TestReflectionTextFidelity:
    """Guards for the text-transcription and anti-anchoring judge instructions."""

    def test_prompt_requires_text_transcription(self):
        prompt = REFLECTION_PROMPT_TEMPLATE.format(product_name="bag", generation_prompt_section="x")

        assert "original_text" in prompt
        assert "generated_text" in prompt
        assert "cap the score at 84 or lower" in prompt

    def test_prompt_does_not_trust_generation_intent(self):
        prompt = REFLECTION_PROMPT_TEMPLATE.format(product_name="bag", generation_prompt_section="x")

        assert "Do not assume the product was preserved" in prompt
        assert "does not guarantee the product was preserved" in _format_generation_prompt_section("Keep bag")

    def test_parse_ignores_reasoning_block(self):
        result = _parse_quality_response(
            '<think>compare labels</think>{"original_text": ["A"], "generated_text": [], '
            '"value": 70, "rationale": "text removed", "issues": ["label missing"]}'
        )

        assert result is not None
        assert result["score"] == 70.0
        assert result["issues"] == ["label missing"]
