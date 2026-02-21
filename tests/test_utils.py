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
Unit tests for the shared parse_llm_json utility.
"""
import json
import pytest
from backend.utils import parse_llm_json


class TestParseLlmJsonBasic:
    """Tests for basic JSON parsing."""

    def test_valid_json_dict(self):
        """Test parsing a plain valid JSON dict."""
        text = '{"title": "Test", "price": 19.99}'
        result = parse_llm_json(text)
        assert result == {"title": "Test", "price": 19.99}

    def test_valid_json_with_whitespace(self):
        """Test parsing JSON with leading/trailing whitespace."""
        text = '  \n {"title": "Test"} \n  '
        result = parse_llm_json(text)
        assert result == {"title": "Test"}

    def test_returns_none_for_non_dict_json(self):
        """Test that JSON arrays, strings, numbers return None."""
        assert parse_llm_json('["a", "b"]') is None
        assert parse_llm_json('"just a string"') is None
        assert parse_llm_json("42") is None
        assert parse_llm_json("true") is None

    def test_returns_none_for_invalid_json(self):
        """Test that invalid JSON returns None."""
        assert parse_llm_json("This is not JSON") is None
        assert parse_llm_json("{invalid: json}") is None
        assert parse_llm_json("{'single': 'quotes'}") is None

    def test_returns_none_for_empty_string(self):
        """Test that empty or whitespace-only strings return None."""
        assert parse_llm_json("") is None
        assert parse_llm_json("   ") is None
        assert parse_llm_json("\n\t") is None

    def test_nested_json(self):
        """Test parsing nested JSON structures."""
        data = {"product": {"title": "Test", "tags": ["a", "b"]}}
        result = parse_llm_json(json.dumps(data))
        assert result == data

    def test_unicode_json(self):
        """Test parsing JSON with Unicode characters."""
        data = {"title": "Bolso Elegante", "description": "Crème brûlée"}
        result = parse_llm_json(json.dumps(data, ensure_ascii=False))
        assert result == data


class TestParseLlmJsonMarkdownFences:
    """Tests for stripping markdown code fences."""

    def test_strips_json_fence(self):
        """Test stripping ```json ... ``` fences."""
        data = {"title": "Test"}
        text = f"```json\n{json.dumps(data)}\n```"
        assert parse_llm_json(text) == data

    def test_strips_generic_fence(self):
        """Test stripping ``` ... ``` fences."""
        data = {"value": 85.0, "issues": []}
        text = f"```\n{json.dumps(data)}\n```"
        assert parse_llm_json(text) == data

    def test_strips_fence_with_surrounding_text(self):
        """Test stripping fences when there is text before/after."""
        data = {"title": "Test"}
        text = f"Here is the JSON:\n```json\n{json.dumps(data)}\n```\nDone."
        assert parse_llm_json(text) == data

    def test_json_fence_preferred_over_generic(self):
        """Test that ```json fence is tried first."""
        data = {"title": "Correct"}
        text = f"```json\n{json.dumps(data)}\n```"
        assert parse_llm_json(text) == data


class TestParseLlmJsonExtractBraces:
    """Tests for extract_braces option."""

    def test_extracts_json_from_surrounding_text(self):
        """Test extracting JSON from surrounding prose."""
        data = {"title": "Test", "score": 42}
        text = f'Here is the result: {json.dumps(data)} -- end'
        result = parse_llm_json(text, extract_braces=True)
        assert result == data

    def test_extracts_outermost_braces(self):
        """Test that outermost braces are used for nested objects."""
        text = '{"outer": {"inner": "value"}}'
        result = parse_llm_json(text, extract_braces=True)
        assert result == {"outer": {"inner": "value"}}

    def test_no_braces_returns_none(self):
        """Test that text without braces returns None."""
        result = parse_llm_json("no braces here", extract_braces=True)
        assert result is None

    def test_disabled_by_default(self):
        """Test that brace extraction is off by default."""
        text = 'prefix {"title": "Test"} suffix'
        # Without extract_braces, this is not valid JSON
        assert parse_llm_json(text) is None


class TestParseLlmJsonStripComments:
    """Tests for strip_comments option."""

    def test_strips_line_comments(self):
        """Test stripping // line comments."""
        text = '{\n"title": "Test", // this is a comment\n"score": 42\n}'
        result = parse_llm_json(text, strip_comments=True)
        assert result == {"title": "Test", "score": 42}

    def test_strips_block_comments(self):
        """Test stripping /* */ block comments."""
        text = '{"title": "Test", /* block comment */ "score": 42}'
        result = parse_llm_json(text, strip_comments=True)
        assert result == {"title": "Test", "score": 42}

    def test_strips_multiline_block_comments(self):
        """Test stripping multi-line block comments."""
        text = '{"title": "Test", /* multi\nline\ncomment */ "score": 42}'
        result = parse_llm_json(text, strip_comments=True)
        assert result == {"title": "Test", "score": 42}

    def test_disabled_by_default(self):
        """Test that comment stripping is off by default."""
        text = '{"title": "Test" // comment\n}'
        # Without strip_comments, JSON with comments is invalid
        assert parse_llm_json(text) is None


class TestParseLlmJsonCombined:
    """Tests for combined options."""

    def test_fence_plus_extract_braces_plus_strip_comments(self):
        """Test all options together, simulating a messy LLM response."""
        inner = '{\n"title": "Product", // name\n"score": 99 /* perfect */\n}'
        text = f"Here is the JSON:\n```json\n{inner}\n```\nDone."
        result = parse_llm_json(text, extract_braces=True, strip_comments=True)
        assert result == {"title": "Product", "score": 99}

    def test_extract_braces_with_strip_comments(self):
        """Test brace extraction and comment stripping together."""
        text = 'Result: {"value": 75.0, // quality score\n"issues": []} end'
        result = parse_llm_json(text, extract_braces=True, strip_comments=True)
        assert result == {"value": 75.0, "issues": []}

    def test_fence_stripping_then_extract_braces(self):
        """Test that fence stripping happens before brace extraction."""
        data = {"title": "Test"}
        text = f"```json\nHere: {json.dumps(data)}\n```"
        # Fence stripping extracts "Here: {...}", then brace extraction gets the dict
        result = parse_llm_json(text, extract_braces=True)
        assert result == data
