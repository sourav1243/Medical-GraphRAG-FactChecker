# Copyright 2025 Sourav Kumar Sharma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test suite for utility functions.
"""

import pytest

from src.utils import (
    detect_language,
    preprocess_claim,
    safe_parse_llm_json,
)


class TestSafeParseLLMJson:
    """Tests for safe_parse_llm_json function."""

    def test_parse_clean_json(self):
        """Test parsing clean JSON."""
        raw = '{"verdict": "SUPPORTED", "confidence_score": 0.9}'
        result = safe_parse_llm_json(raw)

        assert result["verdict"] == "SUPPORTED"
        assert result["confidence_score"] == 0.9

    def test_parse_json_with_preamble(self):
        """Test parsing JSON with preamble text."""
        raw = 'Sure! Here is the result: {"verdict": "CONTRADICTED", "confidence_score": 0.85}'
        result = safe_parse_llm_json(raw)

        assert result["verdict"] == "CONTRADICTED"
        assert result["confidence_score"] == 0.85

    def test_parse_json_with_extra_text(self):
        """Test parsing JSON with trailing text."""
        raw = '{"verdict": "UNVERIFIABLE"}\n\nThis claim cannot be verified based on current evidence.'
        result = safe_parse_llm_json(raw)

        assert result["verdict"] == "UNVERIFIABLE"

    def test_parse_no_json_returns_error(self):
        """Test handling of response with no JSON."""
        raw = "I cannot determine the verdict."
        result = safe_parse_llm_json(raw)

        assert result["verdict"] == "ERROR"
        assert "no_json_in_response" in result["error"]

    def test_parse_invalid_json_returns_error(self):
        """Test handling of invalid JSON."""
        raw = '{ "verdict": "SUPPORTED", broken }'
        result = safe_parse_llm_json(raw)

        assert result["verdict"] == "ERROR"
        assert "json_parse_failed" in result["error"]


class TestPreprocessClaim:
    """Tests for preprocess_claim function."""

    def test_preprocess_strips_whitespace(self):
        """Test that whitespace is stripped."""
        claim = "  Some medical claim  "
        result = preprocess_claim(claim)

        assert result == "Some medical claim"

    def test_preprocess_handles_hindi(self):
        """Test preprocessing Hindi text."""
        claim = "गर्म पानी पीने से कोरोना ठीक होता है।"
        result = preprocess_claim(claim)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_preprocess_handles_german(self):
        """Test preprocessing German text."""
        claim = "  Trinken von warmem Wasser heilt COVID-19.   "
        result = preprocess_claim(claim)

        assert result == "Trinken von warmem Wasser heilt COVID-19."

    def test_preprocess_handles_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            preprocess_claim("")

    def test_preprocess_handles_whitespace_only_raises(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError):
            preprocess_claim("   ")


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detect_english(self):
        """Test English detection."""
        assert detect_language("This is an English sentence.") == "en"
        assert detect_language("Hello world") == "en"

    def test_detect_hindi(self):
        """Test Hindi detection."""
        assert detect_language("गर्म पानी पीने से कोरोना ठीक होता है।") == "hi"
        assert detect_language("यह हिंदी में है") == "hi"

    def test_detect_german(self):
        """Test German detection."""
        assert detect_language("Wärmer") == "de"
        assert detect_language("Öffnen") == "de"
        assert detect_language("Über") == "de"

    def test_detect_tamil(self):
        """Test Tamil detection."""
        assert detect_language("மஞ்சள் பால்") == "ta"
        assert detect_language("துளசி இலைகள்") == "ta"

    def test_detect_telugu(self):
        """Test Telugu detection."""
        assert detect_language("తాగుడం") == "te"

    def test_detect_empty_string(self):
        """Test that empty string defaults to English."""
        assert detect_language("") == "en"

    def test_detect_english_numbers(self):
        """Test English with numbers defaults to English."""
        assert detect_language("COVID-19 treatment 2024") == "en"
