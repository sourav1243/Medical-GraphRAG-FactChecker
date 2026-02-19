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
Utility Functions Module

Provides utility functions for JSON parsing, text preprocessing, and other
common operations used across the application.
"""

import json
import re
from typing import Any


def safe_parse_llm_json(raw: str) -> dict[str, Any]:
    """
    Extract and parse JSON from an LLM response that may contain extra text.

    This function handles cases where the LLM returns JSON embedded in
    explanatory text or with preamble content.

    Args:
        raw: The raw string response from the LLM

    Returns:
        A dictionary containing the parsed JSON, or an error response
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {
            "verdict": "ERROR",
            "error": "no_json_in_response",
            "raw": raw[:200],
        }

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        return {
            "verdict": "ERROR",
            "error": f"json_parse_failed: {e}",
            "raw": raw[:200],
        }


def preprocess_claim(text: str) -> str:
    """
    Preprocess a claim text for processing.

    Strips whitespace and validates that the input is not empty.

    Args:
        text: The claim text to preprocess

    Returns:
        The cleaned claim text

    Raises:
        ValueError: If the input text is empty or whitespace-only
    """
    if not text or not text.strip():
        raise ValueError("Claim text cannot be empty")

    return text.strip()


def detect_language(text: str) -> str:
    """
    Detect the language of the input text based on character ranges.

    This is a simple heuristic-based detection. For production use,
    consider using a proper language detection library like langdetect.

    Args:
        text: The text to detect language for

    Returns:
        ISO 639-1 language code (e.g., 'en', 'hi', 'de', 'ta', 'te')
    """
    if not text:
        return "en"

    text = text.strip()

    hindi_range = re.compile(r"[\u0900-\u097F]")
    if hindi_range.search(text):
        return "hi"

    german_range = re.compile(r"[äöüßÄÖÜ]")
    if german_range.search(text):
        return "de"

    tamil_range = re.compile(r"[\u0B80-\u0BFF]")
    if tamil_range.search(text):
        return "ta"

    telugu_range = re.compile(r"[\u0C00-\u0C7F]")
    if telugu_range.search(text):
        return "te"

    return "en"
