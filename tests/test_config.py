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
Test suite for configuration management.
"""

import pytest
from pydantic import ValidationError


def test_settings_has_required_fields(monkeypatch):
    """Test that Settings class has the expected required fields."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    from src.config import Settings

    s = Settings()
    assert hasattr(s, "neo4j_uri")
    assert hasattr(s, "embedding_model")
    assert hasattr(s, "embedding_dim")


def test_settings_default_values(monkeypatch):
    """Test that default values are set correctly."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    from src.config import Settings

    s = Settings()
    assert s.neo4j_username == "neo4j"
    assert s.top_k_retrieval == 8
    assert s.similarity_threshold_match == 0.85
    assert s.similarity_threshold_weak == 0.75


def test_embedding_model_default(monkeypatch):
    """Test that embedding model defaults to BAAI/bge-m3."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    from src.config import Settings

    s = Settings()
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024


def test_data_dir_property(monkeypatch):
    """Test the data_dir property returns correct path."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    from src.config import Settings

    s = Settings()
    assert "data" in str(s.data_dir)
    assert "embeddings" in str(s.embeddings_dir)
