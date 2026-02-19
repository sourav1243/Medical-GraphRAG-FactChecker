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


def test_settings_loads_from_env(monkeypatch):
    """Test that settings loads correctly from environment variables."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    from importlib import reload
    import src.config

    reload(src.config)

    settings = src.config.settings
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dim == 1024


def test_settings_fails_without_required(monkeypatch):
    """Test that Settings raises ValidationError for missing required fields."""
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    import src.config
    from importlib import reload

    reload(src.config)

    with pytest.raises(ValidationError):
        src.config.Settings()


def test_default_values(monkeypatch):
    """Test that default values are set correctly."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    import src.config
    from importlib import reload

    reload(src.config)

    settings = src.config.Settings()
    assert settings.neo4j_username == "neo4j"
    assert settings.top_k_retrieval == 8
    assert settings.similarity_threshold_match == 0.85
    assert settings.similarity_threshold_weak == 0.75


def test_data_dir_property(monkeypatch):
    """Test the data_dir property returns correct path."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    import src.config
    from importlib import reload

    reload(src.config)

    settings = src.config.Settings()
    assert "data" in str(settings.data_dir)
