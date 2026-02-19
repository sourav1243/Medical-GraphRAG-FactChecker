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
Configuration Module

Manages environment variables and configuration settings for the application.
Uses pydantic-settings for robust configuration management with validation.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with pydantic validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    neo4j_uri: str = Field(..., description="Neo4j AuraDB connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")

    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_model: str = Field(
        default="google/gemini-2.0-flash-001",
        description="OpenRouter model name",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Embedding model name",
    )
    embedding_dim: int = Field(default=1024, description="Embedding dimension")

    pubmed_sample_size: int = Field(default=200, description="PubMed sample size")
    top_k_retrieval: int = Field(default=8, description="Top-k for retrieval")
    similarity_threshold_match: float = Field(
        default=0.85, description="Threshold for strong match"
    )
    similarity_threshold_weak: float = Field(
        default=0.75, description="Threshold for weak match"
    )

    @property
    def base_dir(self) -> Path:
        """Get the base directory of the project."""
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.base_dir / "data"

    @property
    def embeddings_dir(self) -> Path:
        """Get the embeddings directory."""
        return self.base_dir / "embeddings"


settings = Settings()
