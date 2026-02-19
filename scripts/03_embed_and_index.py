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
Embedding Generation Module

This module generates vector embeddings for medical texts using
sentence-transformers, enabling semantic similarity search.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def generate_embeddings(
    data_path: str | None = None,
    output_path: str | None = None,
) -> np.ndarray:
    """
    Generate embeddings for PubMedQA data.

    Args:
        data_path: Path to cleaned PubMedQA JSON
        output_path: Path to save embeddings

    Returns:
        numpy.ndarray: Embedding matrix
    """
    if data_path is None:
        data_path = str(settings.data_dir / "pubmedqa_clean.json")
    if output_path is None:
        output_path = str(settings.embeddings_dir / "pubmed_embeddings.npy")

    logger.info("Loading embedding model: %s", settings.embedding_model)

    try:
        model = SentenceTransformer(settings.embedding_model)
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e, exc_info=True)
        raise

    logger.info("Loading data from %s...", data_path)
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error("Data file not found at %s", data_path)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in data file: %s", e)
        raise

    contexts = [item["context"] for item in data]

    logger.info(
        "Generating embeddings for %d documents (dimension: %d)...",
        len(contexts),
        settings.embedding_dim,
    )

    try:
        embeddings = model.encode(
            contexts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    except Exception as e:
        logger.error("Failed to generate embeddings: %s", e, exc_info=True)
        raise

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    logger.info("Saved embeddings to %s", output_path)
    logger.info("Embedding shape: %s", embeddings.shape)

    return embeddings


if __name__ == "__main__":
    generate_embeddings()
