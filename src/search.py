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
Search Module

Provides search functionality for the medical fact-checking system.
Includes vector similarity search and naive RAG implementation.
"""

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

model: SentenceTransformer | None = None
pubmed_embeddings: np.ndarray | None = None
pubmed_data: list[dict[str, Any]] | None = None


def get_model() -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global model
    if model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded successfully")
    return model


def load_embeddings(
    embeddings_path: str = "embeddings/pubmed_embeddings.npy",
    data_path: str = "data/pubmedqa_clean.json",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Load embeddings and corresponding data.

    Args:
        embeddings_path: Path to the embeddings file
        data_path: Path to the JSON data file

    Returns:
        Tuple of (embeddings array, data list)
    """
    global pubmed_embeddings, pubmed_data

    if pubmed_embeddings is None or pubmed_data is None:
        import json

        logger.info("Loading embeddings from: %s", embeddings_path)
        pubmed_embeddings = np.load(embeddings_path)
        logger.info("Embeddings shape: %s", pubmed_embeddings.shape)

        logger.info("Loading data from: %s", data_path)
        with open(data_path, encoding="utf-8") as f:
            pubmed_data = json.load(f)
        logger.info("Loaded %d data records", len(pubmed_data))

    return pubmed_embeddings, pubmed_data


def cosine_similarity(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and corpus embeddings.

    Args:
        query_embedding: Query vector (D,)
        corpus_embeddings: Corpus matrix (N, D)

    Returns:
        numpy.ndarray: Similarity scores (N,)
    """
    return np.dot(corpus_embeddings, query_embedding)


def search_pubmed_naive(
    query: str,
    top_k: int = 5,
    embeddings: np.ndarray | None = None,
    data: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Naive RAG search: only vector similarity, no graph traversal.

    Args:
        query: Query text
        top_k: Number of results to return
        embeddings: Optional pre-loaded embeddings
        data: Optional pre-loaded data

    Returns:
        List of similar documents with scores
    """
    if embeddings is None or data is None:
        embeddings, data = load_embeddings()

    embedder = get_model()
    query_embedding = embedder.encode(query, normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "question": data[idx]["question"],
            "context": data[idx]["context"][:300] + "...",
            "answer": data[idx]["answer"],
            "label": data[idx].get("label", "unknown"),
        })

    return results


def search_similar(
    query: str,
    embeddings: np.ndarray,
    data: list[dict[str, Any]],
    embedder: SentenceTransformer,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Search for similar documents using cosine similarity.

    Args:
        query: Query text
        embeddings: Embedding matrix
        data: Original data list
        embedder: Sentence transformer model
        top_k: Number of results to return

    Returns:
        List of top-k similar documents with scores
    """
    query_embedding = embedder.encode(query, normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "question": data[idx]["question"],
            "context": data[idx]["context"][:300] + "...",
            "answer": data[idx]["answer"],
        })

    return results
