"""
Embedding Generation Module

This module generates vector embeddings for medical texts using
sentence-transformers, enabling semantic similarity search.
"""

import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings(
    data_path: str = "data/pubmedqa_clean.json",
    output_path: str = "embeddings/pubmed_embeddings.npy"
) -> np.ndarray:
    """
    Generate embeddings for PubMedQA data.
    
    Args:
        data_path: Path to cleaned PubMedQA JSON
        output_path: Path to save embeddings
    
    Returns:
        numpy.ndarray: Embedding matrix
    """
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    contexts = [item["context"] for item in data]
    
    print(f"Generating embeddings for {len(contexts)} documents...")
    embeddings = model.encode(
        contexts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    
    print(f"Saved embeddings to {output_path}")
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings


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


def search_similar(
    query: str,
    embeddings: np.ndarray,
    data: list,
    model,
    top_k: int = 5
) -> list:
    """
    Search for similar documents using cosine similarity.
    
    Args:
        query: Query text
        embeddings: Embedding matrix
        data: Original data list
        model: Sentence transformer model
        top_k: Number of results to return
    
    Returns:
        list: Top-k similar documents with scores
    """
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "question": data[idx]["question"],
            "context": data[idx]["context"][:300] + "...",
            "answer": data[idx]["answer"]
        })
    
    return results


if __name__ == "__main__":
    generate_embeddings()
