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
Fact-Check Pipeline

This module implements the complete fact-checking pipeline that combines
vector similarity search with knowledge graph traversal to verify medical claims.
"""

import json
import sys
from pathlib import Path

import numpy as np
import requests
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger
from src.utils import safe_parse_llm_json

logger = get_logger(__name__)


def call_llm(prompt: str, max_tokens: int = 1000) -> str:
    """
    Call OpenRouter API for LLM processing.

    Args:
        prompt: The prompt to send to the LLM
        max_tokens: Maximum tokens in the response

    Returns:
        str: The LLM response text, or empty string on failure
    """
    url = f"{settings.openrouter_base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": "https://github.com/sourav1243",
        "X-Title": "Medical-GraphRAG-FactChecker",
    }
    payload = {
        "model": settings.openrouter_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.Timeout:
        logger.error("LLM API call timed out after 60 seconds")
        return ""
    except requests.ConnectionError as e:
        logger.error("Cannot reach OpenRouter API: %s", e)
        return ""
    except requests.HTTPError as e:
        logger.error(
            "OpenRouter API returned HTTP %s: %s", response.status_code, e
        )
        return ""
    except (KeyError, json.JSONDecodeError) as e:
        logger.error("Failed to parse LLM response: %s", e)
        return ""
    except Exception as e:
        logger.error("Unexpected error calling LLM: %s", e, exc_info=True)
        return ""


def translate_to_english(text: str) -> str:
    """
    Translate non-English text to English.

    Args:
        text: The text to translate

    Returns:
        str: The translated text, or original if translation fails
    """
    prompt = f"Translate to English: {text}\n\nEnglish:"
    translation = call_llm(prompt, max_tokens=500)
    return translation.strip() if translation else text


def search_vector_index(
    query_text: str,
    pubmed_embeddings: np.ndarray,
    pubmed_data: list,
    embedder: SentenceTransformer,
    top_k: int = 5,
) -> list:
    """
    Search using vector similarity.

    Args:
        query_text: The query text
        pubmed_embeddings: The embeddings matrix
        pubmed_data: The data list
        embedder: The sentence transformer model
        top_k: Number of results to return

    Returns:
        list: List of search results with scores
    """
    query_embedding = embedder.encode(query_text, normalize_embeddings=True)
    scores = np.dot(pubmed_embeddings, query_embedding)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({
            "score": float(scores[idx]),
            "question": pubmed_data[idx]["question"],
            "context": pubmed_data[idx]["context"],
            "answer": pubmed_data[idx]["answer"],
        })
    return results


def search_knowledge_graph(
    driver: GraphDatabase.driver, query_text: str, embedder: SentenceTransformer
) -> dict:
    """
    Search Neo4j knowledge graph.

    Args:
        driver: Neo4j driver
        query_text: The query text
        embedder: The sentence transformer model

    Returns:
        dict: Dictionary with 'entities' and 'relations' lists
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WITH e, rand() as r
            ORDER BY r
            LIMIT 50
            RETURN e.name as name, e.type as type
            """
        )

        entities = []
        for record in result:
            entities.append({
                "name": record["name"],
                "type": record["type"],
            })

        result = session.run(
            """
            MATCH (a:Entity)-[r:RELATES]->(b:Entity)
            RETURN a.name as source, r.type as rel_type, b.name as target
            LIMIT 20
            """
        )

        relations = []
        for record in result:
            relations.append({
                "source": record["source"],
                "type": record["rel_type"],
                "target": record["target"],
            })

    return {"entities": entities, "relations": relations}


def build_context(
    claim_text: str,
    pubmed_embeddings: np.ndarray,
    pubmed_data: list,
    embedder: SentenceTransformer,
    driver: GraphDatabase.driver,
) -> str:
    """
    Build comprehensive context from vector search and knowledge graph.

    Args:
        claim_text: The claim to build context for
        pubmed_embeddings: The embeddings matrix
        pubmed_data: The data list
        embedder: The sentence transformer model
        driver: Neo4j driver

    Returns:
        str: The assembled context string
    """
    vector_results = search_vector_index(
        claim_text, pubmed_embeddings, pubmed_data, embedder
    )
    kg_results = search_knowledge_graph(driver, claim_text, embedder)

    context_parts = [
        "=== MEDICAL LITERATURE (Vector Search) ===\n"
    ]
    for i, r in enumerate(vector_results, 1):
        context_parts.append(f"[{i}] Score: {r['score']:.3f}")
        context_parts.append(f"Q: {r['question'][:100]}...")
        context_parts.append(f"Context: {r['context'][:300]}...")
        context_parts.append(f"Answer: {r['answer']}\n")

    context_parts.append("\n=== KNOWLEDGE GRAPH (Entities & Relations) ===\n")
    entities = kg_results.get("entities", [])
    relations = kg_results.get("relations", [])

    if entities:
        context_parts.append("Key Entities:")
        for e in entities[:15]:
            context_parts.append(f"  - {e['type']}: {e['name'][:40]}")

    if relations:
        context_parts.append("\nRelationships:")
        for rel in relations[:10]:
            context_parts.append(
                f"  {rel['source'][:20]} -> {rel['type']} -> {rel['target'][:20]}"
            )

    return "\n".join(context_parts)


def fact_check_claim(
    claim_text: str,
    original_language: str,
    pubmed_embeddings: np.ndarray,
    pubmed_data: list,
    embedder: SentenceTransformer,
    driver: GraphDatabase.driver,
) -> dict:
    """
    Fact-check a single claim.

    Args:
        claim_text: The claim to fact-check
        original_language: ISO language code
        pubmed_embeddings: The embeddings matrix
        pubmed_data: The data list
        embedder: The sentence transformer model
        driver: Neo4j driver

    Returns:
        dict: Fact-check result with verdict, confidence, and explanation
    """
    english_claim = claim_text
    if original_language != "en":
        english_claim = translate_to_english(claim_text)
        logger.info("    Translated: %s...", english_claim[:60])

    context = build_context(
        english_claim, pubmed_embeddings, pubmed_data, embedder, driver
    )

    prompt = f"""You are a rigorous medical fact-checker. Evaluate the claim using the provided medical literature and knowledge graph context.

Claim to fact-check: {claim_text}
English translation: {english_claim}

CONTEXT:
{context}

Output ONLY a JSON object:
{{
    "verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE",
    "confidence_score": <float 0.0 to 1.0>,
    "explanation": "<2-3 sentence explanation>",
    "claim_language": "<ISO code>"
}}"""

    try:
        raw_response = call_llm(prompt, max_tokens=500)

        if not raw_response:
            return {
                "verdict": "ERROR",
                "confidence_score": 0.0,
                "explanation": "Empty response",
                "claim_language": original_language,
            }

        parsed = safe_parse_llm_json(raw_response)

        if parsed.get("verdict") == "ERROR":
            return {
                "verdict": "ERROR",
                "confidence_score": 0.0,
                "explanation": parsed.get("error", "Parse error"),
                "claim_language": original_language,
            }

        return parsed

    except Exception as e:
        return {
            "verdict": "ERROR",
            "confidence_score": 0.0,
            "explanation": f"API error: {str(e)}",
            "claim_language": original_language,
        }


def run_fact_check(
    claims_path: str | None = None,
    results_path: str | None = None,
):
    """
    Run the complete fact-checking pipeline.

    Args:
        claims_path: Path to input claims JSON
        results_path: Path to save results
    """
    if claims_path is None:
        claims_path = str(settings.data_dir / "scraped_claims.json")
    if results_path is None:
        results_path = str(settings.data_dir / "fact_check_results.json")

    logger.info("Initializing fact-checking pipeline...")

    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        driver.verify_connectivity()
    except AuthError as e:
        logger.error(
            "Neo4j authentication failed. Check credentials in .env: %s", e
        )
        raise SystemExit(1)
    except ServiceUnavailable as e:
        logger.error(
            "Neo4j instance is not reachable at %s: %s",
            settings.neo4j_uri,
            e,
        )
        raise SystemExit(1)

    logger.info("Connected to Neo4j!")

    logger.info("Loading embedding model: %s...", settings.embedding_model)
    try:
        embedder = SentenceTransformer(settings.embedding_model)
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        driver.close()
        raise

    logger.info("Loading PubMed data and embeddings...")
    try:
        with open(settings.data_dir / "pubmedqa_clean.json", encoding="utf-8") as f:
            pubmed_data = json.load(f)
    except FileNotFoundError:
        logger.error("PubMedQA data not found. Run scripts/01_fetch_pubmedqa.py first")
        driver.close()
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in PubMedQA data: %s", e)
        driver.close()
        raise

    try:
        pubmed_embeddings = np.load(settings.embeddings_dir / "pubmed_embeddings.npy")
    except FileNotFoundError:
        logger.error(
            "Embeddings not found. Run scripts/03_embed_and_index.py first"
        )
        driver.close()
        raise

    logger.info("Loaded embeddings: %s", pubmed_embeddings.shape)

    try:
        with open(claims_path, encoding="utf-8") as f:
            claims = json.load(f)
    except FileNotFoundError:
        logger.error("Claims file not found: %s", claims_path)
        driver.close()
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in claims file: %s", e)
        driver.close()
        raise

    logger.info("Processing %d claims...", len(claims))

    results = []
    verdict_counts = {
        "SUPPORTED": 0,
        "CONTRADICTED": 0,
        "UNVERIFIABLE": 0,
        "ERROR": 0,
    }

    for i, item in enumerate(claims):
        claim_text = item["claim_text"]
        original_language = item.get("language", "en")
        logger.info("[%d/%d] Checking: %s...", i + 1, len(claims), claim_text[:50])

        result = fact_check_claim(
            claim_text,
            original_language,
            pubmed_embeddings,
            pubmed_data,
            embedder,
            driver,
        )

        results.append({
            "claim": claim_text,
            "source": item.get("source_url", ""),
            "original_language": original_language,
            "verdict": result.get("verdict", "ERROR"),
            "confidence_score": result.get("confidence_score", 0.0),
            "explanation": result.get("explanation", ""),
            "detected_language": result.get("claim_language", original_language),
        })

        verdict = result.get("verdict", "ERROR")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        logger.info(
            "  -> %s (confidence: %.2f)",
            verdict,
            result.get("confidence_score", 0),
        )

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Total claims processed: %d", len(results))
    for verdict, count in sorted(verdict_counts.items()):
        percentage = (count / len(results)) * 100
        logger.info("  %s: %d (%.1f%%)", verdict, count, percentage)

    logger.info("Results saved to: %s", results_path)

    driver.close()
    logger.info("Fact-checking complete!")


if __name__ == "__main__":
    run_fact_check()
