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
Knowledge Graph Builder

This module constructs a medical knowledge graph in Neo4j by extracting
entities and relationships from PubMedQA data using LLM-based extraction.
"""

import json
import re
from pathlib import Path

import numpy as np
import requests
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def call_llm(prompt: str, max_tokens: int = 1500) -> str:
    """
    Call OpenRouter API for entity extraction.

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


def extract_entities_and_relations(text: str) -> dict:
    """
    Extract medical entities and relationships using LLM.

    Args:
        text: The text to extract entities from

    Returns:
        dict: Dictionary with 'entities' and 'relations' lists
    """
    prompt = f"""Extract medical entities and relationships from this text.

TEXT: {text[:2000]}

Return JSON with:
{{
  "entities": [{{"name": "name", "type": "Type"}}],
  "relations": [{{"source": "entity1", "target": "entity2", "type": "RELATES"}}]
}}

Types: Disease, Drug, Symptom, Treatment, Gene, Protein
Relationships: TREATS, CAUSES, PREVENTS, INTERACTS_WITH

If none, return empty arrays."""

    response = call_llm(prompt, max_tokens=1000)

    try:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            result = json.loads(json_match.group())
            if "entities" in result and "relations" in result:
                return result
    except json.JSONDecodeError:
        logger.warning("Failed to parse entity extraction response")
    except Exception as e:
        logger.error("Error in entity extraction: %s", e)

    return {"entities": [], "relations": []}


def build_knowledge_graph(
    data_path: str | None = None,
    max_records: int | None = None,
):
    """
    Build knowledge graph in Neo4j with entities and relationships.

    Args:
        data_path: Path to PubMedQA data
        max_records: Maximum records to process
    """
    if data_path is None:
        data_path = str(settings.data_dir / "pubmedqa_clean.json")
    if max_records is None:
        max_records = settings.pubmed_sample_size

    logger.info("Connecting to Neo4j AuraDB...")

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

    logger.info("Connected to Neo4j successfully!")

    logger.info("Loading embedding model: %s...", settings.embedding_model)
    try:
        embedder = SentenceTransformer(settings.embedding_model)
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        driver.close()
        raise

    logger.info("Loading PubMedQA data from %s...", data_path)
    try:
        with open(data_path, encoding="utf-8") as f:
            pubmed_data = json.load(f)
    except FileNotFoundError:
        logger.error("Data file not found at %s", data_path)
        driver.close()
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in data file: %s", e)
        driver.close()
        raise

    records = pubmed_data[:max_records]
    logger.info("Processing %d records...", len(records))

    logger.info("Clearing existing graph data...")
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    except Exception as e:
        logger.error("Failed to clear graph data: %s", e)
        driver.close()
        raise

    success_count = 0
    error_count = 0

    for i, item in enumerate(records):
        question = item.get("question", "")
        context = item.get("context", "")
        answer = item.get("answer", item.get("long_answer", ""))
        text = f"Q: {question}\n\nContext: {context}\n\nAnswer: {answer}"

        try:
            extraction = extract_entities_and_relations(text)
            entities = extraction.get("entities", [])
            relations = extraction.get("relations", [])

            embedding = embedder.encode(text[:1000]).tolist()

            with driver.session() as session:
                session.run(
                    """
                    CREATE (c:Chunk {
                        text: $text,
                        question: $question,
                        answer: $answer,
                        embedding: $embedding
                    })
                    """,
                    text=text[:1500],
                    question=question[:300],
                    answer=answer[:200],
                    embedding=embedding,
                )

            entity_map = {}
            for entity in entities:
                name = entity.get("name", "").strip()
                entity_type = entity.get("type", "Unknown")
                if name and len(name) > 1:
                    with driver.session() as session:
                        result = session.run(
                            """
                            MERGE (e:Entity {name: $name})
                            SET e.type = $type
                            """,
                            name=name,
                            type=entity_type,
                        )
                        if result.single():
                            entity_map[name] = True

            for rel in relations:
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                rel_type = rel.get("type", "RELATES").strip()

                if source and target and source in entity_map and target in entity_map:
                    with driver.session() as session:
                        session.run(
                            """
                            MATCH (a:Entity {name: $source})
                            MATCH (b:Entity {name: $target})
                            MERGE (a)-[r:RELATES {type: $type}]->(b)
                            """,
                            source=source,
                            target=target,
                            type=rel_type,
                        )

            success_count += 1

            if (i + 1) % 5 == 0:
                logger.info("Progress: %d/%d records processed", i + 1, len(records))

        except Exception as e:
            error_count += 1
            logger.error("Error on record %d: %s", i + 1, str(e)[:60])

    logger.info("Completed! Success: %d, Errors: %d", success_count, error_count)

    logger.info("Verifying graph...")
    try:
        with driver.session() as session:
            result = session.run("MATCH (n:Chunk) RETURN count(n) as c")
            chunk_count = result.single()["c"]

            result = session.run("MATCH (n:Entity) RETURN count(n) as c")
            entity_count = result.single()["c"]

            result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as c")
            rel_count = result.single()["c"]

        logger.info(
            "Graph stats - Chunks: %d, Entities: %d, Relations: %d",
            chunk_count,
            entity_count,
            rel_count,
        )
    except Exception as e:
        logger.warning("Failed to get graph stats: %s", e)

    driver.close()
    logger.info("Knowledge graph built successfully!")


if __name__ == "__main__":
    build_knowledge_graph()
