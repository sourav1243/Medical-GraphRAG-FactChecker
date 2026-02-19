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
Graph Module

Provides Neo4j knowledge graph query functionality for the fact-checking system.
"""

from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

driver: GraphDatabase.driver | None = None


def get_driver() -> GraphDatabase.driver:
    """
    Get or create a Neo4j driver instance.

    Returns:
        Neo4j driver instance

    Raises:
        SystemExit: If connection fails
    """
    global driver

    if driver is None:
        try:
            logger.info("Connecting to Neo4j at: %s", settings.neo4j_uri)
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
            driver.verify_connectivity()
            logger.info("Neo4j connection established successfully")
        except AuthError as e:
            logger.error(
                "Neo4j authentication failed. Check NEO4J_USERNAME and "
                "NEO4J_PASSWORD in .env: %s",
                e,
            )
            raise SystemExit(1)
        except ServiceUnavailable as e:
            logger.error(
                "Neo4j instance is not reachable at %s: %s",
                settings.neo4j_uri,
                e,
            )
            raise SystemExit(1)

    return driver


def close_driver() -> None:
    """Close the Neo4j driver if it exists."""
    global driver
    if driver is not None:
        driver.close()
        driver = None
        logger.info("Neo4j connection closed")


def query_graph(query_text: str) -> list[dict[str, Any]]:
    """
    Query the knowledge graph for entities and relationships.

    Args:
        query_text: The text to search for in the graph

    Returns:
        List of dictionaries containing graph query results
    """
    neo4j_driver = get_driver()

    with neo4j_driver.session() as session:
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


def get_graph_stats() -> dict[str, int]:
    """
    Get statistics about the knowledge graph.

    Returns:
        Dictionary with counts of chunks, entities, and relations
    """
    neo4j_driver = get_driver()

    with neo4j_driver.session() as session:
        result = session.run("MATCH (n:Chunk) RETURN count(n) as c")
        chunk_count = result.single()["c"]

        result = session.run("MATCH (n:Entity) RETURN count(n) as c")
        entity_count = result.single()["c"]

        result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as c")
        rel_count = result.single()["c"]

    return {
        "chunks": chunk_count,
        "entities": entity_count,
        "relations": rel_count,
    }


def create_vector_index(dimensions: int = 1024) -> None:
    """
    Create a vector index in Neo4j for hybrid retrieval.

    Args:
        dimensions: Dimension of the embedding vectors
    """
    neo4j_driver = get_driver()

    with neo4j_driver.session() as session:
        session.run(
            """
            CREATE VECTOR INDEX IF NOT EXISTS chunk_embeddings
            FOR (c:Chunk) ON c.embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            dimensions=dimensions,
        )
        logger.info("Vector index created with %d dimensions", dimensions)
