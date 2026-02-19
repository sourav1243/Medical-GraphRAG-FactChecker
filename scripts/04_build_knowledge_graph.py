"""
Knowledge Graph Builder

This module constructs a medical knowledge graph in Neo4j by extracting
entities and relationships from PubMedQA data using LLM-based extraction.
"""

import json
import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import requests
import numpy as np


load_dotenv()

MAX_RECORDS = 50
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def call_llm(prompt: str, max_tokens: int = 1500) -> str:
    """Call OpenRouter API for entity extraction."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/sourav1243",
        "X-Title": "Medical-GraphRAG-FactChecker"
    }
    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        return ""
    except Exception as e:
        print(f"API error: {e}")
        return ""


def extract_entities_and_relations(text: str) -> dict:
    """Extract medical entities and relationships using LLM."""
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
    except:
        pass
    
    return {"entities": [], "relations": []}


def build_knowledge_graph(
    data_path: str = "data/pubmedqa_clean.json",
    max_records: int = 50
):
    """
    Build knowledge graph in Neo4j with entities and relationships.
    
    Args:
        data_path: Path to PubMedQA data
        max_records: Maximum records to process
    """
    print("Connecting to Neo4j AuraDB...")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    
    driver.verify_connectivity()
    print("Connected to Neo4j successfully!")
    
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Loading PubMedQA data from {data_path}...")
    with open(data_path, encoding="utf-8") as f:
        pubmed_data = json.load(f)
    
    records = pubmed_data[:max_records]
    print(f"Processing {len(records)} records...")
    
    print("Clearing existing graph data...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    
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
                session.run("""
                    CREATE (c:Chunk {
                        text: $text,
                        question: $question,
                        answer: $answer,
                        embedding: $embedding
                    })
                """, text=text[:1500], question=question[:300], answer=answer[:200], embedding=embedding)
            
            entity_map = {}
            for entity in entities:
                name = entity.get("name", "").strip()
                entity_type = entity.get("type", "Unknown")
                if name and len(name) > 1:
                    with driver.session() as session:
                        result = session.run("""
                            MERGE (e:Entity {name: $name})
                            SET e.type = $type
                        """, name=name, type=entity_type)
                        if result.single():
                            entity_map[name] = True
            
            for rel in relations:
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                rel_type = rel.get("type", "RELATES").strip()
                
                if source and target and source in entity_map and target in entity_map:
                    with driver.session() as session:
                        session.run("""
                            MATCH (a:Entity {name: $source})
                            MATCH (b:Entity {name: $target})
                            MERGE (a)-[r:RELATES {type: $type}]->(b)
                        """, source=source, target=target, type=rel_type)
            
            success_count += 1
            
            if (i + 1) % 5 == 0:
                print(f"Progress: {i + 1}/{len(records)} records processed")
                
        except Exception as e:
            error_count += 1
            print(f"Error on record {i + 1}: {str(e)[:60]}...")
    
    print(f"\nCompleted! Success: {success_count}, Errors: {error_count}")
    
    print("\nVerifying graph...")
    with driver.session() as session:
        result = session.run("MATCH (n:Chunk) RETURN count(n) as c")
        chunk_count = result.single()["c"]
        
        result = session.run("MATCH (n:Entity) RETURN count(n) as c")
        entity_count = result.single()["c"]
        
        result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as c")
        rel_count = result.single()["c"]
        
        print(f"Chunks: {chunk_count}, Entities: {entity_count}, Relations: {rel_count}")
    
    driver.close()
    print("\nKnowledge graph built successfully!")


if __name__ == "__main__":
    build_knowledge_graph()
