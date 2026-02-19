"""
Fact-Check Pipeline

This module implements the complete fact-checking pipeline that combines
vector similarity search with knowledge graph traversal to verify medical claims.
"""

import json
import os
import re
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import requests
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def call_llm(prompt: str, max_tokens: int = 1000) -> str:
    """Call OpenRouter API for LLM processing."""
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


def translate_to_english(text: str) -> str:
    """Translate non-English text to English."""
    prompt = f"Translate to English: {text}\n\nEnglish:"
    translation = call_llm(prompt, max_tokens=500)
    return translation.strip() if translation else text


def search_vector_index(query_text: str, pubmed_embeddings: np.ndarray, pubmed_data: list, embedder, top_k: int = 5) -> list:
    """Search using vector similarity."""
    query_embedding = embedder.encode(query_text, normalize_embeddings=True)
    scores = np.dot(pubmed_embeddings, query_embedding)
    top_idx = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_idx:
        results.append({
            "score": float(scores[idx]),
            "question": pubmed_data[idx]["question"],
            "context": pubmed_data[idx]["context"],
            "answer": pubmed_data[idx]["answer"]
        })
    return results


def search_knowledge_graph(driver, query_text: str, embedder) -> dict:
    """Search Neo4j knowledge graph."""
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WITH e, rand() as r
            ORDER BY r
            LIMIT 50
            RETURN e.name as name, e.type as type
        """)
        
        entities = []
        for record in result:
            entities.append({
                "name": record["name"],
                "type": record["type"]
            })
        
        result = session.run("""
            MATCH (a:Entity)-[r:RELATES]->(b:Entity)
            RETURN a.name as source, r.type as rel_type, b.name as target
            LIMIT 20
        """)
        
        relations = []
        for record in result:
            relations.append({
                "source": record["source"],
                "type": record["rel_type"],
                "target": record["target"]
            })
    
    return {"entities": entities, "relations": relations}


def build_context(claim_text: str, pubmed_embeddings: np.ndarray, pubmed_data: list, embedder, driver) -> str:
    """Build comprehensive context from vector search and knowledge graph."""
    vector_results = search_vector_index(claim_text, pubmed_embeddings, pubmed_data, embedder)
    kg_results = search_knowledge_graph(driver, claim_text, embedder)
    
    context_parts = ["=== MEDICAL LITERATURE (Vector Search) ===\n"]
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
            context_parts.append(f"  {rel['source'][:20]} -> {rel['type']} -> {rel['target'][:20]}")
    
    return "\n".join(context_parts)


def fact_check_claim(claim_text: str, original_language: str, pubmed_embeddings: np.ndarray, pubmed_data: list, embedder, driver) -> dict:
    """Fact-check a single claim."""
    english_claim = claim_text
    if original_language != "en":
        english_claim = translate_to_english(claim_text)
        print(f"    Translated: {english_claim[:60]}...")
    
    context = build_context(english_claim, pubmed_embeddings, pubmed_data, embedder, driver)
    
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
            return {"verdict": "ERROR", "confidence_score": 0.0, "explanation": "Empty response", "claim_language": original_language}
        
        json_match = re.search(r"\{[^{}]*\}", raw_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return {"verdict": "ERROR", "confidence_score": 0.0, "explanation": f"Parse error", "claim_language": original_language}
            
    except Exception as e:
        return {"verdict": "ERROR", "confidence_score": 0.0, "explanation": f"API error: {str(e)}", "claim_language": original_language}


def run_fact_check(
    claims_path: str = "data/scraped_claims.json",
    results_path: str = "data/fact_check_results.json"
):
    """
    Run the complete fact-checking pipeline.
    
    Args:
        claims_path: Path to input claims JSON
        results_path: Path to save results
    """
    print("Initializing fact-checking pipeline...")
    
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    
    driver.verify_connectivity()
    print("Connected to Neo4j!")
    
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Loading PubMed data and embeddings...")
    with open("data/pubmedqa_clean.json", encoding="utf-8") as f:
        pubmed_data = json.load(f)
    
    pubmed_embeddings = np.load("embeddings/pubmed_embeddings.npy")
    print(f"Loaded embeddings: {pubmed_embeddings.shape}")
    
    with open(claims_path, encoding="utf-8") as f:
        claims = json.load(f)
    
    print(f"\nProcessing {len(claims)} claims...")
    
    results = []
    verdict_counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIABLE": 0, "ERROR": 0}
    
    for i, item in enumerate(claims):
        claim_text = item["claim_text"]
        original_language = item.get("language", "en")
        print(f"[{i+1}/{len(claims)}] Checking: {claim_text[:50]}...")
        
        result = fact_check_claim(claim_text, original_language, pubmed_embeddings, pubmed_data, embedder, driver)
        
        results.append({
            "claim": claim_text,
            "source": item.get("source_url", ""),
            "original_language": original_language,
            "verdict": result.get("verdict", "ERROR"),
            "confidence_score": result.get("confidence_score", 0.0),
            "explanation": result.get("explanation", ""),
            "detected_language": result.get("claim_language", original_language)
        })
        
        verdict = result.get("verdict", "ERROR")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        print(f"  -> {verdict} (confidence: {result.get('confidence_score', 0):.2f})")
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total claims processed: {len(results)}")
    for verdict, count in sorted(verdict_counts.items()):
        percentage = (count / len(results)) * 100
        print(f"  {verdict}: {count} ({percentage:.1f}%)")
    
    print(f"\nResults saved to: {results_path}")
    
    driver.close()
    print("\nFact-checking complete!")


if __name__ == "__main__":
    run_fact_check()
