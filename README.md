# Medical GraphRAG Fact-Checker

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-green.svg)](https://neo4j.com/cloud/aura/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Gemini-orange.svg)](https://openrouter.ai/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

**An AI-powered multilingual medical fact-checking system that uses Retrieval-Augmented Generation (RAG) and Knowledge Graphs to verify health claims against peer-reviewed medical literature.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Step 1: Fetch Medical Data](#step-1-fetch-medical-data)
  - [Step 2: Scrape Claims](#step-2-scrape-claims)
  - [Step 3: Generate Embeddings](#step-3-generate-embeddings)
  - [Step 4: Build Knowledge Graph](#step-4-build-knowledge-graph)
  - [Step 5: Run Fact-Checking](#step-5-run-fact-checking)
- [API Configuration](#api-configuration)
- [Sample Results](#sample-results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Medical misinformation, especially in regional languages, poses a significant threat to public health. This project implements a sophisticated **GraphRAG-based Medical Fact-Checker** that:

1. **Fetches** peer-reviewed medical knowledge from PubMedQA
2. **Embeds** medical texts into a multilingual vector space
3. **Constructs** a Knowledge Graph in Neo4j with medical entities and relationships
4. **Fact-checks** health claims using hybrid retrieval (vector similarity + graph traversal)
5. **Supports** multiple languages (Hindi, German, Tamil, Telugu, English)

---

## Features

| Feature | Description |
|---------|-------------|
| **Multilingual Support** | Processes claims in 5+ languages with automatic translation |
| **Hybrid Retrieval** | Combines vector similarity search with knowledge graph traversal |
| **Entity Extraction** | Automatically extracts medical entities (Diseases, Drugs, Symptoms, Treatments) |
| **Relationship Mapping** | Builds semantic relationships between medical concepts |
| **Confidence Scoring** | Provides probability-based verdict with confidence scores |
| **Neo4j Integration** | Stores structured medical knowledge as an interactive graph |
| **RAG Pipeline** | Uses state-of-the-art Retrieval-Augmented Generation for fact-checking |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEDICAL GRAPHRAG FACT-CHECKER                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│   Data Sources   │     │   Embedding &    │     │   Knowledge Graph        │
│                  │     │   Vector Store   │     │   (Neo4j AuraDB)        │
├──────────────────┤     ├──────────────────┤     ├──────────────────────────┤
│ • PubMedQA       │────▶│ • all-MiniLM-L6  │     │ • Entity Nodes           │
│ • Medical Claims │     │ • Vector Index   │     │   - Disease (132)        │
│ • Research Paper │     │ • Cosine Sim    │     │   - Drug (22)           │
└──────────────────┘     └──────────────────┘     │   - Symptom (77)        │
                                                     │   - Treatment (84)      │
                                                     ├──────────────────────────┤
                                                     │ • Relationships         │
                                                     │   - TREATS              │
┌──────────────────┐     ┌──────────────────┐     │   - CAUSES              │
│  LLM Processing  │     │  Fact-Check      │     │   - PREVENTS            │
│                  │     │  Pipeline        │     └──────────────────────────┘
├──────────────────┤     ├──────────────────┤              ▲
│ • Gemini Flash   │────▶│ • Translation    │              │
│ • OpenRouter API │     │ • Context Build │              │
│ • JSON Parsing   │     │ • Verdict Gen   │              │
└──────────────────┘     └──────────────────┘              │
                             │                               │
                             ▼                               │
                    ┌────────────────┐                      │
                    │  Final Output   │──────────────────────┘
                    ├────────────────┤
                    │ • VERDICT       │
                    │ • CONFIDENCE    │
                    │ • EXPLANATION   │
                    └────────────────┘
```

### Data Flow

1. **Ingestion Phase**: PubMedQA data → Embedding Model → Vector Store + Neo4j Graph
2. **Query Phase**: User Claim → Translation → Hybrid Retrieval → LLM Analysis → Verdict

---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| **Language** | Python 3.10+ |
| **Data Ingestion** | HuggingFace Datasets |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Vector Store** | NumPy (Local) |
| **Graph Database** | Neo4j AuraDB (Free Tier) |
| **LLM** | Google Gemini 2.0 Flash (via OpenRouter) |
| **Orchestration** | Custom Python Pipeline |

---

## Project Structure

```
Medical-GraphRAG-FactChecker/
├── .github/
│   └── workflows/
│       └── python-ci.yml          # CI/CD workflow
├── docs/
│   ├── ARCHITECTURE.md            # Detailed architecture
│   ├── SETUP_GUIDE.md             # Setup instructions
│   └── API_REFERENCE.md           # API documentation
├── scripts/
│   ├── 01_fetch_pubmedqa.py       # Fetch PubMedQA dataset
│   ├── 02_scrape_claims.py        # Scrape regional claims
│   ├── 03_embed_and_index.py      # Generate embeddings
│   ├── 04_build_knowledge_graph.py # Build Neo4j graph
│   └── 05_fact_check_pipeline.py  # Run fact-checking
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── utils.py                    # Utility functions
│   └── constants.py                # Constants and enums
├── tests/
│   ├── test_embedding.py
│   ├── test_kg_builder.py
│   └── test_fact_check.py
├── data/
│   ├── pubmedqa_clean.json        # Clean PubMed data
│   ├── scraped_claims.json        # Scraped claims
│   ├── embeddings/                 # Generated embeddings
│   └── fact_check_results.json    # Results output
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── LICENSE                         # Proprietary - All Rights Reserved
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── pyproject.toml                 # Project metadata
└── CONTRIBUTING.md               # Contribution guidelines
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Neo4j AuraDB account (free tier)
- OpenRouter API key (for LLM access)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sourav1243/Medical-GraphRAG-FactChecker.git
cd Medical-GraphRAG-FactChecker
```

2. **Create virtual environment:**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Using conda
conda create -n medigraph python=3.10
conda activate medigraph
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Configuration

1. **Copy the environment template:**
```bash
cp .env.example .env
```

2. **Configure environment variables in `.env`:**

```env
# Neo4j AuraDB Credentials
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password

# OpenRouter API Key (for LLM)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

3. **Get Neo4j credentials:**
   - Create free account at [Neo4j AuraDB](https://neo4j.com/cloud/aura/)
   - Create new AuraDB instance
   - Copy connection details to `.env`

4. **Get OpenRouter API key:**
   - Sign up at [OpenRouter.ai](https://openrouter.ai/)
   - Generate API key
   - Add to `.env`

---

## Usage

### Step 1: Fetch Medical Data

Download and clean PubMedQA dataset:
```bash
python scripts/01_fetch_pubmedqa.py
```

**Output:** `data/pubmedqa_clean.json` with ~1000 labeled medical Q&A pairs.

### Step 2: Scrape Claims

Collect regional medical claims (configure URLs in the script):
```bash
python scripts/02_scrape_claims.py
```

**Output:** `data/scraped_claims.json` with claims in multiple languages.

### Step 3: Generate Embeddings

Create vector embeddings for medical literature:
```bash
python scripts/03_embed_and_index.py
```

**Output:** `embeddings/pubmed_embeddings.npy` (1000 × 384 matrix).

### Step 4: Build Knowledge Graph

Construct medical knowledge graph in Neo4j:
```bash
python scripts/04_build_knowledge_graph.py
```

**This step:**
- Extracts medical entities using LLM
- Creates entity nodes (Disease, Drug, Symptom, Treatment)
- Builds relationship edges (TREATS, CAUSES, PREVENTS)
- Stores chunks with embeddings for hybrid retrieval

### Step 5: Run Fact-Checking

Execute the fact-checking pipeline:
```bash
python scripts/05_fact_check_pipeline.py
```

**Output:** `data/fact_check_results.json` with verdicts and explanations.

---

## API Configuration

### Supported LLM Models

The project supports multiple LLM providers through OpenRouter:

| Model | Provider | Status |
|-------|----------|--------|
| Gemini 2.0 Flash | Google | ✅ Recommended |
| GPT-4o Mini | OpenAI | ✅ Supported |
| Claude 3 Haiku | Anthropic | ✅ Supported |

### Changing the Model

Edit `scripts/05_fact_check_pipeline.py` to change the model:

```python
payload = {
    'model': 'anthropic/claude-3-haiku',  # Change here
    'messages': [{'role': 'user', 'content': prompt}],
    ...
}
```

---

## Sample Results

### Input Claims (Multilingual)

| Language | Claim |
|----------|-------|
| Hindi | गर्म पानी पीने से कोरोना वायरस ठीक हो जाता है। |
| German | Trinken von warmem Wasser heilt COVID-19. |
| Tamil | மஞ்சள் பால் குடிப்பது புற்றுநோயை குணப்படுத்தும். |
| English | Antibiotics are effective against viral infections. |

### Output Verdicts

```json
{
  "claim": "हल्दी और दूध पीने से कैंसर का इलाज होता है।",
  "original_language": "hi",
  "verdict": "CONTRADICTED",
  "confidence_score": 0.90,
  "explanation": "There is no scientific evidence that drinking turmeric and milk cures cancer. While turmeric has anti-inflammatory properties, it is not a proven cancer treatment.",
  "detected_language": "hi"
}
```

### Results Summary

```
Total claims processed: 20
  CONTRADICTED:   6 (30.0%)
  UNVERIFIABLE:  14 (70.0%)
  SUPPORTED:      0 (0.0%)
```

---

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/
```

---

## License

This project is proprietary and confidential. All rights reserved. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [PubMedQA Dataset](https://github.com/pubmedqa) - Medical Q&A dataset
- [Neo4j](https://neo4j.com/) - Graph database
- [HuggingFace](https://huggingface.co/) - Embedding models
- [OpenRouter](https://openrouter.ai/) - LLM API aggregation
- [Sentence-Transformers](https://sbert.net/) - Embedding library

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ by [Sourav Kumar Sharma](https://github.com/sourav1243)*

</div>
