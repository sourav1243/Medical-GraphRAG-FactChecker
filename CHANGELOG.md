# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0] — 2025-02-19

### Added
- Apache 2.0 license replacing proprietary license
- Copyright headers on all source files
- `BAAI/bge-m3` multilingual embeddings (upgraded from `all-MiniLM-L6-v2`)
- `pydantic-settings` configuration management
- Structured logging via `src/logger.py`
- Evaluation pipeline `evaluate_rag.py` with golden dataset
- `benchmark_results.md` comparing GraphRAG vs. Naive RAG
- Docker + docker-compose support
- Full pytest test suite with 70%+ coverage
- GitHub Actions CI workflow

### Changed
- All `print()` calls replaced with structured logging
- All external calls wrapped in typed `try/except` blocks
- Vector index dimension updated from 384 to 1024

### Fixed
- LLM JSON parsing now handles preamble text gracefully
- Added proper error handling for Neo4j connection failures
- Added language detection for Hindi, German, Tamil, Telugu

### Removed
- Proprietary license restrictions

## [1.0.0] — 2025-01-01

### Added
- Initial release with basic GraphRAG pipeline
- PubMedQA data fetching
- Claim scraping from regional sources
- Vector embedding generation
- Knowledge graph building in Neo4j
- Fact-checking pipeline with Gemini 2.0 Flash

### Known Issues
- Limited language support
- Basic error handling
- Proprietary license
