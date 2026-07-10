# RAG Filter Strategies - Demo Files

Comprehensive demonstration of inclusion/exclusion filter strategies for RAG (Retrieval-Augmented Generation) systems.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Demo Categories](#demo-categories)
- [Quick Start](#quick-start)
- [Filter Strategy Selection Guide](#filter-strategy-selection-guide)

## Prerequisites

- Python 3.9+
- OpenAI API key (for embeddings and LLM examples)
- Optional: Pinecone, Qdrant, Weaviate, or Elasticsearch instances

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
rag_filter_demos/
├── 01_domain_filtering/          # Category-based filtering
├── 02_temporal_filtering/        # Date and time-based filtering
├── 03_source_credibility/        # Source quality and document type filtering
├── 04_security_governance/       # Access control and compliance
├── 05_intelligent_filtering/     # LLM-powered filter extraction
├── 06_production_patterns/       # Production-ready implementations
└── 07_advanced_patterns/         # Complex multi-criteria filtering
```

## Demo Categories

### 01 - Domain Filtering

Filter documents by categories like department, topic, or project.

- **Basic**: Simple metadata filtering patterns (new - foundational concepts)
- **ChromaDB**: Full-featured vector store filtering
- **Pinecone**: Cloud-native vector database filtering

### 02 - Temporal Filtering

Filter by date ranges, publication years, or document freshness.

- **Basic**: Core date filtering concepts (new - foundational patterns)
- **Qdrant**: High-performance vector search with range filters
- **Weaviate**: Hybrid search with time-based constraints

### 03 - Source Credibility

Filter by source reliability, document format, and content status.

- **Credibility Scoring**: Multi-criteria quality filtering
- **Document Types**: Format and status-based filtering
- **Elasticsearch**: Traditional search engine with vector capabilities

### 04 - Security & Governance

Implement user-specific and compliance-based filtering.

- **Access Control**: Row-level security with user permissions
- **Role-Based**: Dynamic filtering based on user roles
- **Compliance**: GDPR, HIPAA, SOX governance rules

### 05 - Intelligent Filtering

LLM-powered automatic filter extraction from natural language.

- **Self-Query Retriever**: LangChain's built-in solution
- **Custom Extractor**: Pydantic-based custom implementation
- **Basic Pydantic**: Foundational structured extraction patterns (new)

### 06 - Production Patterns

Production-ready implementations with error handling.

- **Production Searcher**: Complete layered filtering system
- **Fallback Strategies**: Robust error handling and degradation
- **Caching**: Performance optimization with TTL caching

### 07 - Advanced Patterns

Complex filtering scenarios for sophisticated requirements.

- **Hybrid Multi-Criteria**: Complex AND/OR filtering logic
- **Weaviate Advanced**: GraphQL-like filter composition

## Quick Start

```bash
# Basic domain filtering (no external services needed)
python 01_domain_filtering/04_generic_metadata_filtering.py

# ChromaDB domain filtering
python 01_domain_filtering/01_basic_chromadb.py

# Intelligent filtering with LLM
python 05_intelligent_filtering/03_basic_pydantic_extraction.py
```

## Filter Strategy Selection Guide

| Use Case                  | Demo File                                              | Priority           |
| ------------------------- | ------------------------------------------------------ | ------------------ |
| Simple category filtering | `01_domain_filtering/04_generic_metadata_filtering.py` | Simplicity         |
| Date-based queries        | `02_temporal_filtering/03_basic_date_filtering.py`     | Accuracy           |
| User-facing search        | `05_intelligent_filtering/`                            | UX, Flexibility    |
| Enterprise security       | `04_security_governance/`                              | Security           |
| High performance          | `06_production_patterns/`                              | Speed, Reliability |
| Complex domain logic      | `07_advanced_patterns/`                                | Precision          |
