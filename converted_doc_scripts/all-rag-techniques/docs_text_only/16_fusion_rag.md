# Fusion Retrieval: Combining Vector and Keyword Search

In this notebook, I implement a fusion retrieval system that combines the strengths of semantic vector search with keyword-based BM25 retrieval. This approach improves retrieval quality by capturing both conceptual similarity and exact keyword matches.

## Why Fusion Retrieval Matters

Traditional RAG systems typically rely on vector search alone, but this has limitations:

- Vector search excels at semantic similarity but may miss exact keyword matches
- Keyword search is great for specific terms but lacks semantic understanding
- Different queries perform better with different retrieval methods

Fusion retrieval gives us the best of both worlds by:

- Performing both vector-based and keyword-based retrieval
- Normalizing the scores from each approach
- Combining them with a weighted formula
- Ranking documents based on the combined score

## Setting Up the Environment
We begin by importing necessary libraries.

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

## Document Processing Functions

## Creating Our Vector Store

## BM25 Implementation

## Fusion Retrieval Function

## Document Processing Pipeline

## Response Generation

## Main Retrieval Function

## Comparing Retrieval Methods

## Evaluation Functions

## Complete Evaluation Pipeline

## Evaluating Fusion Retrieval
