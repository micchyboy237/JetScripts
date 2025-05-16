# Proposition Chunking for Enhanced RAG

In this notebook, I implement proposition chunking - an advanced technique to break down documents into atomic, factual statements for more accurate retrieval. Unlike traditional chunking that simply divides text by character count, proposition chunking preserves the semantic integrity of individual facts.

Proposition chunking delivers more precise retrieval by:

1. Breaking content into atomic, self-contained facts
2. Creating smaller, more granular units for retrieval
3. Enabling more precise matching between queries and relevant content
4. Filtering out low-quality or incomplete propositions

Let's build a complete implementation without relying on LangChain or FAISS.

## Setting Up the Environment
We begin by importing necessary libraries.

## Extracting Text from a PDF File
To implement RAG, we first need a source of textual data. In this case, we extract text from a PDF file using the PyMuPDF library.

## Chunking the Extracted Text
Once we have the extracted text, we divide it into smaller, overlapping chunks to improve retrieval accuracy.

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

## Simple Vector Store Implementation
We'll create a basic vector store to manage document chunks and their embeddings.

## Creating Embeddings

## Proposition Generation

## Quality Checking for Propositions

## Complete Proposition Processing Pipeline

## Building Vector Stores for Both Approaches

## Query and Retrieval Functions

## Response Generation and Evaluation

## Complete End-to-End Evaluation Pipeline

## Evaluation of Proposition Chunking
