## Introduction to Semantic Chunking
Text chunking is an essential step in Retrieval-Augmented Generation (RAG), where large text bodies are divided into meaningful segments to improve retrieval accuracy.
Unlike fixed-length chunking, semantic chunking splits text based on the content similarity between sentences.

### Breakpoint Methods:
- **Percentile**: Finds the Xth percentile of all similarity differences and splits chunks where the drop is greater than this value.
- **Standard Deviation**: Splits where similarity drops more than X standard deviations below the mean.
- **Interquartile Range (IQR)**: Uses the interquartile distance (Q3 - Q1) to determine split points.

This notebook implements semantic chunking **using the percentile method** and evaluates its performance on a sample text.

## Setting Up the Environment
We begin by importing necessary libraries.

## Extracting Text from a PDF File
To implement RAG, we first need a source of textual data. In this case, we extract text from a PDF file using the PyMuPDF library.

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

## Creating Sentence-Level Embeddings
We split text into sentences and generate embeddings.

## Calculating Similarity Differences
We compute cosine similarity between consecutive sentences.

## Implementing Semantic Chunking
We implement three different methods for finding breakpoints.

## Splitting Text into Semantic Chunks
We split the text based on computed breakpoints.

## Creating Embeddings for Semantic Chunks
We create embeddings for each chunk for later retrieval.

## Performing Semantic Search
We implement cosine similarity to retrieve the most relevant chunks.

## Generating a Response Based on Retrieved Chunks

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.
