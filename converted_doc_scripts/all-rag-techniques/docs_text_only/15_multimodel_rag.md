# Multi-Modal RAG with Image Captioning

In this notebook, I implement a Multi-Modal RAG system that extracts both text and images from documents, generates captions for images, and uses both content types to respond to queries. This approach enhances traditional RAG by incorporating visual information into the knowledge base.

Traditional RAG systems only work with text, but many documents contain crucial information in images, charts, and tables. By captioning these visual elements and incorporating them into our retrieval system, we can:

- Access information locked in figures and diagrams
- Understand tables and charts that complement the text
- Create a more comprehensive knowledge base
- Answer questions that rely on visual data

## Setting Up the Environment
We begin by importing necessary libraries.

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

## Document Processing Functions

## Chunking Text Content

## Image Captioning with OpenAI Vision

## Simple Vector Store Implementation

## Creating Embeddings

## Complete Processing Pipeline

## Query Processing and Response Generation

## Evaluation Against Text-Only RAG

## Evaluation on Multi-Modal RAG vs Text-Only RAG
