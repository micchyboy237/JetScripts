## Context-Enriched Retrieval in RAG
Retrieval-Augmented Generation (RAG) enhances AI responses by retrieving relevant knowledge from external sources. Traditional retrieval methods return isolated text chunks, which can lead to incomplete answers.

To address this, we introduce Context-Enriched Retrieval, which ensures that retrieved information includes neighboring chunks for better coherence.

Steps in This Notebook:
- Data Ingestion: Extract text from a PDF.
- Chunking with Overlapping Context: Split text into overlapping chunks to preserve context.
- Embedding Creation: Convert text chunks into numerical representations.
- Context-Aware Retrieval: Retrieve relevant chunks along with their neighbors for better completeness.
- Response Generation: Use a language model to generate responses based on retrieved context.
- Evaluation: Assess the model's response accuracy.

## Setting Up the Environment
We begin by importing necessary libraries.

## Extracting Text from a PDF File
To implement RAG, we first need a source of textual data. In this case, we extract text from a PDF file using the PyMuPDF library.

## Chunking the Extracted Text
Once we have the extracted text, we divide it into smaller, overlapping chunks to improve retrieval accuracy.

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

## Extracting and Chunking Text from a PDF File
Now, we load the PDF, extract text, and split it into chunks.

## Creating Embeddings for Text Chunks
Embeddings transform text into numerical vectors, which allow for efficient similarity search.

## Implementing Context-Aware Semantic Search
We modify retrieval to include neighboring chunks for better context.

## Running a Query with Context Retrieval
We now test the context-enriched retrieval.

## Generating a Response Using Retrieved Context
We now generate a response using LLM.

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.
