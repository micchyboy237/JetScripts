# Introduction to Simple RAG

Retrieval-Augmented Generation (RAG) is a hybrid approach that combines information retrieval with generative models. It enhances the performance of language models by incorporating external knowledge, which improves accuracy and factual correctness.

In a Simple RAG setup, we follow these steps:

1. **Data Ingestion**: Load and preprocess the text data.
2. **Chunking**: Break the data into smaller chunks to improve retrieval performance.
3. **Embedding Creation**: Convert the text chunks into numerical representations using an embedding model.
4. **Semantic Search**: Retrieve relevant chunks based on a user query.
5. **Response Generation**: Use a language model to generate a response based on retrieved text.

This notebook implements a Simple RAG approach, evaluates the model’s response, and explores various improvements.

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

## Performing Semantic Search
We implement cosine similarity to find the most relevant text chunks for a user query.

## Running a Query on Extracted Chunks

## Generating a Response Based on Retrieved Chunks

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.
