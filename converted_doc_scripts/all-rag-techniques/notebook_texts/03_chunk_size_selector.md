## Evaluating Chunk Sizes in Simple RAG

Choosing the right chunk size is crucial for improving retrieval accuracy in a Retrieval-Augmented Generation (RAG) pipeline. The goal is to balance retrieval performance with response quality.

This section evaluates different chunk sizes by:

1. Extracting text from a PDF.
2. Splitting text into chunks of varying sizes.
3. Creating embeddings for each chunk.
4. Retrieving relevant chunks for a query.
5. Generating a response using retrieved chunks.
6. Evaluating faithfulness and relevancy.
7. Comparing results for different chunk sizes.

## Setting Up the Environment
We begin by importing necessary libraries.

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

## Extracting Text from the PDF
First, we will extract text from the `AI_Information.pdf` file.

## Chunking the Extracted Text
To improve retrieval, we split the extracted text into overlapping chunks of different sizes.

## Creating Embeddings for Text Chunks
Embeddings convert text into numerical representations for similarity search.

## Performing Semantic Search
We use cosine similarity to find the most relevant text chunks for a user query.

## Generating a Response Based on Retrieved Chunks
Let's  generate a response based on the retrieved text for chunk size `256`.

## Evaluating the AI Response
We score responses based on faithfulness and relevancy using powerfull llm
