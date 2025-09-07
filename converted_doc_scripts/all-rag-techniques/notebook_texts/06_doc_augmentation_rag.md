# Document Augmentation RAG with Question Generation

This notebook implements an enhanced RAG approach using document augmentation through question generation. By generating relevant questions for each text chunk, we improve the retrieval process, leading to better responses from the language model.

In this implementation, we follow these steps:

1. **Data Ingestion**: Extract text from a PDF file.
2. **Chunking**: Split the text into manageable chunks.
3. **Question Generation**: Generate relevant questions for each chunk.
4. **Embedding Creation**: Create embeddings for both chunks and generated questions.
5. **Vector Store Creation**: Build a simple vector store using NumPy.
6. **Semantic Search**: Retrieve relevant chunks and questions for user queries.
7. **Response Generation**: Generate answers based on retrieved content.
8. **Evaluation**: Assess the quality of the generated responses.

## Setting Up the Environment
We begin by importing necessary libraries.

## Extracting Text from a PDF File
To implement RAG, we first need a source of textual data. In this case, we extract text from a PDF file using the PyMuPDF library.

## Chunking the Extracted Text
Once we have the extracted text, we divide it into smaller, overlapping chunks to improve retrieval accuracy.

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

## Generating Questions for Text Chunks
This is the key enhancement over simple RAG. We generate questions that could be answered by each text chunk.

## Creating Embeddings for Text
We generate embeddings for both text chunks and generated questions.

## Building a Simple Vector Store
We'll implement a simple vector store using NumPy.

## Processing Documents with Question Augmentation
Now we'll put everything together to process documents, generate questions, and build our augmented vector store.

## Extracting and Processing the Document

## Performing Semantic Search
We implement a semantic search function similar to the simple RAG implementation but adapted to our augmented vector store.

## Running a Query on the Augmented Vector Store

## Generating Context for Response
Now we prepare the context by combining information from relevant chunks and questions.

## Generating a Response Based on Retrieved Chunks

## Generating and Displaying the Response

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.

## Running the Evaluation

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
