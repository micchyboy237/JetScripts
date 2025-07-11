from dotenv import load_dotenv
from evaluation.evalute_rag import evaluate_rag
from google.colab import userdata
from helper_functions import (EmbeddingProvider,
retrieve_context_per_question,
replace_t_with_space,
get_langchain_embedding_provider,
show_context)
from jet.logger import CustomLogger
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb)

# Simple RAG (Retrieval-Augmented Generation) System

## Overview

This code implements a basic Retrieval-Augmented Generation (RAG) system for processing and querying PDF documents. The system encodes the document content into a vector store, which can then be queried to retrieve relevant information.

## Key Components

1. PDF processing and text extraction
2. Text chunking for manageable processing
3. Vector store creation using [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) and Ollama embeddings
4. Retriever setup for querying the processed documents
5. Evaluation of the RAG system

## Method Details

### Document Preprocessing

1. The PDF is loaded using PyPDFLoader.
2. The text is split into chunks using RecursiveCharacterTextSplitter with specified chunk size and overlap.

### Text Cleaning

A custom function `replace_t_with_space` is applied to clean the text chunks. This likely addresses specific formatting issues in the PDF.

### Vector Store Creation

1. Ollama embeddings are used to create vector representations of the text chunks.
2. A FAISS vector store is created from these embeddings for efficient similarity search.

### Retriever Setup

1. A retriever is configured to fetch the top 2 most relevant chunks for a given query.

### Encoding Function

The `encode_pdf` function encapsulates the entire process of loading, chunking, cleaning, and encoding the PDF into a vector store.

## Key Features

1. Modular Design: The encoding process is encapsulated in a single function for easy reuse.
2. Configurable Chunking: Allows adjustment of chunk size and overlap.
3. Efficient Retrieval: Uses FAISS for fast similarity search.
4. Evaluation: Includes a function to evaluate the RAG system's performance.

## Usage Example

The code includes a test query: "What is the main cause of climate change?". This demonstrates how to use the retriever to fetch relevant context from the processed document.

## Evaluation

The system includes an `evaluate_rag` function to assess the performance of the retriever, though the specific metrics used are not detailed in the provided code.

## Benefits of this Approach

1. Scalability: Can handle large documents by processing them in chunks.
2. Flexibility: Easy to adjust parameters like chunk size and number of retrieved results.
3. Efficiency: Utilizes FAISS for fast similarity search in high-dimensional spaces.
4. Integration with Advanced NLP: Uses Ollama embeddings for state-of-the-art text representation.

## Conclusion

This simple RAG system provides a solid foundation for building more complex information retrieval and question-answering systems. By encoding document content into a searchable vector store, it enables efficient retrieval of relevant information in response to queries. This approach is particularly useful for applications requiring quick access to specific information within large documents or document collections.

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Simple RAG (Retrieval-Augmented Generation) System")

# !pip install pypdf==5.6.0
# !pip install PyMuPDF==1.26.1
# !pip install python-dotenv==1.1.0
# !pip install langchain-community==0.3.25
# !pip install jet.llm.ollama.base_langchain==0.3.23
# !pip install rank_bm25==0.2.2
# !pip install faiss-cpu==1.11.0
# !pip install deepeval==3.1.0

# !git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')




load_dotenv()

# if not userdata.get('OPENAI_API_KEY'):
#     os.environ["OPENAI_API_KEY"] = input("Please enter your Ollama API key: ")
else:
#     os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')





"""
### Read Docs
"""
logger.info("### Read Docs")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"

"""
### Encode document
"""
logger.info("### Encode document")

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using Ollama embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)

    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)

"""
### Create retriever
"""
logger.info("### Create retriever")

chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})

"""
### Test retriever
"""
logger.info("### Test retriever")

test_query = "What is the main cause of climate change?"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(context)

"""
### Evaluate results
"""
logger.info("### Evaluate results")

evaluate_rag(chunks_query_retriever)

logger.info("\n\n[DONE]", bright=True)