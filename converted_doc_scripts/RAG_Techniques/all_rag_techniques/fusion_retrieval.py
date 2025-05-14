from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.logger import CustomLogger
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
from typing import List
import numpy as np
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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/fusion_retrieval.ipynb)

# Fusion Retrieval in Document Search

## Overview

This code implements a Fusion Retrieval system that combines vector-based similarity search with keyword-based BM25 retrieval. The approach aims to leverage the strengths of both methods to improve the overall quality and relevance of document retrieval.

## Motivation

Traditional retrieval methods often rely on either semantic understanding (vector-based) or keyword matching (BM25). Each approach has its strengths and weaknesses. Fusion retrieval aims to combine these methods to create a more robust and accurate retrieval system that can handle a wider range of queries effectively.

## Key Components

1. PDF processing and text chunking
2. Vector store creation using FAISS and Ollama embeddings
3. BM25 index creation for keyword-based retrieval
4. Custom fusion retrieval function that combines both methods

## Method Details

### Document Preprocessing

1. The PDF is loaded and split into chunks using RecursiveCharacterTextSplitter.
2. Chunks are cleaned by replacing 't' with spaces (likely addressing a specific formatting issue).

### Vector Store Creation

1. Ollama embeddings are used to create vector representations of the text chunks.
2. A FAISS vector store is created from these embeddings for efficient similarity search.

### BM25 Index Creation

1. A BM25 index is created from the same text chunks used for the vector store.
2. This allows for keyword-based retrieval alongside the vector-based method.

### Fusion Retrieval Function

The `fusion_retrieval` function is the core of this implementation:

1. It takes a query and performs both vector-based and BM25-based retrieval.
2. Scores from both methods are normalized to a common scale.
3. A weighted combination of these scores is computed (controlled by the `alpha` parameter).
4. Documents are ranked based on the combined scores, and the top-k results are returned.

## Benefits of this Approach

1. Improved Retrieval Quality: By combining semantic and keyword-based search, the system can capture both conceptual similarity and exact keyword matches.
2. Flexibility: The `alpha` parameter allows for adjusting the balance between vector and keyword search based on specific use cases or query types.
3. Robustness: The combined approach can handle a wider range of queries effectively, mitigating weaknesses of individual methods.
4. Customizability: The system can be easily adapted to use different vector stores or keyword-based retrieval methods.

## Conclusion

Fusion retrieval represents a powerful approach to document search that combines the strengths of semantic understanding and keyword matching. By leveraging both vector-based and BM25 retrieval methods, it offers a more comprehensive and flexible solution for information retrieval tasks. This approach has potential applications in various fields where both conceptual similarity and keyword relevance are important, such as academic research, legal document search, or general-purpose search engines.

<div style="text-align: center;">

<img src="../images/fusion_retrieval.svg" alt="Fusion Retrieval" style="width:100%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Fusion Retrieval in Document Search")

# !pip install langchain numpy python-dotenv rank-bm25

# !git clone https://github.com/N7/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')





load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Define document path
"""
logger.info("### Define document path")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"

"""
### Encode the pdf to vector store and return split document from the step before to create BM25 instance
"""
logger.info("### Encode the pdf to vector store and return split document from the step before to create BM25 instance")

def encode_pdf_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200):
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

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts

"""
### Create vectorstore and get the chunked documents
"""
logger.info("### Create vectorstore and get the chunked documents")

vectorstore, cleaned_texts = encode_pdf_and_get_split_documents(path)

"""
### Create a bm25 index for retrieving documents by keywords
"""
logger.info("### Create a bm25 index for retrieving documents by keywords")

def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    BM25 (Best Matching 25) is a ranking function used in information retrieval.
    It's based on the probabilistic retrieval framework and is an improvement over TF-IDF.

    Args:
    documents (List[Document]): List of documents to index.

    Returns:
    BM25Okapi: An index that can be used for BM25 scoring.
    """
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

bm25 = create_bm25_index(cleaned_texts) # Create BM25 index from the cleaned texts (chunks)

"""
### Define a function that retrieves both semantically and by keyword, normalizes the scores and gets the top k documents
"""
logger.info("### Define a function that retrieves both semantically and by keyword, normalizes the scores and gets the top k documents")

def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    Perform fusion retrieval combining keyword-based (BM25) and vector-based search.

    Args:
    vectorstore (VectorStore): The vectorstore containing the documents.
    bm25 (BM25Okapi): Pre-computed BM25 index.
    query (str): The query string.
    k (int): The number of documents to retrieve.
    alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

    Returns:
    List[Document]: The top k documents based on the combined scores.
    """

    epsilon = 1e-8

    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    bm25_scores = bm25.get_scores(query.split())

    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) -  np.min(bm25_scores) + epsilon)

    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

    sorted_indices = np.argsort(combined_scores)[::-1]

    return [all_docs[i] for i in sorted_indices[:k]]

"""
### Use Case example
"""
logger.info("### Use Case example")

query = "What are the impacts of climate change on the environment?"

top_docs = fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5)
docs_content = [doc.page_content for doc in top_docs]
show_context(docs_content)

logger.info("\n\n[DONE]", bright=True)