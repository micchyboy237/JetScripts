from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.logger import CustomLogger
from scipy.special import logsumexp
from typing import Tuple, List, Any
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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/dartboard.ipynb)

# Dartboard RAG: Retrieval-Augmented Generation with Balanced Relevance and Diversity

## Overview
The **Dartboard RAG** process addresses a common challenge in large knowledge bases: ensuring the retrieved information is both relevant and non-redundant. By explicitly optimizing a combined relevance-diversity scoring function, it prevents multiple top-k documents from offering the same information. This approach is drawn from the elegant method in thepaper:

> [*Better RAG using Relevant Information Gain*](https://arxiv.org/abs/2407.12101)

The paper outlines three variations of the core idea—hybrid RAG (dense + sparse), a cross-encoder version, and a vanilla approach. The **vanilla approach** conveys the fundamental concept most directly, and this implementation extends it with optional weights to control the balance between relevance and diversity.

## Motivation

1. **Dense, Overlapping Knowledge Bases**  
   In large databases, documents may repeat similar content, causing redundancy in top-k retrieval.

2. **Improved Information Coverage**  
   Combining relevance and diversity yields a richer set of documents, mitigating the “echo chamber” effect of overly similar content.


## Key Components

1. **Relevance & Diversity Combination**  
   - Computes a score factoring in both how pertinent a document is to the query and how distinct it is from already chosen documents.

2. **Weighted Balancing**  
   - Introduces RELEVANCE_WEIGHT and DIVERSITY_WEIGHT to allow dynamic control of scoring.  
   - Helps in avoiding overly diverse but less relevant results.

3. **Production-Ready Code**  
   - Derived from the official implementation yet reorganized for clarity.  
   - Allows easier integration into existing RAG pipelines.

## Method Details

1. **Document Retrieval**  
   - Obtain an initial set of candidate documents based on similarity (e.g., cosine or BM25).  
   - Typically retrieves top-N candidates as a starting point.

2. **Scoring & Selection**  
   - Each document’s overall score combines **relevance** and **diversity**:  
   - Select the highest-scoring document, then penalize documents that are overly similar to it.  
   - Repeat until top-k documents are identified.

3. **Hybrid / Fusion & Cross-Encoder Support**  
   Essentially, all you need are distances between documents and the query, and distances between documents. You can easily extract these from hybrid / fusion retrieval or from cross-encoder retrieval. The only recommendation I have is to rely less on raking based scores.
   - For **hybrid / fusion retrieval**: Merge similarities (dense and sparse / BM25) into a single distance. This can be achieved by combining cosine similarity over the dense and the sparse vectors (e.g. averaging them). the move to distances is straightforward (1 - mean cosine similarity). 
   - For **cross-encoders**: You can directly use the cross-encoder similarity scores (1- similarity), potentially adjusting with scaling factors.

4. **Balancing & Adjustment**  
   - Tune DIVERSITY_WEIGHT and RELEVANCE_WEIGHT based on your needs and the density of your dataset.  



By integrating both **relevance** and **diversity** into retrieval, the Dartboard RAG approach ensures that top-k documents collectively offer richer, more comprehensive information—leading to higher-quality responses in Retrieval-Augmented Generation systems.

The paper also has an official code implemention, and this code is based on it, but I think this one here is more readable, manageable and production ready.

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Dartboard RAG: Retrieval-Augmented Generation with Balanced Relevance and Diversity")

# !pip install numpy python-dotenv

# !git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')


load_dotenv()
# if not os.getenv('OPENAI_API_KEY'):
    logger.debug("Please enter your Ollama API key: ")
#     os.environ["OPENAI_API_KEY"] = input("Please enter your Ollama API key: ")
else:
#     os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


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
    documents=documents*5 # load every document 5 times to emulate a dense dataset

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)

    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

"""
### Create Vector store
"""
logger.info("### Create Vector store")

chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)

"""
### Some helper functions for using the vector store for retrieval.
this part is same like simple_rag.ipynb, only its using the actual FAISS index (not the wrapper)
"""
logger.info("### Some helper functions for using the vector store for retrieval.")

def idx_to_text(idx:int):
    """
    Convert a Vector store index to the corresponding text.
    """
    docstore_id = chunks_vector_store.index_to_docstore_id[idx]
    document = chunks_vector_store.docstore.search(docstore_id)
    return document.page_content


def get_context(query:str,k:int=5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Retrieve top k context items for a query using top k retrieval.
    """
    q_vec=chunks_vector_store.embedding_function.embed_documents([query])
    _,indices=chunks_vector_store.index.search(np.array(q_vec),k=k)

    texts = [idx_to_text(i) for i in indices[0]]
    return texts

test_query = "What is the main cause of climate change?"

"""
### Regular top k retrieval
- This demonstration shows that when database is dense (here we simulate density by loading each document 5 times), the results are not good, we don't get the most relevant results. Note that the top 3 results are all repetitions of the same document.
"""
logger.info("### Regular top k retrieval")

texts=get_context(test_query,k=3)
show_context(texts)

"""
## Now for the real part :)

### More utils for distances normalization
"""
logger.info("## Now for the real part :)")

def lognorm(dist:np.ndarray, sigma:float):
    """
    Calculate the log-normal probability for a given distance and sigma.
    """
    if sigma < 1e-9:
        return -np.inf * dist
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)

"""
## Greedy Dartboard Search

This is the core algorithm: A search algorithm that selects a diverse set of relevant documents from a collection by balancing two factors: relevance to the query and diversity among selected documents.

Given distances between a query and documents, plus distances between all documents, the algorithm:

1. Selects the most relevant document first
2. Iteratively selects additional documents by combining:
   - Relevance to the original query
   - Diversity from previously selected documents

The balance between relevance and diversity is controlled by weights:
- `DIVERSITY_WEIGHT`: Importance of difference from existing selections
- `RELEVANCE_WEIGHT`: Importance of relevance to query
- `SIGMA`: Smoothing parameter for probability conversion

The algorithm returns both the selected documents and their selection scores, making it useful for applications like search results where you want relevant but varied results.

For example, when searching news articles, it would first return the most relevant article, then find articles that are both on-topic and provide new information, avoiding redundant selections.
"""
logger.info("## Greedy Dartboard Search")

DIVERSITY_WEIGHT = 1.0  # Weight for diversity in document selection
RELEVANCE_WEIGHT = 1.0  # Weight for relevance to query
SIGMA = 0.1  # Smoothing parameter for probability distribution

def greedy_dartsearch(
    query_distances: np.ndarray,
    document_distances: np.ndarray,
    documents: List[str],
    num_results: int
) -> Tuple[List[str], List[float]]:
    """
    Perform greedy dartboard search to select top k documents balancing relevance and diversity.

    Args:
        query_distances: Distance between query and each document
        document_distances: Pairwise distances between documents
        documents: List of document texts
        num_results: Number of documents to return

    Returns:
        Tuple containing:
        - List of selected document texts
        - List of selection scores for each document
    """
    sigma = max(SIGMA, 1e-5)

    query_probabilities = lognorm(query_distances, sigma)
    document_probabilities = lognorm(document_distances, sigma)


    most_relevant_idx = np.argmax(query_probabilities)
    selected_indices = np.array([most_relevant_idx])
    selection_scores = [1.0] # dummy score for the first document
    max_distances = document_probabilities[most_relevant_idx]

    while len(selected_indices) < num_results:
        updated_distances = np.maximum(max_distances, document_probabilities)

        combined_scores = (
            updated_distances * DIVERSITY_WEIGHT +
            query_probabilities * RELEVANCE_WEIGHT
        )

        normalized_scores = logsumexp(combined_scores, axis=1)
        normalized_scores[selected_indices] = -np.inf

        best_idx = np.argmax(normalized_scores)
        best_score = np.max(normalized_scores)

        max_distances = updated_distances[best_idx]
        selected_indices = np.append(selected_indices, best_idx)
        selection_scores.append(best_score)

    selected_documents = [documents[i] for i in selected_indices]
    return selected_documents, selection_scores

"""
## Dartboard Context Retrieval

### Main function for using the dartboard retrieval. This serves instead of get_context (which is simple RAG). It:

1. Takes a text query, vectorizes it, gets the top k documents (and their vectors) via simple RAG
2. Uses these vectors to calculate the similarities to query and between candidate matches
3. Runs the dartboard algorithm to refine the candidate matches to a final list of k documents
4. Returns the final list of documents and their scores
"""
logger.info("## Dartboard Context Retrieval")

def get_context_with_dartboard(
    query: str,
    num_results: int = 5,
    oversampling_factor: int = 3
) -> Tuple[List[str], List[float]]:
    """
    Retrieve most relevant and diverse context items for a query using the dartboard algorithm.

    Args:
        query: The search query string
        num_results: Number of context items to return (default: 5)
        oversampling_factor: Factor to oversample initial results for better diversity (default: 3)

    Returns:
        Tuple containing:
        - List of selected context texts
        - List of selection scores

    Note:
        The function uses cosine similarity converted to distance. Initial retrieval
        fetches oversampling_factor * num_results items to ensure sufficient diversity
        in the final selection.
    """
    query_embedding = chunks_vector_store.embedding_function.embed_documents([query])
    _, candidate_indices = chunks_vector_store.index.search(
        np.array(query_embedding),
        k=num_results * oversampling_factor
    )

    candidate_vectors = np.array(
        chunks_vector_store.index.reconstruct_batch(candidate_indices[0])
    )
    candidate_texts = [idx_to_text(idx) for idx in candidate_indices[0]]

    document_distances = 1 - np.dot(candidate_vectors, candidate_vectors.T)
    query_distances = 1 - np.dot(query_embedding, candidate_vectors.T)

    selected_texts, selection_scores = greedy_dartsearch(
        query_distances,
        document_distances,
        candidate_texts,
        num_results
    )

    return selected_texts, selection_scores

"""
### dartboard retrieval - results on same query, k, and dataset
- As you can see now the top 3 results are not mere repetitions.
"""
logger.info("### dartboard retrieval - results on same query, k, and dataset")

texts,scores=get_context_with_dartboard(test_query,k=3)
show_context(texts)

logger.info("\n\n[DONE]", bright=True)