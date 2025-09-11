from datasets import Dataset
from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.llm.ollama.base_langchain.embeddings import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_mongodb import MongoDBAtlasVectorSearch
from ollama import Ollama
from pymongo import MongoClient
from ragas import RunConfig
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator
from tqdm import tqdm
from typing import Dict, List, Optional
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/rag_chunking_strategies.ipynb)

# RAG Series Part 3: Choosing the right chunking strategy for RAG

In this notebook, we will explore and evaluate different chunking techniques for RAG.

## Step 1: Install required libraries
"""
logger.info("# RAG Series Part 3: Choosing the right chunking strategy for RAG")

# ! pip install -qU langchain langchain-ollama langchain-mongodb langchain-experimental ragas pymongo tqdm

"""
## Step 2: Setup pre-requisites

- Set the MongoDB connection string. Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.

- Set the Ollama API key. Steps to obtain an API key as [here](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
"""
logger.info("## Step 2: Setup pre-requisites")

# import getpass


# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Ollama API Key:")
ollama_client = Ollama()

# MONGODB_URI = getpass.getpass("Enter your MongoDB connection string:")

"""
## Step 3: Load the dataset
"""
logger.info("## Step 3: Load the dataset")


web_loader = WebBaseLoader(
    [
        "https://peps.python.org/pep-0483/",
        "https://peps.python.org/pep-0008/",
        "https://peps.python.org/pep-0257/",
    ]
)

pages = web_loader.load()

len(pages)

"""
## Step 4: Define chunking functions
"""
logger.info("## Step 4: Define chunking functions")


def fixed_token_split(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Fixed token chunking

    Args:
        docs (List[Document]): List of documents to chunk
        chunk_size (int): Chunk size (number of tokens)
        chunk_overlap (int): Token overlap between chunks

    Returns:
        List[Document]: List of chunked documents
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def recursive_split(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    language: Optional[Language] = None,
) -> List[Document]:
    """
    Recursive chunking

    Args:
        docs (List[Document]): List of documents to chunk
        chunk_size (int): Chunk size (number of tokens)
        chunk_overlap (int): Token overlap between chunks
        language (Optional[Language], optional): Language enum name. Defaults to None.

    Returns:
        List[Document]: List of chunked documents
    """
    separators = ["\n\n", "\n", " ", ""]

    if language is not None:
        try:
            separators = RecursiveCharacterTextSplitter.get_separators_for_language(
                language
            )
        except (NameError, ValueError):
            logger.debug(
                f"No separators found for language {language}. Using defaults.")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return splitter.split_documents(docs)


def semantic_split(docs: List[Document]) -> List[Document]:
    """
    Semantic chunking

    Args:
        docs (List[Document]): List of documents to chunk

    Returns:
        List[Document]: List of chunked documents
    """
    splitter = SemanticChunker(
        OllamaEmbeddings(model="mxbai-embed-large"), breakpoint_threshold_type="percentile"
    )
    return splitter.split_documents(docs)


"""
## Step 5: Generate the evaluation dataset
"""
logger.info("## Step 5: Generate the evaluation dataset")


RUN_CONFIG = RunConfig(max_workers=4, max_wait=180)

generator_llm = ChatOllama(model="llama3.2")
critic_llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

generator = TestsetGenerator.from_langchain(
    generator_llm, critic_llm, embeddings)

distributions = {simple: 0.5, multi_context: 0.4, reasoning: 0.1}

testset = generator.generate_with_langchain_docs(
    pages, 10, distributions, run_config=RUN_CONFIG
)

testset = testset.to_pandas()

len(testset)

testset.head()

"""
## Step 6: Evaluate chunking strategies
"""
logger.info("## Step 6: Evaluate chunking strategies")


client = MongoClient(
    MONGODB_URI, appname="devrel.showcase.chunking_strategies")
DB_NAME = "evals"
COLLECTION_NAME = "chunking"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]


def create_vector_store(docs: List[Document]) -> MongoDBAtlasVectorSearch:
    """
    Create MongoDB Atlas vector store

    Args:
        docs (List[Document]): List of documents to create the vector store

    Returns:
        MongoDBAtlasVectorSearch: MongoDB Atlas vector store
    """
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    return vector_store

# import nest_asyncio

# nest_asyncio.apply()


tqdm.get_lock().locks = []

QUESTIONS = testset.question.to_list()
GROUND_TRUTH = testset.ground_truth.to_list()


def perform_eval(docs: List[Document]) -> Dict[str, float]:
    """
    Perform RAGAS evaluation

    Args:
        docs (List[Document]): List of documents to create the vector store

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    eval_data = {
        "question": QUESTIONS,
        "ground_truth": GROUND_TRUTH,
        "contexts": [],
    }

    logger.debug(
        f"Deleting existing documents in the collection {DB_NAME}.{COLLECTION_NAME}")
    MONGODB_COLLECTION.delete_many({})
    logger.debug("Deletion complete")
    vector_store = create_vector_store(docs)

    logger.debug("Getting contexts for evaluation set")
    for question in tqdm(QUESTIONS):
        eval_data["contexts"].append(
            [doc.page_content for doc in vector_store.similarity_search(
                question, k=3)]
        )
    dataset = Dataset.from_dict(eval_data)

    logger.debug("Running evals")
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
        run_config=RUN_CONFIG,
        raise_exceptions=False,
    )
    return result


for chunk_size in [100, 200, 500, 1000]:
    chunk_overlap = int(0.15 * chunk_size)
    logger.debug(f"CHUNK SIZE: {chunk_size}")
    logger.debug("------ Fixed token without overlap ------")
    logger.debug(
        f"Result: {perform_eval(fixed_token_split(pages, chunk_size, 0))}")
    logger.debug("------ Fixed token with overlap ------")
    logger.debug(
        f"Result: {perform_eval(fixed_token_split(pages, chunk_size, chunk_overlap))}"
    )
    logger.debug("------ Recursive with overlap ------")
    logger.debug(
        f"Result: {perform_eval(recursive_split(pages, chunk_size, chunk_overlap))}")
    logger.debug("------ Recursive Python splitter with overlap ------")
    logger.debug(
        f"Result: {perform_eval(recursive_split(pages, chunk_size, chunk_overlap, Language.PYTHON))}"
    )
logger.debug("------ Semantic chunking ------")
logger.debug(f"Result: {perform_eval(semantic_split(pages))}")

logger.info("\n\n[DONE]", bright=True)
