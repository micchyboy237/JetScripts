from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_nebius import NebiusEmbeddings
from scipy.spatial.distance import cosine
import asyncio
import numpy as np
import os
import shutil

async def main():
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger.basicConfig(filename=log_file)
    logger.info(f"Logs: {log_file}")
    
    PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    """
    ---
    sidebar_label: Nebius
    ---
    
    # Nebius Text Embeddings
    
    [Nebius AI Studio](https://studio.nebius.ai/) provides API access to high-quality embedding models through a unified interface. The Nebius embedding models convert text into numerical vectors that capture semantic meaning, making them useful for various applications like semantic search, clustering, and recommendations.
    
    ## Overview
    
    The `NebiusEmbeddings` class provides access to Nebius AI Studio's embedding models through LangChain. These embeddings can be used for semantic search, document similarity, and other NLP tasks requiring vector representations of text.
    
    ### Integration details
    
    - **Provider**: Nebius AI Studio
    - **Model Types**: Text embedding models
    - **Primary Use Case**: Generate vector representations of text for semantic similarity and retrieval
    - **Available Models**: Various embedding models including BAAI/bge-en-icl and others
    - **Dimensions**: Varies by model (typically 1024-4096 dimensions)
    
    ## Setup
    
    ### Installation
    
    The Nebius integration can be installed via pip:
    """
    logger.info("# Nebius Text Embeddings")
    
    # %pip install --upgrade langchain-nebius
    
    """
    ### Credentials
    
    Nebius requires an API key that can be passed as an initialization parameter `api_key` or set as the environment variable `NEBIUS_API_KEY`. You can obtain an API key by creating an account on [Nebius AI Studio](https://studio.nebius.ai/).
    """
    logger.info("### Credentials")
    
    # import getpass
    
    if "NEBIUS_API_KEY" not in os.environ:
    #     os.environ["NEBIUS_API_KEY"] = getpass.getpass("Enter your Nebius API key: ")
    
    """
    ## Instantiation
    
    The `NebiusEmbeddings` class can be instantiated with optional parameters for the API key and model name:
    """
    logger.info("## Instantiation")
    
    
    embeddings = NebiusEmbeddings(
        model="BAAI/bge-en-icl"  # The default embedding model
    )
    
    """
    ### Available Models
    
    The list of supported models is available at https://studio.nebius.com/?modality=embedding
    
    ## Indexing and Retrieval
    
    Embedding models are often used in retrieval-augmented generation (RAG) flows, both for indexing data and later retrieving it. The following example demonstrates how to use `NebiusEmbeddings` with a vector store for document retrieval.
    """
    logger.info("### Available Models")
    
    
    docs = [
        Document(
            page_content="Machine learning algorithms build mathematical models based on sample data"
        ),
        Document(page_content="Deep learning uses neural networks with many layers"),
        Document(page_content="Climate change is a major global environmental challenge"),
        Document(
            page_content="Neural networks are inspired by the human brain's structure"
        ),
    ]
    
    vector_store = FAISS.from_documents(docs, embeddings)
    
    query = "How does the brain influence AI?"
    results = vector_store.similarity_search(query, k=2)
    
    logger.debug("Search results for query:", query)
    for i, doc in enumerate(results):
        logger.debug(f"Result {i + 1}: {doc.page_content}")
    
    """
    ### Using with InMemoryVectorStore
    
    You can also use the `InMemoryVectorStore` for lightweight applications:
    """
    logger.info("### Using with InMemoryVectorStore")
    
    
    text = "LangChain is a framework for developing applications powered by language models"
    
    vectorstore = InMemoryVectorStore.from_texts(
        [text],
        embedding=embeddings,
    )
    
    retriever = vectorstore.as_retriever()
    
    docs = retriever.invoke("What is LangChain?")
    logger.debug(f"Retrieved document: {docs[0].page_content}")
    
    """
    ## Direct Usage
    
    You can directly use the `NebiusEmbeddings` class to generate embeddings for text without using a vector store.
    
    ### Embedding a Single Text
    
    You can use the `embed_query` method to embed a single piece of text:
    """
    logger.info("## Direct Usage")
    
    query = "What is machine learning?"
    query_embedding = embeddings.embed_query(query)
    
    logger.debug(f"Embedding dimension: {len(query_embedding)}")
    logger.debug(f"First few values: {query_embedding[:5]}")
    
    """
    ### Embedding Multiple Texts
    
    You can embed multiple texts at once using the `embed_documents` method:
    """
    logger.info("### Embedding Multiple Texts")
    
    documents = [
        "Machine learning is a branch of artificial intelligence",
        "Deep learning is a subfield of machine learning",
        "Natural language processing deals with interactions between computers and human language",
    ]
    
    document_embeddings = embeddings.embed_documents(documents)
    
    logger.debug(f"Number of document embeddings: {len(document_embeddings)}")
    logger.debug(f"Each embedding has {len(document_embeddings[0])} dimensions")
    
    """
    ### Async Support
    
    NebiusEmbeddings supports async operations:
    """
    logger.info("### Async Support")
    
    
    
    async def generate_embeddings_async():
        query_result = await embeddings.aembed_query("What is the capital of France?")
        logger.success(format_json(query_result))
        logger.debug(f"Async query embedding dimension: {len(query_result)}")
    
        docs = [
            "Paris is the capital of France",
            "Berlin is the capital of Germany",
            "Rome is the capital of Italy",
        ]
        docs_result = await embeddings.aembed_documents(docs)
        logger.success(format_json(docs_result))
        logger.debug(f"Async document embeddings count: {len(docs_result)}")
    
    
    await generate_embeddings_async()
    
    """
    ### Document Similarity Example
    """
    logger.info("### Document Similarity Example")
    
    
    documents = [
        "Machine learning algorithms build mathematical models based on sample data",
        "Deep learning uses neural networks with many layers",
        "Climate change is a major global environmental challenge",
        "Neural networks are inspired by the human brain's structure",
    ]
    
    embeddings_list = embeddings.embed_documents(documents)
    
    
    def calculate_similarity(embedding1, embedding2):
        return 1 - cosine(embedding1, embedding2)
    
    
    logger.debug("Document Similarity Matrix:")
    for i, emb_i in enumerate(embeddings_list):
        similarities = []
        for j, emb_j in enumerate(embeddings_list):
            similarity = calculate_similarity(emb_i, emb_j)
            similarities.append(f"{similarity:.4f}")
        logger.debug(f"Document {i + 1}: {similarities}")
    
    """
    ## API Reference
    
    For more details about the Nebius AI Studio API, visit the [Nebius AI Studio Documentation](https://studio.nebius.ai/docs/api-reference).
    
    
    """
    logger.info("## API Reference")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())