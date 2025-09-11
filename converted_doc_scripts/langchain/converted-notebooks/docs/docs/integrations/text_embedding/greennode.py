from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_greennode import GreenNodeEmbeddings
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
    sidebar_label: GreenNode
    ---
    
    # GreenNodeEmbeddings
    
    >[GreenNode](https://greennode.ai/) is a global AI solutions provider and a **NVIDIA Preferred Partner**, delivering full-stack AI capabilities—from infrastructure to application—for enterprises across the US, MENA, and APAC regions. Operating on **world-class infrastructure** (LEED Gold, TIA‑942, Uptime Tier III), GreenNode empowers enterprises, startups, and researchers with a comprehensive suite of AI services
    
    This notebook provides a guide to getting started with `GreenNodeEmbeddings`. It enables you to perform semantic document search using various built-in connectors or your own custom data sources by generating high-quality vector representations of text.
    
    ## Overview
    ### Integration details
    
    | Provider | Package |
    |:--------:|:-------:|
    | [GreenNode](/docs/integrations/providers/greennode/) | [langchain-greennode](https://python.langchain.com/v0.2/api_reference/langchain_greennode/embeddings/langchain_greennode.embeddingsGreenNodeEmbeddings.html) |
    
    ## Setup
    
    To access GreenNode embedding models you'll need to create a GreenNode account, get an API key, and install the `langchain-greennode` integration package.
    
    ### Credentials
    
    GreenNode requires an API key for authentication, which can be provided either as the `api_key` parameter during initialization or set as the environment variable `GREENNODE_API_KEY`. You can obtain an API key by registering for an account on [GreenNode Serverless AI](https://aiplatform.console.greennode.ai/playground).
    """
    logger.info("# GreenNodeEmbeddings")
    
    # import getpass
    
    if not os.getenv("GREENNODE_API_KEY"):
    #     os.environ["GREENNODE_API_KEY"] = getpass.getpass("Enter your GreenNode API key: ")
    
    """
    If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
    """
    logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")
    
    
    
    """
    ### Installation
    
    The LangChain GreenNode integration lives in the `langchain-greennode` package:
    """
    logger.info("### Installation")
    
    # %pip install -qU langchain-greennode
    
    """
    ## Instantiation
    
    The `GreenNodeEmbeddings` class can be instantiated with optional parameters for the API key and model name:
    """
    logger.info("## Instantiation")
    
    
    embeddings = GreenNodeEmbeddings(
        model="BAAI/bge-m3"  # The default embedding model
    )
    
    """
    ## Indexing and Retrieval
    
    Embedding models play a key role in retrieval-augmented generation (RAG) workflows by enabling both the indexing of content and its efficient retrieval. 
    Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.
    """
    logger.info("## Indexing and Retrieval")
    
    
    text = "LangChain is the framework for building context-aware reasoning applications"
    
    vectorstore = InMemoryVectorStore.from_texts(
        [text],
        embedding=embeddings,
    )
    
    retriever = vectorstore.as_retriever()
    
    retrieved_documents = retriever.invoke("What is LangChain?")
    
    retrieved_documents[0].page_content
    
    """
    ## Direct Usage
    
    The `GreenNodeEmbeddings` class can be used independently to generate text embeddings without the need for a vector store. This is useful for tasks such as similarity scoring, clustering, or custom processing pipelines.
    
    ### Embed single texts
    
    You can embed single texts or documents with `embed_query`:
    """
    logger.info("## Direct Usage")
    
    single_vector = embeddings.embed_query(text)
    logger.debug(str(single_vector)[:100])  # Show the first 100 characters of the vector
    
    """
    ### Embed multiple texts
    
    You can embed multiple texts with `embed_documents`:
    """
    logger.info("### Embed multiple texts")
    
    text2 = (
        "LangGraph is a library for building stateful, multi-actor applications with LLMs"
    )
    two_vectors = embeddings.embed_documents([text, text2])
    for vector in two_vectors:
        logger.debug(str(vector)[:100])  # Show the first 100 characters of the vector
    
    """
    ### Async Support
    
    GreenNodeEmbeddings supports async operations:
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
    
    For more details about the GreenNode Serverless AI API, visit the [GreenNode Serverless AI Documentation](https://aiplatform.console.greennode.ai/api-docs/maas).
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