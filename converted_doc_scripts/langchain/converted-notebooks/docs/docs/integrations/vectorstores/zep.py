from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import ZepVectorStore
from langchain_community.vectorstores.zep import CollectionConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
from zep_python import ZepClient
import os
import shutil
import time

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
    # Zep
    > Recall, understand, and extract data from chat histories. Power personalized AI experiences.
    
    > [Zep](https://www.getzep.com) is a long-term memory service for AI Assistant apps.
    > With Zep, you can provide AI assistants with the ability to recall past conversations, no matter how distant,
    > while also reducing hallucinations, latency, and cost.
    
    > Interested in Zep Cloud? See [Zep Cloud Installation Guide](https://help.getzep.com/sdks) and [Zep Cloud Vector Store example](https://help.getzep.com/langchain/examples/vectorstore-example)
    
    ## Open Source Installation and Setup
    
    > Zep Open Source project: [https://github.com/getzep/zep](https://github.com/getzep/zep)
    >
    > Zep Open Source Docs: [https://docs.getzep.com/](https://docs.getzep.com/)
    
    You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
    
    ## Usage
    
    In the examples below, we're using Zep's auto-embedding feature which automatically embeds documents on the Zep server 
    using low-latency embedding models.
    
    ## Note
    - These examples use Zep's async interfaces. Call sync interfaces by removing the `a` prefix from the method names.
    - If you pass in an `Embeddings` instance Zep will use this to embed documents rather than auto-embed them.
    You must also set your document collection to `isAutoEmbedded === false`. 
    - If you set your collection to `isAutoEmbedded === false`, you must pass in an `Embeddings` instance.
    
    ## Load or create a Collection from documents
    """
    logger.info("# Zep")
    
    
    
    ZEP_API_URL = "http://localhost:8000"  # this is the API url of your Zep instance
    ZEP_API_KEY = "<optional_key>"  # optional API Key for your Zep instance
    collection_name = f"babbage{uuid4().hex}"  # a unique collection name. alphanum only
    
    config = CollectionConfig(
        name=collection_name,
        description="<optional description>",
        metadata={"optional_metadata": "associated with the collection"},
        is_auto_embedded=True,  # we'll have Zep embed our documents using its low-latency embedder
        embedding_dimensions=1536,  # this should match the model you've configured Zep to use.
    )
    
    article_url = "https://www.gutenberg.org/cache/epub/71292/pg71292.txt"
    loader = WebBaseLoader(article_url)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    vs = ZepVectorStore.from_documents(
        docs,
        collection_name=collection_name,
        config=config,
        api_url=ZEP_API_URL,
        api_key=ZEP_API_KEY,
        embedding=None,  # we'll have Zep embed our documents using its low-latency embedder
    )
    
    async def wait_for_ready(collection_name: str) -> None:
    
    
        client = ZepClient(ZEP_API_URL, ZEP_API_KEY)
    
        while True:
            c = await client.document.aget_collection(collection_name)
            logger.success(format_json(c))
            logger.debug(
                "Embedding status: "
                f"{c.document_embedded_count}/{c.document_count} documents embedded"
            )
            time.sleep(1)
            if c.status == "ready":
                break
    
    
    await wait_for_ready(collection_name)
    
    """
    ## Simarility Search Query over the Collection
    """
    logger.info("## Simarility Search Query over the Collection")
    
    query = "what is the structure of our solar system?"
    docs_scores = await vs.asimilarity_search_with_relevance_scores(query, k=3)
    logger.success(format_json(docs_scores))
    
    for d, s in docs_scores:
        logger.debug(d.page_content, " -> ", s, "\n====\n")
    
    """
    ## Search over Collection Re-ranked by MMR
    
    Zep offers native, hardware-accelerated MMR re-ranking of search results.
    """
    logger.info("## Search over Collection Re-ranked by MMR")
    
    query = "what is the structure of our solar system?"
    docs = await vs.asearch(query, search_type="mmr", k=3)
    logger.success(format_json(docs))
    
    for d in docs:
        logger.debug(d.page_content, "\n====\n")
    
    """
    # Filter by Metadata
    
    Use a metadata filter to narrow down results. First, load another book: "Adventures of Sherlock Holmes"
    """
    logger.info("# Filter by Metadata")
    
    article_url = "https://www.gutenberg.org/files/48320/48320-0.txt"
    loader = WebBaseLoader(article_url)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    await vs.aadd_documents(docs)
    
    await wait_for_ready(collection_name)
    
    """
    We see results from both books. Note the `source` metadata
    """
    logger.info("We see results from both books. Note the `source` metadata")
    
    query = "Was he interested in astronomy?"
    docs = await vs.asearch(query, search_type="similarity", k=3)
    logger.success(format_json(docs))
    
    for d in docs:
        logger.debug(d.page_content, " -> ", d.metadata, "\n====\n")
    
    """
    Now, we set up a filter
    """
    logger.info("Now, we set up a filter")
    
    filter = {
        "where": {
            "jsonpath": (
                "$[*] ? (@.source == 'https://www.gutenberg.org/files/48320/48320-0.txt')"
            )
        },
    }
    
    docs = await vs.asearch(query, search_type="similarity", metadata=filter, k=3)
    logger.success(format_json(docs))
    
    for d in docs:
        logger.debug(d.page_content, " -> ", d.metadata, "\n====\n")
    
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