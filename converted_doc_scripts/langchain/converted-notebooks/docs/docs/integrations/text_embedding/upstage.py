from jet.logger import logger
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_upstage import UpstageEmbeddings
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
    sidebar_label: Upstage
    ---
    
    # UpstageEmbeddings
    
    This notebook covers how to get started with Upstage embedding models.
    
    ## Installation
    
    Install `langchain-upstage` package.
    
    ```bash
    pip install -U langchain-upstage
    ```
    
    ## Environment Setup
    
    Make sure to set the following environment variables:
    
    - `UPSTAGE_API_KEY`: Your Upstage API key from [Upstage console](https://console.upstage.ai/).
    """
    logger.info("# UpstageEmbeddings")
    
    
    os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"
    
    """
    ## Usage
    
    Initialize `UpstageEmbeddings` class.
    """
    logger.info("## Usage")
    
    
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    
    """
    Use `embed_documents` to embed list of texts or documents.
    """
    logger.info("Use `embed_documents` to embed list of texts or documents.")
    
    doc_result = embeddings.embed_documents(
        ["Sung is a professor.", "This is another document"]
    )
    logger.debug(doc_result)
    
    """
    Use `embed_query` to embed query string.
    """
    logger.info("Use `embed_query` to embed query string.")
    
    query_result = embeddings.embed_query("What does Sung do?")
    logger.debug(query_result)
    
    """
    Use `aembed_documents` and `aembed_query` for async operations.
    """
    logger.info("Use `aembed_documents` and `aembed_query` for async operations.")
    
    await embeddings.aembed_query("My query to look up")
    
    await embeddings.aembed_documents(
        ["This is a content of the document", "This is another document"]
    )
    
    """
    ## Using with vector store
    
    You can use `UpstageEmbeddings` with vector store component. The following demonstrates a simple example.
    """
    logger.info("## Using with vector store")
    
    
    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
    )
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke("Where did Harrison work?")
    logger.debug(docs)
    
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