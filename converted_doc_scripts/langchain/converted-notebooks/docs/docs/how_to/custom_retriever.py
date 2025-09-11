from jet.logger import logger
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
import os
import shutil
    from a native async implementation of `_aget_relevant_documents`.

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
    title: Custom Retriever
    ---
    
    # How to create a custom Retriever
    
    ## Overview
    
    Many LLM applications involve retrieving information from external data sources using a [Retriever](/docs/concepts/retrievers/). 
    
    A retriever is responsible for retrieving a list of relevant [Documents](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) to a given user `query`.
    
    The retrieved documents are often formatted into prompts that are fed into an LLM, allowing the LLM to use the information in the to generate an appropriate response (e.g., answering a user question based on a knowledge base).
    
    ## Interface
    
    To create your own retriever, you need to extend the `BaseRetriever` class and implement the following methods:
    
    | Method                         | Description                                      | Required/Optional |
    |--------------------------------|--------------------------------------------------|-------------------|
    | `_get_relevant_documents`      | Get documents relevant to a query.               | Required          |
    | `_aget_relevant_documents`     | Implement to provide async native support.       | Optional          |
    
    
    The logic inside of `_get_relevant_documents` can involve arbitrary calls to a database or to the web using requests.
    
    :::tip
    By inherting from `BaseRetriever`, your retriever automatically becomes a LangChain [Runnable](/docs/concepts/runnables) and will gain the standard `Runnable` functionality out of the box!
    :::
    
    
    :::info
    You can use a `RunnableLambda` or `RunnableGenerator` to implement a retriever.
    
    The main benefit of implementing a retriever as a `BaseRetriever` vs. a `RunnableLambda` (a custom [runnable function](/docs/how_to/functions)) is that a `BaseRetriever` is a well
    known LangChain entity so some tooling for monitoring may implement specialized behavior for retrievers. Another difference
    is that a `BaseRetriever` will behave slightly differently from `RunnableLambda` in some APIs; e.g., the `start` event
    in `astream_events` API will be `on_retriever_start` instead of `on_chain_start`.
    :::
    
    ## Example
    
    Let's implement a toy retriever that returns all documents whose text contains the text in the user query.
    """
    logger.info("# How to create a custom Retriever")
    
    
    
    
    class ToyRetriever(BaseRetriever):
        """A toy retriever that contains the top k documents that contain the user query.
    
        This retriever only implements the sync method _get_relevant_documents.
    
        If the retriever were to involve file access or network access, it could benefit
    
        As usual, with Runnables, there's a default async implementation that's provided
        that delegates to the sync implementation running on another thread.
        """
    
        documents: List[Document]
        """List of documents to retrieve from."""
        k: int
        """Number of top results to return"""
    
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            """Sync implementations for retriever."""
            matching_documents = []
            for document in self.documents:
                if len(matching_documents) > self.k:
                    return matching_documents
    
                if query.lower() in document.page_content.lower():
                    matching_documents.append(document)
            return matching_documents
    
    """
    ## Test it 🧪
    """
    logger.info("## Test it 🧪")
    
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"type": "dog", "trait": "loyalty"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"type": "cat", "trait": "independence"},
        ),
        Document(
            page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
            metadata={"type": "fish", "trait": "low maintenance"},
        ),
        Document(
            page_content="Parrots are intelligent birds capable of mimicking human speech.",
            metadata={"type": "bird", "trait": "intelligence"},
        ),
        Document(
            page_content="Rabbits are social animals that need plenty of space to hop around.",
            metadata={"type": "rabbit", "trait": "social"},
        ),
    ]
    retriever = ToyRetriever(documents=documents, k=3)
    
    retriever.invoke("that")
    
    """
    It's a **runnable** so it'll benefit from the standard Runnable Interface! 🤩
    """
    logger.info("It's a **runnable** so it'll benefit from the standard Runnable Interface! 🤩")
    
    await retriever.ainvoke("that")
    
    retriever.batch(["dog", "cat"])
    
    for event in retriever.stream_events("bar", version="v1"):
        logger.debug(event)
    
    """
    ## Contributing
    
    We appreciate contributions of interesting retrievers!
    
    Here's a checklist to help make sure your contribution gets added to LangChain:
    
    Documentation:
    
    * The retriever contains doc-strings for all initialization arguments, as these will be surfaced in the [API Reference](https://python.langchain.com/api_reference/langchain/index.html).
    * The class doc-string for the model contains a link to any relevant APIs used for the retriever (e.g., if the retriever is retrieving from wikipedia, it'll be good to link to the wikipedia API!)
    
    Tests:
    
    * [ ] Add unit or integration tests to verify that `invoke` and `ainvoke` work.
    
    Optimizations:
    
    If the retriever is connecting to external data sources (e.g., an API or a file), it'll almost certainly benefit from an async native optimization!
     
    * [ ] Provide a native async implementation of `_aget_relevant_documents` (used by `ainvoke`)
    """
    logger.info("## Contributing")
    
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