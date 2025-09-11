from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_nebius import ChatNebius
from langchain_nebius import NebiusEmbeddings, NebiusRetriever
from langchain_nebius import NebiusRetrievalTool
import asyncio
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
    
    # Nebius Retriever
    
    The `NebiusRetriever` enables efficient similarity search using embeddings from [Nebius AI Studio](https://studio.nebius.ai/). It leverages high-quality embedding models to enable semantic search over documents.
    
    This retriever is optimized for scenarios where you need to perform similarity search over a collection of documents, but don't need to persist the vectors to a vector database. It performs vector similarity search in-memory using matrix operations, making it efficient for medium-sized document collections.
    
    ## Setup
    
    ### Installation
    
    The Nebius integration can be installed via pip:
    """
    logger.info("# Nebius Retriever")
    
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
    
    The `NebiusRetriever` requires a `NebiusEmbeddings` instance and a list of documents. Here's how to initialize it:
    """
    logger.info("## Instantiation")
    
    
    docs = [
        Document(
            page_content="Paris is the capital of France", metadata={"country": "France"}
        ),
        Document(
            page_content="Berlin is the capital of Germany", metadata={"country": "Germany"}
        ),
        Document(
            page_content="Rome is the capital of Italy", metadata={"country": "Italy"}
        ),
        Document(
            page_content="Madrid is the capital of Spain", metadata={"country": "Spain"}
        ),
        Document(
            page_content="London is the capital of the United Kingdom",
            metadata={"country": "UK"},
        ),
        Document(
            page_content="Moscow is the capital of Russia", metadata={"country": "Russia"}
        ),
        Document(
            page_content="Washington DC is the capital of the United States",
            metadata={"country": "USA"},
        ),
        Document(
            page_content="Tokyo is the capital of Japan", metadata={"country": "Japan"}
        ),
        Document(
            page_content="Beijing is the capital of China", metadata={"country": "China"}
        ),
        Document(
            page_content="Canberra is the capital of Australia",
            metadata={"country": "Australia"},
        ),
    ]
    
    embeddings = NebiusEmbeddings()
    
    retriever = NebiusRetriever(
        embeddings=embeddings,
        docs=docs,
        k=3,  # Number of documents to return
    )
    
    """
    ## Usage
    
    ### Retrieve Relevant Documents
    
    You can use the retriever to find documents related to a query:
    """
    logger.info("## Usage")
    
    query = "What are some capitals in Europe?"
    results = retriever.invoke(query)
    
    logger.debug(f"Query: {query}")
    logger.debug(f"Top {len(results)} results:")
    for i, doc in enumerate(results):
        logger.debug(f"{i + 1}. {doc.page_content} (Country: {doc.metadata['country']})")
    
    """
    ### Using get_relevant_documents
    
    You can also use the `get_relevant_documents` method directly (though `invoke` is the preferred interface):
    """
    logger.info("### Using get_relevant_documents")
    
    query = "What are the capitals in Asia?"
    results = retriever.get_relevant_documents(query)
    
    logger.debug(f"Query: {query}")
    logger.debug(f"Top {len(results)} results:")
    for i, doc in enumerate(results):
        logger.debug(f"{i + 1}. {doc.page_content} (Country: {doc.metadata['country']})")
    
    """
    ### Customizing Number of Results
    
    You can adjust the number of results at query time by passing `k` as a parameter:
    """
    logger.info("### Customizing Number of Results")
    
    query = "Where is France?"
    results = retriever.invoke(query, k=1)  # Override default k
    
    logger.debug(f"Query: {query}")
    logger.debug(f"Top {len(results)} result:")
    for i, doc in enumerate(results):
        logger.debug(f"{i + 1}. {doc.page_content} (Country: {doc.metadata['country']})")
    
    """
    ### Async Support
    
    NebiusRetriever supports async operations:
    """
    logger.info("### Async Support")
    
    
    
    async def retrieve_async():
        query = "What are some capital cities?"
        results = await retriever.ainvoke(query)
        logger.success(format_json(results))
    
        logger.debug(f"Async query: {query}")
        logger.debug(f"Top {len(results)} results:")
        for i, doc in enumerate(results):
            logger.debug(f"{i + 1}. {doc.page_content} (Country: {doc.metadata['country']})")
    
    
    await retrieve_async()
    
    """
    ### Handling Empty Documents
    """
    logger.info("### Handling Empty Documents")
    
    empty_retriever = NebiusRetriever(
        embeddings=embeddings,
        docs=[],
        k=2,  # Empty document list
    )
    
    results = empty_retriever.invoke("What are the capitals of European countries?")
    logger.debug(f"Number of results: {len(results)}")
    
    """
    ## Use within a chain
    
    NebiusRetriever works seamlessly in LangChain RAG pipelines. Here's an example of creating a simple RAG chain with the NebiusRetriever:
    """
    logger.info("## Use within a chain")
    
    
    llm = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast")
    
    prompt = ChatPromptTemplate.from_template(
        """
    Answer the question based only on the following context:
    
    Context:
    {context}
    
    Question: {question}
    """
    )
    
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke("What are three European capitals?")
    logger.debug(answer)
    
    """
    ### Creating a Search Tool
    
    You can use the `NebiusRetrievalTool` to create a tool for agents:
    """
    logger.info("### Creating a Search Tool")
    
    
    tool = NebiusRetrievalTool(
        retriever=retriever,
        name="capital_search",
        description="Search for information about capital cities around the world",
    )
    
    result = tool.invoke({"query": "capitals in Europe", "k": 3})
    logger.debug("Tool results:")
    logger.debug(result)
    
    """
    ## How It Works
    
    The NebiusRetriever works by:
    
    1. During initialization:
       - It stores the provided documents
       - It uses the provided NebiusEmbeddings to compute embeddings for all documents
       - These embeddings are stored in memory for quick retrieval
    
    2. During retrieval (`invoke` or `get_relevant_documents`):
       - It embeds the query using the same embedding model
       - It computes similarity scores between the query embedding and all document embeddings
       - It returns the top-k documents sorted by similarity
    
    This approach is efficient for medium-sized document collections, as it avoids the need for a separate vector database while still providing high-quality semantic search.
    
    ## API reference
    
    For more details about the Nebius AI Studio API, visit the [Nebius AI Studio Documentation](https://studio.nebius.com/api-reference).
    
    
    """
    logger.info("## How It Works")
    
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