from langchain_core.runnables import RunnableLambda
from jet.adapters.langchain.pg_vector_parent_document_retriever import PgVectorParentDocumentRetriever
from jet.db.postgres.pgvector import PgVectorClient
from jet.file.utils import save_file
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from jet.transformers.formatters import format_json
from datasets import load_dataset
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.agents import tool
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Dict
from typing import Generator, List
from typing_extensions import TypedDict
import asyncio
import os
import pandas as pd
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


async def main():
    """
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/advanced_techniques/langchain_parent_document_retrieval.ipynb)

    [![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/parent-doc-retrieval/?utm_campaign=devrel&utm_source=cross-post&utm_medium=organic_social&utm_content=https%3A%2F%2Fgithub.com%2Fmongodb-developer%2FGenAI-Showcase&utm_term=apoorva.joshi)

    # Parent Document Retrieval Using MongoDB and LangChain

    This notebook shows you how to implement parent document retrieval in your RAG application using MongoDB's LangChain integration.

    ## Step 1: Install required libraries

    - **datasets**: Python package to download datasets from Hugging Face

    - **pymongo**: Python driver for MongoDB

    - **langchain**: Python package for LangChain's core modules

    - **langchain-ollama**: Python package to use Ollama models via LangChain

    - **langgraph**: Python package to orchestrate LLM workflows as graphs

    - **langchain-mongodb**: Python package to use MongoDB features in LangChain

    - **langchain-ollama**: Python package to use Ollama models via LangChain
    """
    logger.info("# Parent Document Retrieval Using MongoDB and LangChain")

    # ! pip install -qU datasets pymongo langchain langgraph langchain-mongodb langchain-ollama

    """
    ## Step 2: Setup prerequisites
    
    - **Set the MongoDB connection string**: Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.
    
    - **Set the Ollama API key**: Steps to obtain an API key are [here](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
    
    - **Set the Hugging Face token**: Steps to create a token are [here](https://huggingface.co/docs/hub/en/security-tokens#how-to-manage-user-access-tokens). You only need **read** token for this tutorial.
    """
    logger.info("## Step 2: Setup prerequisites")

    DB_USER = "jethroestrada"
    DB_PASSWORD = ""
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "langchain"
    COLLECTION_NAME = "parent_doc"

    # import getpass

    # os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Ollama API Key:")

    # MONGODB_URI = getpass.getpass("Enter your MongoDB connection string:")
    pgvector_client = PgVectorClient(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        overwrite_db=True
    )

    # os.environ["HF_TOKEN"] = getpass.getpass("Enter your HF Access Token:")

    """
    ## Step 3: Load the dataset
    """
    logger.info("## Step 3: Load the dataset")

    data = load_dataset("mongodb-eai/docs", streaming=True, split="train")
    data_head = data.take(10)
    df = pd.DataFrame(data_head)

    df.head()

    """
    ## Step 4: Convert dataset to LangChain Documents
    """
    logger.info("## Step 4: Convert dataset to LangChain Documents")

    docs = []
    metadata_fields = ["updated", "url", "title"]
    for _, row in df.iterrows():
        content = row["body"]
        metadata = row["metadata"]
        for field in metadata_fields:
            metadata[field] = row[field]
        docs.append(Document(page_content=content, metadata=metadata))

    docs[0]

    len(docs)

    """
    ## Step 5: Instantiate the retriever
    """
    logger.info("## Step 5: Instantiate the retriever")

    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

    def get_splitter(chunk_size: int) -> RecursiveCharacterTextSplitter:
        """
        Returns a token-based text splitter with overlap

        Args:
            chunk_size (_type_): Chunk size in number of tokens

        Returns:
            RecursiveCharacterTextSplitter: Recursive text splitter object
        """
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=0.15 * chunk_size,
        )

    """
    ### Parent document retriever
    """
    logger.info("### Parent document retriever")

    parent_doc_retriever = PgVectorParentDocumentRetriever(
        client=pgvector_client,
        embedding_model=embedding_model,
        child_splitter=get_splitter(200),
        database_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        parent_table_name=f"{COLLECTION_NAME}_parent",
        child_table_name=f"{COLLECTION_NAME}_child",
        text_key="page_content",
        search_kwargs={"top_k": 10},
    )

    """
    ## Step 6: Ingest documents into MongoDB
    """
    logger.info("## Step 6: Ingest documents into MongoDB")

    BATCH_SIZE = 256
    MAX_CONCURRENCY = 4

    async def process_batch(batch: Generator, semaphore: asyncio.Semaphore) -> None:
        """
        Ingest batches of documents into MongoDB

        Args:
            batch (Generator): Chunk of documents to ingest
            semaphore (as): Asyncio semaphore
        """
        async with semaphore:
            await parent_doc_retriever.aadd_documents(batch)
            logger.debug(f"Processed {len(batch)} documents")

        logger.success(format_json(result))

    def get_batches(docs: List[Document], batch_size: int) -> Generator:
        """
        Return batches of documents to ingest into MongoDB

        Args:
            docs (List[Document]): List of LangChain documents
            batch_size (int): Batch size

        Yields:
            Generator: Batch of documents
        """
        for i in range(0, len(docs), batch_size):
            yield docs[i: i + batch_size]

    async def process_docs(docs: List[Document]) -> List[None]:
        """
        Asynchronously ingest LangChain documents into MongoDB

        Args:
            docs (List[Document]): List of LangChain documents

        Returns:
            List[None]: Results of the task executions
        """
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        batches = get_batches(docs, BATCH_SIZE)

        tasks = []
        for batch in batches:
            tasks.append(process_batch(batch, semaphore))
        results = await asyncio.gather(*tasks)
        logger.success(format_json(results))
        return results

    pgvector_client.drop_all_rows(f"{COLLECTION_NAME}_parent")
    pgvector_client.drop_all_rows(f"{COLLECTION_NAME}_child")
    logger.debug("Deletion complete.")
    results = await parent_doc_retriever.aadd_documents(docs)
    logger.success(format_json(results))

    """
    ## Step 7: Create a vector search index (Removed)
    """

    """
    ## Step 8: Usage
    
    ### In a RAG application
    """
    logger.info("## Step 8: Usage")

    # Update the retrieve dictionary
    retrieve = {
        "context": RunnableLambda(lambda x: parent_doc_retriever.invoke(x)) | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough(),
    }
    template = """Answer the question based only on the following context. If no context is provided, respond with I DON't KNOW: \
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="llama3.2")
    parse_output = StrOutputParser()
    rag_chain = retrieve | prompt | llm | parse_output

    logger.debug(rag_chain.invoke("How do I improve slow queries in MongoDB?"))

    """
    ### In an AI agent
    """
    logger.info("### In an AI agent")

    @tool
    def get_info_about_mongodb(user_query: str) -> str:
        """
        Retrieve information about MongoDB.

        Args:
        user_query (str): The user's query string.

        Returns:
        str: The retrieved information formatted as a string.
        """
        docs = parent_doc_retriever.invoke(user_query)
        context = "\n\n".join([d.page_content for d in docs])
        return context

    tools = [get_info_about_mongodb]

    llm = ChatOllama(model="llama3.2")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "You are a helpful AI assistant."
                " You are provided with tools to answer questions about MongoDB."
                " Think step-by-step and use these tools to get the information required to answer the user query."
                " Do not re-run tools unless absolutely necessary."
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW."
                " You have access to the following tools: {tool_names}."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(tool_names=", ".join(
        [tool.name for tool in tools]))
    llm_with_tools = prompt | llm.bind_tools(tools)

    class GraphState(TypedDict):
        messages: Annotated[list, add_messages]

    def agent(state: GraphState) -> Dict[str, List]:
        """
        Agent node

        Args:
            state (GraphState): Graph state

        Returns:
            Dict[str, List]: Updates to the graph state
        """
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(GraphState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    app = graph.compile()

    render_mermaid_graph(graph, f"{OUTPUT_DIR}/graph_output.png")

    inputs = {
        "messages": [
            ("user", "How do I improve slow queries in MongoDB?"),
        ]
    }

    for output in app.stream(inputs):
        for key, value in output.items():
            logger.debug(f"Node {key}:")
            logger.debug(value)
    logger.debug("---FINAL ANSWER---")
    logger.debug(value["messages"][-1].content)
    save_file(value["messages"][-1], f"{OUTPUT_DIR}/response.json")
    save_file(value["messages"][-1].content, f"{OUTPUT_DIR}/final_answer.md")
    save_file(app, f"{OUTPUT_DIR}/workflow_state.json")

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
