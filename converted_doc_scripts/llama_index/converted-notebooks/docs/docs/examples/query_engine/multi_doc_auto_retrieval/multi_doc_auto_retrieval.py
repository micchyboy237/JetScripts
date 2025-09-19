async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core import QueryBundle
    from llama_index.core import SummaryIndex
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.async_utils import run_jobs
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexAutoRetriever
    from llama_index.core.schema import IndexNode
    from llama_index.core.vector_stores import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )
    from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
    from llama_index.readers.github import (
        GitHubRepositoryIssuesReader,
        GitHubIssuesClient,
    )
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    import os
    import shutil
    import weaviate

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Structured Hierarchical Retrieval
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    Doing RAG well over multiple documents is hard. A general framework is given a user query, first select the relevant documents before selecting the content inside.
    
    But selecting the documents can be tough - how can we dynamically select documents based on different properties depending on the user query? 
    
    In this notebook we show you our multi-document RAG architecture:
    
    - Represent each document as a concise **metadata** dictionary containing different properties: an extracted summary along with structured metadata.
    - Store this metadata dictionary as filters within a vector database.
    - Given a user query, first do **auto-retrieval** - infer the relevant semantic query and the set of filters to query this data (effectively combining text-to-SQL and semantic search).
    """
    logger.info("# Structured Hierarchical Retrieval")

    # %pip install llama-index-readers-github
    # %pip install llama-index-vector-stores-weaviate
    # %pip install llama-index-llms-ollama

    # !pip install llama-index llama-hub

    """
    ## Setup and Download Data
    
    In this section, we'll load in LlamaIndex Github issues.
    """
    logger.info("## Setup and Download Data")

    # import nest_asyncio

    # nest_asyncio.apply()

    os.environ["GITHUB_TOKEN"] = "ghp_..."
    # os.environ["OPENAI_API_KEY"] = "sk-..."

    github_client = GitHubIssuesClient()
    loader = GitHubRepositoryIssuesReader(
        github_client,
        owner="run-llama",
        repo="llama_index",
        verbose=True,
    )

    orig_docs = loader.load_data()

    limit = 100

    docs = []
    for idx, doc in enumerate(orig_docs):
        doc.metadata["index_id"] = int(doc.id_)
        if idx >= limit:
            break
        docs.append(doc)

    """
    ## Setup the Vector Store and Index
    """
    logger.info("## Setup the Vector Store and Index")

    auth_config = weaviate.AuthApiKey(
        api_key="XRa15cDIkYRT7AkrpqT6jLfE4wropK1c1TGk"
    )
    client = weaviate.Client(
        "https://llama-index-test-v0oggsoz.weaviate.network",
        auth_client_secret=auth_config,
    )

    class_name = "LlamaIndex_docs"

    client.schema.delete_class(class_name)

    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name=class_name
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    doc_index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context
    )

    """
    ## Create IndexNodes for retrieval and filtering
    """
    logger.info("## Create IndexNodes for retrieval and filtering")

    async def aprocess_doc(doc, include_summary: bool = True):
        """Process doc."""
        metadata = doc.metadata

        date_tokens = metadata["created_at"].split("T")[0].split("-")
        year = int(date_tokens[0])
        month = int(date_tokens[1])
        day = int(date_tokens[2])

        assignee = (
            "" if "assignee" not in doc.metadata else doc.metadata["assignee"]
        )
        size = ""
        if len(doc.metadata["labels"]) > 0:
            size_arr = [l for l in doc.metadata["labels"] if "size:" in l]
            size = size_arr[0].split(":")[1] if len(size_arr) > 0 else ""
        new_metadata = {
            "state": metadata["state"],
            "year": year,
            "month": month,
            "day": day,
            "assignee": assignee,
            "size": size,
        }

        summary_index = SummaryIndex.from_documents([doc])
        query_str = "Give a one-sentence concise summary of this issue."
        query_engine = summary_index.as_query_engine(
            llm=OllamaFunctionCalling(model="llama3.2")
        )
        summary_txt = query_engine.query(query_str)
        logger.success(format_json(summary_txt))
        summary_txt = str(summary_txt)

        index_id = doc.metadata["index_id"]
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="index_id", operator=FilterOperator.EQ, value=int(index_id)
                ),
            ]
        )

        index_node = IndexNode(
            text=summary_txt,
            metadata=new_metadata,
            obj=doc_index.as_retriever(filters=filters),
            index_id=doc.id_,
        )

        return index_node

    async def aprocess_docs(docs):
        """Process metadata on docs."""

        index_nodes = []
        tasks = []
        for doc in docs:
            task = aprocess_doc(doc)
            tasks.append(task)

        index_nodes = await run_jobs(tasks, show_progress=True, workers=3)
        logger.success(format_json(index_nodes))

        return index_nodes

    index_nodes = await aprocess_docs(docs)
    logger.success(format_json(index_nodes))

    index_nodes[5].metadata

    """
    ## Create the Top-Level AutoRetriever
    
    We load both the summarized metadata as well as the original docs into the vector database.
    1. **Summarized Metadata**: This goes into the `LlamaIndex_auto` collection.
    2. **Original Docs**: This goes into the `LlamaIndex_docs` collection.
    
    By storing both the summarized metadata as well as the original documents, we can execute our structured, hierarchical retrieval strategies.
    
    We load into a vector database that supports auto-retrieval.
    
    ### Load Summarized Metadata
    
    This goes into `LlamaIndex_auto`
    """
    logger.info("## Create the Top-Level AutoRetriever")

    auth_config = weaviate.AuthApiKey(
        api_key="XRa15cDIkYRT7AkrpqT6jLfE4wropK1c1TGk"
    )
    client = weaviate.Client(
        "https://llama-index-test-v0oggsoz.weaviate.network",
        auth_client_secret=auth_config,
    )

    class_name = "LlamaIndex_auto"

    client.schema.delete_class(class_name)

    vector_store_auto = WeaviateVectorStore(
        weaviate_client=client, index_name=class_name
    )
    storage_context_auto = StorageContext.from_defaults(
        vector_store=vector_store_auto
    )

    index = VectorStoreIndex(
        objects=index_nodes, storage_context=storage_context_auto
    )

    """
    ## Setup Composable Auto-Retriever
    
    In this section we setup our auto-retriever. There's a few steps that we need to perform.
    
    1. **Define the Schema**: Define the vector db schema (e.g. the metadata fields). This will be put into the LLM input prompt when it's deciding what metadata filters to infer.
    2. **Instantiate the VectorIndexAutoRetriever class**: This creates a retriever on top of our summarized metadata index, and takes in the defined schema as input.
    3. **Define a wrapper retriever**: This allows us to postprocess each node into an `IndexNode`, with an index id linking back source document. This will allow us to do recursive retrieval in the next section (which depends on IndexNode objects linking to downstream retrievers/query engines/other Nodes). **NOTE**: We are working on improving this abstraction.
    
    Running this retriever will retrieve based on our text summaries and metadat of our top-level `IndeNode` objects. Then, their underlying retrievers will be used to retrieve content from the specific github issue.
    
    ### 1. Define the Schema
    """
    logger.info("## Setup Composable Auto-Retriever")

    vector_store_info = VectorStoreInfo(
        content_info="Github Issues",
        metadata_info=[
            MetadataInfo(
                name="state",
                description="Whether the issue is `open` or `closed`",
                type="string",
            ),
            MetadataInfo(
                name="year",
                description="The year issue was created",
                type="integer",
            ),
            MetadataInfo(
                name="month",
                description="The month issue was created",
                type="integer",
            ),
            MetadataInfo(
                name="day",
                description="The day issue was created",
                type="integer",
            ),
            MetadataInfo(
                name="assignee",
                description="The assignee of the ticket",
                type="string",
            ),
            MetadataInfo(
                name="size",
                description="How big the issue is (XS, S, M, L, XL, XXL)",
                type="string",
            ),
        ],
    )

    """
    ### 2. Instantiate VectorIndexAutoRetriever
    """
    logger.info("### 2. Instantiate VectorIndexAutoRetriever")

    retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=vector_store_info,
        similarity_top_k=2,
        empty_query_top_k=10,  # if only metadata filters are specified, this is the limit
        verbose=True,
    )

    """
    ## Try It Out
    
    Now we can start retrieving relevant context over Github Issues! 
    
    To complete the RAG pipeline setup we'll combine our recursive retriever with our `RetrieverQueryEngine` to generate a response in addition to the retrieved nodes.
    
    ### Try Out Retrieval
    """
    logger.info("## Try It Out")

    nodes = retriever.retrieve(QueryBundle(
        "Tell me about some issues on 01/11"))

    """
    The result is the source chunks in the relevant docs. 
    
    Let's look at the date attached to the source chunk (was present in the original metadata).
    """
    logger.info("The result is the source chunks in the relevant docs.")

    logger.debug(f"Number of source nodes: {len(nodes)}")
    nodes[0].node.metadata

    """
    ### Plug into `RetrieverQueryEngine`
    
    We plug into RetrieverQueryEngine to synthesize a result.
    """
    logger.info("### Plug into `RetrieverQueryEngine`")

    llm = OllamaFunctionCalling(model="llama3.2")

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    response = query_engine.query("Tell me about some issues on 01/11")

    logger.debug(str(response))

    response = query_engine.query(
        "Tell me about some open issues related to agents"
    )

    logger.debug(str(response))

    """
    ## Concluding Thoughts
    
    This shows you how to create a structured retrieval layer over your document summaries, allowing you to dynamically pull in the relevant documents based on the user query.
    
    You may notice similarities between this and our [multi-document agents](https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents.html). Both architectures are aimed for powerful multi-document retrieval.
    
    The goal of this notebook is to show you how to apply structured querying in a multi-document setting. You can actually apply this auto-retrieval algorithm to our multi-agent setup too. The multi-agent setup is primarily focused on adding agentic reasoning across documents and per documents, alloinwg multi-part queries using chain-of-thought.
    """
    logger.info("## Concluding Thoughts")

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
