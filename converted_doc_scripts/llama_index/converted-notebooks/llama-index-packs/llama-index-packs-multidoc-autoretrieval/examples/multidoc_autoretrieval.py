async def main():
    from jet.transformers.formatters import format_json
    from copy import deepcopy
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import Document, ServiceContext
    from llama_index.core import StorageContext
    from llama_index.core import VectorStoreIndex
    from llama_index.core.async_utils import run_jobs
    from llama_index.core.indices import SummaryIndex
    from llama_index.core.llama_pack import download_llama_pack
    from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
    from llama_index.readers.github import GitHubRepositoryIssuesReader, GitHubIssuesClient
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    from tqdm.asyncio import tqdm_asyncio
    import asyncio
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
    # Multidoc Autoretrieval Pack
    
    This is the LlamaPack version of our structured hierarchical retrieval guide in the [core repo](https://docs.llamaindex.ai/en/stable/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.html).
    
    ## Setup and Download Data
    
    In this section, we'll load in LlamaIndex Github issues.
    """
    logger.info("# Multidoc Autoretrieval Pack")
    
    # %pip install llama-index-readers-github
    # %pip install llama-index-vector-stores-weaviate
    # %pip install llama-index-llms-ollama
    
    # import nest_asyncio
    
    # nest_asyncio.apply()
    
    
    os.environ["GITHUB_TOKEN"] = ""
    
    
    
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
        doc.metadata["index_id"] = doc.id_
        if idx >= limit:
            break
        docs.append(doc)
    
    
    
    async def aprocess_doc(doc, include_summary: bool = True):
        """Process doc."""
        logger.debug(f"Processing {doc.id_}")
        metadata = doc.metadata
    
        date_tokens = metadata["created_at"].split("T")[0].split("-")
        year = int(date_tokens[0])
        month = int(date_tokens[1])
        day = int(date_tokens[2])
    
        assignee = "" if "assignee" not in doc.metadata else doc.metadata["assignee"]
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
            "index_id": doc.id_,
        }
    
        summary_index = SummaryIndex.from_documents([doc])
        query_str = "Give a one-sentence concise summary of this issue."
        query_engine = summary_index.as_query_engine(
            service_context=ServiceContext.from_defaults(llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096))
        )
        summary_txt = str(query_engine.query(query_str))
    
        new_doc = Document(text=summary_txt, metadata=new_metadata)
        return new_doc
    
    
    async def aprocess_docs(docs):
        """Process metadata on docs."""
    
        new_docs = []
        tasks = []
        for doc in docs:
            task = aprocess_doc(doc)
            tasks.append(task)
    
        new_docs = await run_jobs(tasks, show_progress=True, workers=5)
        logger.success(format_json(new_docs))
    
    
        return new_docs
    
    new_docs = await aprocess_docs(docs)
    logger.success(format_json(new_docs))
    
    new_docs[5].metadata
    
    """
    ## Setup Weaviate Indices
    """
    logger.info("## Setup Weaviate Indices")
    
    
    
    auth_config = weaviate.AuthApiKey(api_key="")
    client = weaviate.Client(
        "https://<weaviate-cluster>.weaviate.network",
        auth_client_secret=auth_config,
    )
    
    doc_metadata_index_name = "LlamaIndex_auto"
    doc_chunks_index_name = "LlamaIndex_AutoDoc"
    
    client.schema.delete_class(doc_metadata_index_name)
    client.schema.delete_class(doc_chunks_index_name)
    
    """
    ### Setup Metadata Schema
    
    This is required for autoretrieval; we put this in the prompt.
    """
    logger.info("### Setup Metadata Schema")
    
    
    
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
    ## Download LlamaPack
    """
    logger.info("## Download LlamaPack")
    
    
    MultiDocAutoRetrieverPack = download_llama_pack(
        "MultiDocAutoRetrieverPack", "./multidoc_autoretriever_pack"
    )
    
    pack = MultiDocAutoRetrieverPack(
        client,
        doc_metadata_index_name,
        doc_chunks_index_name,
        new_docs,
        docs,
        vector_store_info,
        auto_retriever_kwargs={
            "verbose": True,
            "similarity_top_k": 2,
            "empty_query_top_k": 10,
        },
        verbose=True,
    )
    
    """
    ## Run LlamaPack
    
    Now let's try the LlamaPack on some queries!
    """
    logger.info("## Run LlamaPack")
    
    response = pack.run("Tell me about some issues on 12/11")
    logger.debug(str(response))
    
    response = pack.run("Tell me about some open issues related to agents")
    logger.debug(str(response))
    
    """
    ### Retriever-only
    
    We can also get the retriever module and just run that.
    """
    logger.info("### Retriever-only")
    
    retriever = pack.get_modules()["recursive_retriever"]
    nodes = retriever.retrieve("Tell me about some open issues related to agents")
    logger.debug(f"Number of source nodes: {len(nodes)}")
    nodes[0].node.metadata
    
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