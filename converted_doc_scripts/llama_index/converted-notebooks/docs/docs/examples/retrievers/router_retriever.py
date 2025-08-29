async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    SimpleKeywordTableIndex,
    )
    from llama_index.core import SummaryIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.response.notebook_utils import display_source_node
    from llama_index.core.retrievers import RouterRetriever
    from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
    )
    from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
    from llama_index.core.tools import RetrieverTool
    import logging
    import os
    import shutil
    import sys
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/router_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Router Retriever
    In this guide, we define a custom router retriever that selects one or more candidate retrievers in order to execute a given query.
    
    The router (`BaseSelector`) module uses the LLM to dynamically make decisions on which underlying retrieval tools to use. This can be helpful to select one out of a diverse range of data sources. This can also be helpful to aggregate retrieval results across a variety of data sources (if a multi-selector module is used).
    
    This notebook is very similar to the RouterQueryEngine notebook.
    
    ### Setup
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Router Retriever")
    
    # %pip install llama-index-llms-ollama
    
    # !pip install llama-index
    
    # import nest_asyncio
    
    # nest_asyncio.apply()
    
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    
    
    """
    ### Download Data
    """
    logger.info("### Download Data")
    
    # !mkdir -p 'data/paul_graham/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
    
    """
    ### Load Data
    
    We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.
    """
    logger.info("### Load Data")
    
    documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    
    summary_index = SummaryIndex(nodes, storage_context=storage_context)
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)
    
    list_retriever = summary_index.as_retriever()
    vector_retriever = vector_index.as_retriever()
    keyword_retriever = keyword_index.as_retriever()
    
    
    list_tool = RetrieverTool.from_defaults(
        retriever=list_retriever,
        description=(
            "Will retrieve all context from Paul Graham's essay on What I Worked"
            " On. Don't use if the question only requires more specific context."
        ),
    )
    vector_tool = RetrieverTool.from_defaults(
        retriever=vector_retriever,
        description=(
            "Useful for retrieving specific context from Paul Graham essay on What"
            " I Worked On."
        ),
    )
    keyword_tool = RetrieverTool.from_defaults(
        retriever=keyword_retriever,
        description=(
            "Useful for retrieving specific context from Paul Graham essay on What"
            " I Worked On (using entities mentioned in query)"
        ),
    )
    
    """
    ### Define Selector Module for Routing
    
    There are several selectors available, each with some distinct attributes.
    
    The LLM selectors use the LLM to output a JSON that is parsed, and the corresponding indexes are queried.
    
    The Pydantic selectors (currently only supported by `gpt-4-0613` and `gpt-3.5-turbo-0613` (the default)) use the OllamaFunctionCallingAdapter Function Call API to produce pydantic selection objects, rather than parsing raw JSON.
    
    Here we use PydanticSingleSelector/PydanticMultiSelector but you can use the LLM-equivalents as well.
    """
    logger.info("### Define Selector Module for Routing")
    
    
    """
    #### PydanticSingleSelector
    """
    logger.info("#### PydanticSingleSelector")
    
    retriever = RouterRetriever(
        selector=PydanticSingleSelector.from_defaults(llm=llm),
        retriever_tools=[
            list_tool,
            vector_tool,
        ],
    )
    
    nodes = retriever.retrieve(
        "Can you give me all the context regarding the author's life?"
    )
    for node in nodes:
        display_source_node(node)
    
    nodes = retriever.retrieve("What did Paul Graham do after RISD?")
    for node in nodes:
        display_source_node(node)
    
    """
    #### PydanticMultiSelector
    """
    logger.info("#### PydanticMultiSelector")
    
    retriever = RouterRetriever(
        selector=PydanticMultiSelector.from_defaults(llm=llm),
        retriever_tools=[list_tool, vector_tool, keyword_tool],
    )
    
    nodes = retriever.retrieve(
        "What were noteable events from the authors time at Interleaf and YC?"
    )
    for node in nodes:
        display_source_node(node)
    
    nodes = retriever.retrieve(
        "What were noteable events from the authors time at Interleaf and YC?"
    )
    for node in nodes:
        display_source_node(node)
    
    nodes = await retriever.aretrieve(
            "What were noteable events from the authors time at Interleaf and YC?"
        )
    logger.success(format_json(nodes))
    for node in nodes:
        display_source_node(node)
    
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