async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
    from llama_index.core.agent.workflow import ToolCallResult, AgentStream
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.workflow import Context
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_with_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Agent with Query Engine Tools
    
    ## Build Query Engine Tools
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Agent with Query Engine Tools")

    # %pip install llama-index

    # os.environ["OPENAI_API_KEY"] = "sk-..."

    Settings.llm = OllamaFunctionCalling(model="llama3.2")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/lyft"
        )
        lyft_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/uber"
        )
        uber_index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False

    """
    Download Data
    """
    logger.info("Download Data")

    # !mkdir -p 'data/10k/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

    if not index_loaded:
        lyft_docs = SimpleDirectoryReader(
            input_files=[
                "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10k/lyft_2021.pdf"]
        ).load_data()
        uber_docs = SimpleDirectoryReader(
            input_files=[
                "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10k/uber_2021.pdf"]
        ).load_data()

        lyft_index = VectorStoreIndex.from_documents(lyft_docs)
        uber_index = VectorStoreIndex.from_documents(uber_docs)

        lyft_index.storage_context.persist(persist_dir="./storage/lyft")
        uber_index.storage_context.persist(persist_dir="./storage/uber")

    lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
    uber_engine = uber_index.as_query_engine(similarity_top_k=3)

    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=lyft_engine,
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=uber_engine,
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ]

    """
    ## Setup Agent
    
    For LLMs like OllamaFunctionCalling that have a function calling API, we should use the `FunctionAgent`.
    
    For other LLMs, we can use the `ReActAgent`.
    """
    logger.info("## Setup Agent")

    agent = FunctionAgent(tools=query_engine_tools, llm=OllamaFunctionCalling(
        model="llama3.2"))

    ctx = Context(agent)

    """
    ## Let's Try It Out!
    """
    logger.info("## Let's Try It Out!")

    handler = agent.run(
        "What's the revenue for Lyft in 2021 vs Uber?", ctx=ctx)

    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            logger.debug(
                f"Call {ev.tool_name} with args {ev.tool_kwargs}\nReturned: {ev.tool_output}"
            )
        elif isinstance(ev, AgentStream):
            logger.debug(ev.delta, end="", flush=True)

    response = await handler
    logger.success(format_json(response))

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
