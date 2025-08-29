async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.core.agent.workflow import ReActAgent
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
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/react_agent_with_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # ReAct Agent with Query Engine (RAG) Tools
    
    In this section, we show how to setup an agent powered by the ReAct loop for financial analysis.
    
    The agent has access to two "tools": one to query the 2021 Lyft 10-K and the other to query the 2021 Uber 10-K.
    
    Note that you can plug in any LLM to use as a ReAct agent.
    
    ## Build Query Engine Tools
    """
    logger.info("# ReAct Agent with Query Engine (RAG) Tools")
    
    # %pip install llama-index
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    
    Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
    
    
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
            input_files=["./data/10k/lyft_2021.pdf"]
        ).load_data()
        uber_docs = SimpleDirectoryReader(
            input_files=["./data/10k/uber_2021.pdf"]
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
    ## Setup ReAct Agent
    
    Here we setup our ReAct agent with the tools we created above.
    
    You can **optionally** specify a system prompt which will be added to the core ReAct system prompt.
    """
    logger.info("## Setup ReAct Agent")
    
    
    agent = ReActAgent(
        tools=query_engine_tools,
        llm=OllamaFunctionCallingAdapter(model="llama3.2"),
    )
    
    
    ctx = Context(agent)
    
    """
    ## Run Some Example Queries
    
    By streaming the result, we can see the full response, including the thought process and tool calls.
    
    If we wanted to stream only the result, we can buffer the stream and start streaming once `Answer:` is in the response.
    """
    logger.info("## Run Some Example Queries")
    
    
    handler = agent.run("What was Lyft's revenue growth in 2021?", ctx=ctx)
    
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            logger.debug(f"{ev.delta}", end="", flush=True)
    
    response = await handler
    logger.success(format_json(response))
    
    logger.debug(str(response))
    
    handler = agent.run(
        "Compare and contrast the revenue growth of Uber and Lyft in 2021, then give an analysis",
        ctx=ctx,
    )
    
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            logger.debug(f"{ev.delta}", end="", flush=True)
    
    response = await handler
    logger.success(format_json(response))
    
    logger.debug(str(response))
    
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