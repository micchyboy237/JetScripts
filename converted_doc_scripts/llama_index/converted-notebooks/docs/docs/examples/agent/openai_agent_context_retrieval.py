async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import (
        SimpleDirectoryReader,
        VectorStoreIndex,
        StorageContext,
        load_index_from_storage,
    )
    from llama_index.core import Document
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.settings import Settings
    from llama_index.core.tools import BaseTool
    from llama_index.core.tools import FunctionTool
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.workflow import Context
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from typing import Sequence
    import json
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_context_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Context-Augmented Function Calling Agent
    
    In this tutorial, we show you how to to make your agent context-aware.
    
    Our indexing/retrieval modules help to remove the complexity of having too many functions to fit in the prompt.
    
    ## Initial Setup
    
    Here we setup a normal FunctionAgent, and then augment it with context. This agent will perform retrieval first before calling any tools. This can help ground the agent's tool picking and answering capabilities in context.
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Context-Augmented Function Calling Agent")

    # %pip install llama-index

    # os.environ["OPENAI_API_KEY"] = "sk-..."

    Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/march"
        )
        march_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/june"
        )
        june_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/sept"
        )
        sept_index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False

    """
    Download Data
    """
    logger.info("Download Data")

    # !mkdir -p 'data/10q/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_march_2022.pdf' -O 'data/10q/uber_10q_march_2022.pdf'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_june_2022.pdf' -O 'data/10q/uber_10q_june_2022.pdf'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_sept_2022.pdf' -O 'data/10q/uber_10q_sept_2022.pdf'

    if not index_loaded:
        march_docs = SimpleDirectoryReader(
            input_files=[
                f"{os.path.dirname(__file__)}/data/10q/uber_10q_march_2022.pdf"]
        ).load_data()
        june_docs = SimpleDirectoryReader(
            input_files=[
                f"{os.path.dirname(__file__)}/data/10q/uber_10q_june_2022.pdf"]
        ).load_data()
        sept_docs = SimpleDirectoryReader(
            input_files=[
                f"{os.path.dirname(__file__)}/data/10q/uber_10q_sept_2022.pdf"]
        ).load_data()

        march_index = VectorStoreIndex.from_documents(march_docs)
        june_index = VectorStoreIndex.from_documents(june_docs)
        sept_index = VectorStoreIndex.from_documents(sept_docs)

        march_index.storage_context.persist(persist_dir="./storage/march")
        june_index.storage_context.persist(persist_dir="./storage/june")
        sept_index.storage_context.persist(persist_dir="./storage/sept")

    march_engine = march_index.as_query_engine(similarity_top_k=3)
    june_engine = june_index.as_query_engine(similarity_top_k=3)
    sept_engine = sept_index.as_query_engine(similarity_top_k=3)

    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=march_engine,
            name="uber_march_10q",
            description=(
                "Provides information about Uber 10Q filings for March 2022. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=june_engine,
            name="uber_june_10q",
            description=(
                "Provides information about Uber financials for June 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=sept_engine,
            name="uber_sept_10q",
            description=(
                "Provides information about Uber financials for Sept 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ]

    """
    ### Try Context-Augmented Agent
    
    Here we augment our agent with context in different settings:
    - toy context: we define some abbreviations that map to financial terms (e.g. R=Revenue). We supply this as context to the agent
    """
    logger.info("### Try Context-Augmented Agent")

    texts = [
        "Abbreviation: 'Y' = Revenue",
        "Abbreviation: 'X' = Risk Factors",
        "Abbreviation: 'Z' = Costs",
    ]
    docs = [Document(text=t) for t in texts]
    context_index = VectorStoreIndex.from_documents(docs)
    context_retriever = context_index.as_retriever(similarity_top_k=2)

    system_prompt_template = """You are a helpful assistant.
    Here is some context that you can use to answer the user's question and for help with picking the right tool:
    
    {context}
    """

    async def get_agent_with_context_awareness(
        query: str, context_retriever, tools: list[BaseTool]
    ) -> FunctionAgent:
        context_nodes = await context_retriever.aretrieve(query)
        logger.success(format_json(context_nodes))
        context_text = "\n----\n".join([n.node.text for n in context_nodes])

        return FunctionAgent(
            tools=tools,
            llm=OllamaFunctionCallingAdapter(
                model="llama3.2", request_timeout=300.0, context_window=4096),
            system_prompt=system_prompt_template.format(context=context_text),
        )

    query = "What is the 'X' of March 2022?"
    agent = await get_agent_with_context_awareness(
        query, context_retriever, query_engine_tools
    )
    logger.success(format_json(agent))

    response = await agent.run(query)
    logger.success(format_json(response))

    logger.debug(str(response))

    query = "What is the 'Y' and 'Z' in September 2022?"
    agent = await get_agent_with_context_awareness(
        query, context_retriever, query_engine_tools
    )
    logger.success(format_json(agent))

    response = await agent.run(query)
    logger.success(format_json(response))

    logger.debug(str(response))

    """
    ### Managing Context/Memory
    
    By default, each `.run()` call is stateless. We can manage context by using a serializable `Context` object.
    """
    logger.info("### Managing Context/Memory")

    ctx = Context(agent)

    query = "What is the 'Y' and 'Z' in September 2022?"
    agent = await get_agent_with_context_awareness(
        query, context_retriever, query_engine_tools
    )
    logger.success(format_json(agent))
    response = await agent.run(query, ctx=ctx)
    logger.success(format_json(response))

    query = "What did I just ask?"
    agent = await get_agent_with_context_awareness(
        query, context_retriever, query_engine_tools
    )
    logger.success(format_json(agent))
    response = await agent.run(query, ctx=ctx)
    logger.success(format_json(response))
    logger.debug(str(response))

    """
    ### Use Uber 10-Q as context, use Calculator as Tool
    """
    logger.info("### Use Uber 10-Q as context, use Calculator as Tool")

    def magic_formula(revenue: int, cost: int) -> int:
        """Runs MAGIC_FORMULA on revenue and cost."""
        return revenue - cost

    magic_tool = FunctionTool.from_defaults(magic_formula)

    context_retriever = sept_index.as_retriever(similarity_top_k=3)

    query = "Can you run MAGIC_FORMULA on Uber's revenue and cost?"
    agent = await get_agent_with_context_awareness(
        query, context_retriever, [magic_tool]
    )
    logger.success(format_json(agent))
    response = await agent.run(query)
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
