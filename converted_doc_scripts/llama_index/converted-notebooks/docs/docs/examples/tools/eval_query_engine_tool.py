async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core import (
        StorageContext,
        load_index_from_storage,
    )
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.evaluation import RelevancyEvaluator
    from llama_index.core.settings import Settings
    from llama_index.core.tools import ToolMetadata
    from llama_index.core.tools.eval_query_engine import EvalQueryEngineTool
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
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/tools/eval_query_engine_tool.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Evaluation Query Engine Tool
    
    In this section we will show you how you can use an `EvalQueryEngineTool` with an agent. Some reasons you may want to use a `EvalQueryEngineTool`:
    1. Use specific kind of evaluation for a tool, and not just the agent's reasoning
    2. Use a different LLM for evaluating tool responses than the agent LLM
    
    An `EvalQueryEngineTool` is built on top of the `QueryEngineTool`. Along with wrapping an existing [query engine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html), it also must be given an existing [evaluator](https://docs.llamaindex.ai/en/stable/examples/evaluation/answer_and_context_relevancy.html) to evaluate the responses of that query engine.
    
    ## Install Dependencies
    """
    logger.info("# Evaluation Query Engine Tool")

    # %pip install llama-index-embeddings-huggingface
    # %pip install llama-index-llms-ollama
    # %pip install llama-index-agents-openai

    # os.environ["OPENAI_API_KEY"] = "sk-..."

    """
    ## Initialize and Set LLM and Local Embedding Model
    """
    logger.info("## Initialize and Set LLM and Local Embedding Model")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.llm = OllamaFunctionCalling()

    """
    ## Download and Index Data
    This is something we are donig for the sake of this demo. In production environments, data stores and indexes should already exist and not be created on the fly.
    
    ### Create Storage Contexts
    """
    logger.info("## Download and Index Data")

    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/lyft",
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

    """
    ### Load Data
    """
    logger.info("### Load Data")

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

    """
    ## Create Query Engines
    """
    logger.info("## Create Query Engines")

    lyft_engine = lyft_index.as_query_engine(similarity_top_k=5)
    uber_engine = uber_index.as_query_engine(similarity_top_k=5)

    """
    ## Create Evaluator
    """
    logger.info("## Create Evaluator")

    evaluator = RelevancyEvaluator()

    """
    ## Create Query Engine Tools
    """
    logger.info("## Create Query Engine Tools")

    query_engine_tools = [
        EvalQueryEngineTool(
            evaluator=evaluator,
            query_engine=lyft_engine,
            metadata=ToolMetadata(
                name="lyft",
                description=(
                    "Provides information about Lyft's financials for year 2021. "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
        EvalQueryEngineTool(
            evaluator=evaluator,
            query_engine=uber_engine,
            metadata=ToolMetadata(
                name="uber",
                description=(
                    "Provides information about Uber's financials for year 2021. "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
    ]

    """
    ## Setup OllamaFunctionCalling Agent
    """
    logger.info("## Setup OllamaFunctionCalling Agent")

    agent = FunctionAgent(tools=query_engine_tools, llm=OllamaFunctionCalling(
        model="llama3.2"))

    """
    ## Query Engine Passes Evaluation
    
    Here we are asking a question about Lyft's financials. This is what we should expect to happen:
    1. The agent will use the `lyftk` tool first
    2. The `EvalQueryEngineTool` will evaluate the response of the query engine using its evaluator
    3. The output of the query engine will pass evaluation because it contains Lyft's financials
    """
    logger.info("## Query Engine Passes Evaluation")

    response = await agent.run("What was Lyft's revenue growth in 2021?")
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
