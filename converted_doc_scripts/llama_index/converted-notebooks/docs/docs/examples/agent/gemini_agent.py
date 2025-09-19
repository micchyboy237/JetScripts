async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from google.genai import types
    from jet.logger import CustomLogger
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.agent.workflow import ToolCallResult
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.workflow import Context
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.google_genai import GoogleGenAI
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Function Calling Google Gemini Agent
    
    This notebook shows you how to use Google Gemini Agent, powered by function calling capabilities.
    
    Google's Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite, Gemini 2.0 Flash models support function calling capabilities. You can find a comprehensive capabilities overview on the [model overview](https://ai.google.dev/gemini-api/docs/models) page.
    
    ## Initial Setup
    
    Let's start by importing some simple building blocks.
    
    The main thing we need is:
    
    1. the Google Gemini API (using our own llama_index LLM class)
    2. a place to keep conversation history
    3. a definition for tools that our agent can use.
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Function Calling Google Gemini Agent")

    # %pip install llama-index-llms-google-genai llama-index -q

    """
    Let's define some very simple calculator tools for our agent.
    """
    logger.info("Let's define some very simple calculator tools for our agent.")

    def multiply(a: int, b: int) -> int:
        """Multiple two integers and returns the result integer"""
        return a * b

    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b

    """
    Make sure your GOOGLE_API_KEY is set. Otherwise explicitly specify the api_key parameter.
    """
    logger.info(
        "Make sure your GOOGLE_API_KEY is set. Otherwise explicitly specify the api_key parameter.")

    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        generation_config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )  # Disables thinking
        ),
    )

    """
    ## Initialize Google Gemini Agent
    
    Here we initialize a simple Google Gemini Agent agent with calculator functions.
    """
    logger.info("## Initialize Google Gemini Agent")

    agent = FunctionAgent(
        tools=[multiply, add],
        llm=llm,
    )

    async def run_agent_verbose(query: str):
        handler = agent.run(query)
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                logger.debug(
                    f"Called tool {event.tool_name} with args {event.tool_kwargs}\nGot result: {event.tool_output}"
                )

        return await handler

    """
    ### Chat
    """
    logger.info("### Chat")

    response = await run_agent_verbose("What is (121 + 2) * 5?")
    logger.success(format_json(response))
    logger.debug(str(response))

    logger.debug(response.tool_calls)

    """
    ### Managing Context/Memory
    
    By default, `.run()` is stateless. If you want to maintain state, you can pass in a context object.
    """
    logger.info("### Managing Context/Memory")

    agent = FunctionAgent(llm=llm)
    ctx = Context(agent)

    response = await agent.run("My name is John Doe", ctx=ctx)
    logger.success(format_json(response))
    response = await agent.run("What is my name?", ctx=ctx)
    logger.success(format_json(response))

    logger.debug(str(response))

    """
    ## Google Gemini Agent over RAG Pipeline
    
    Build a Anthropic agent over a simple 10K document. We use OllamaFunctionCalling embeddings and Gemini 2.0 Flash to construct the RAG pipeline, and pass it to the Gemini 2.5 Flash agent as a tool.
    """
    logger.info("## Google Gemini Agent over RAG Pipeline")

    # !mkdir -p 'data/10k/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
    query_llm = GoogleGenAI(model="gemini-2.0-flash")

    uber_docs = SimpleDirectoryReader(
        input_files=[
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10k/uber_2021.pdf"]
    ).load_data()

    uber_index = VectorStoreIndex.from_documents(
        uber_docs, embed_model=embed_model
    )
    uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=query_llm)
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=uber_engine,
        name="uber_10k",
        description=(
            "Provides information about Uber financials for year 2021. "
            "Use a detailed plain text question as input to the tool."
        ),
    )

    agent = FunctionAgent(tools=[query_engine_tool], llm=llm, verbose=True)

    response = await agent.run(
        "Tell me both the risk factors and tailwinds for Uber?"
    )
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
