async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.workflow import Context
    from llama_index.embeddings.mistralai import MistralAIEmbedding
    from llama_index.llms.mistralai import MistralAI
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/mistral_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Function Calling Mistral Agent
    
    This notebook shows you how to use our Mistral agent, powered by function calling capabilities.
    
    ## Initial Setup
    
    Let's start by importing some simple building blocks.  
    
    The main thing we need is:
    1. the OllamaFunctionCallingAdapter API (using our own `llama_index` LLM class)
    2. a place to keep conversation history 
    3. a definition for tools that our agent can use.
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Function Calling Mistral Agent")
    
    # %pip install llama-index
    # %pip install llama-index-llms-mistralai
    # %pip install llama-index-embeddings-mistralai
    
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
    Make sure your MISTRAL_API_KEY is set. Otherwise explicitly specify the `api_key` parameter.
    """
    logger.info("Make sure your MISTRAL_API_KEY is set. Otherwise explicitly specify the `api_key` parameter.")
    
    
    llm = MistralAI(model="mistral-large-latest", api_key="...")
    
    """
    ## Initialize Mistral Agent
    
    Here we initialize a simple Mistral agent with calculator functions.
    """
    logger.info("## Initialize Mistral Agent")
    
    
    agent = FunctionAgent(
        tools=[multiply, add],
        llm=llm,
    )
    
    """
    ### Chat
    """
    logger.info("### Chat")
    
    response = await agent.run("What is (121 + 2) * 5?")
    logger.success(format_json(response))
    logger.debug(str(response))
    
    logger.debug(response.tool_calls)
    
    """
    ### Managing Context/Memory
    
    By default, `.run()` is stateless. If you want to maintain state, you can pass in a `context` object.
    """
    logger.info("### Managing Context/Memory")
    
    
    ctx = Context(agent)
    
    response = await agent.run("My name is John Doe", ctx=ctx)
    logger.success(format_json(response))
    response = await agent.run("What is my name?", ctx=ctx)
    logger.success(format_json(response))
    
    logger.debug(str(response))
    
    """
    ## Mistral Agent over RAG Pipeline
    
    Build a Mistral agent over a simple 10K document. We use both Mistral embeddings and mistral-medium to construct the RAG pipeline, and pass it to the Mistral agent as a tool.
    """
    logger.info("## Mistral Agent over RAG Pipeline")
    
    # !mkdir -p 'data/10k/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
    
    
    embed_model = MistralAIEmbedding(api_key="...")
    query_llm = MistralAI(model="mistral-medium", api_key="...")
    
    uber_docs = SimpleDirectoryReader(
        input_files=["./data/10k/uber_2021.pdf"]
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
    
    
    agent = FunctionAgent(tools=[query_engine_tool], llm=llm)
    
    response = await agent.run(
            "Tell me both the risk factors and tailwinds for Uber? Do two parallel tool calls."
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