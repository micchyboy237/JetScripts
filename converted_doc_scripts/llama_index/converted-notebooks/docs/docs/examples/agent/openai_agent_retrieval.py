async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
    from llama_index.core.objects import ObjectIndex
    from llama_index.core.tools import FunctionTool
    from llama_index.core.workflow import Context
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Retrieval-Augmented Agents
    
    In this tutorial, we show you how to use our `FunctionAgent` or `ReActAgent` implementation with a tool retriever, 
    to augment any existing agent and store/index an arbitrary number of tools. 
    
    Our indexing/retrieval modules help to remove the complexity of having too many functions to fit in the prompt.
    
    ## Initial Setup
    
    Let's start by importing some simple building blocks.  
    
    The main thing we need is:
    1. the OllamaFunctionCallingAdapter API
    2. a place to keep conversation history 
    3. a definition for tools that our agent can use.
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Retrieval-Augmented Agents")

    # %pip install llama-index

    # os.environ["OPENAI_API_KEY"] = "sk-..."

    """
    Let's define some very simple calculator tools for our agent.
    """
    logger.info("Let's define some very simple calculator tools for our agent.")

    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b

    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b

    def useless(a: int, b: int) -> int:
        """Toy useless function."""
        pass

    multiply_tool = FunctionTool.from_defaults(multiply, name="multiply")
    add_tool = FunctionTool.from_defaults(add, name="add")

    useless_tools = [
        FunctionTool.from_defaults(useless, name=f"useless_{str(idx)}")
        for idx in range(28)
    ]

    all_tools = [multiply_tool] + [add_tool] + useless_tools

    all_tools_map = {t.metadata.name: t for t in all_tools}

    """
    ## Building an Object Index
    
    We have an `ObjectIndex` construct in LlamaIndex that allows the user to use our index data structures over arbitrary objects.
    The ObjectIndex will handle serialiation to/from the object, and use an underying index (e.g. VectorStoreIndex, SummaryIndex, KeywordTableIndex) as the storage mechanism. 
    
    In this case, we have a large collection of Tool objects, and we'd want to define an ObjectIndex over these Tools.
    
    The index comes bundled with a retrieval mechanism, an `ObjectRetriever`. 
    
    This can be passed in to our agent so that it can 
    perform Tool retrieval during query-time.
    """
    logger.info("## Building an Object Index")

    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )

    """
    To reload the index later, we can use the `from_objects_and_index` method.
    """
    logger.info(
        "To reload the index later, we can use the `from_objects_and_index` method.")

    """
    ## Agent w/ Tool Retrieval
    
    Agents in LlamaIndex can be used with a `ToolRetriever` to retrieve tools during query-time.
    
    During query-time, we would first use the `ObjectRetriever` to retrieve a set of relevant Tools. These tools would then be passed into the agent; more specifically, their function signatures would be passed into the OllamaFunctionCallingAdapter Function calling API.
    """
    logger.info("## Agent w/ Tool Retrieval")

    agent = FunctionAgent(
        tool_retriever=obj_index.as_retriever(similarity_top_k=2),
        llm=OllamaFunctionCallingAdapter(model="llama3.2"),
    )

    ctx = Context(agent)

    resp = await agent.run(
        "What's 212 multiplied by 122? Make sure to use Tools", ctx=ctx
    )
    logger.success(format_json(resp))
    logger.debug(str(resp))
    logger.debug(resp.tool_calls)

    resp = await agent.run(
        "What's 212 added to 122 ? Make sure to use Tools", ctx=ctx
    )
    logger.success(format_json(resp))
    logger.debug(str(resp))
    logger.debug(resp.tool_calls)

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
