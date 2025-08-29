async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.tavily_research.base import TavilyToolSpec
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # Building a Tavily Data Agent
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-tavily-research/examples/tavily.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    This tutorial walks through using the LLM tools provided by the [Tavily API](https://app.tavily.com/) to allow LLMs to easily search and retrieve relevant content from the Internet.
    
    To get started, you will need an [OllamaFunctionCallingAdapter api key](https://platform.openai.com/account/api-keys) and a [Tavily API key](https://app.tavily.com)
    
    We will import the relevant agents and tools and pass them our keys here:
    """
    logger.info("# Building a Tavily Data Agent")
    
    # %pip install llama-index-tools-tavily-research llama-index
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    os.environ["TAVILY_API_KEY"] = "..."
    
    
    tavily_tool = TavilyToolSpec(
        api_key="tvly-api-key",
    )
    
    tavily_tool_list = tavily_tool.to_tool_list()
    for tool in tavily_tool_list:
        logger.debug(tool.metadata.name)
    
    """
    ## Testing the Tavily search tool
    
    We've imported our OllamaFunctionCallingAdapter agent, set up the api key, and initialized our tool. Let's test out the tool before setting up our Agent.
    """
    logger.info("## Testing the Tavily search tool")
    
    tavily_tool.search("What happened in the latest Burning Man festival?", max_results=3)
    
    """
    ### Using the Search tool in an Agent
    
    We can create an agent with access to the Tavily search tool start testing it out:
    """
    logger.info("### Using the Search tool in an Agent")
    
    
    agent = FunctionAgent(
        tools=tavily_tool_list,
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    logger.debug(
        await agent.run(
            "Write a deep analysis in markdown syntax about the latest burning man floods"
        )
    )
    
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