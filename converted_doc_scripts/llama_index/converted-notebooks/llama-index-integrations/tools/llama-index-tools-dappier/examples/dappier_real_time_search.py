async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.dappier import DappierRealTimeSearchToolSpec
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    ## Building a Dappier Real Time Search Agent
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-dappier/examples/dappier_real_time_search.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    This tutorial walks through using the LLM tools provided by [Dappier](https://dappier.com/) to allow LLMs to use Dappier's pre-trained, LLM ready RAG models and natural language APIs to ensure factual, up-to-date, responses from premium content providers across key verticals like News, Finance, Sports, Weather, and more.
    
    
    To get started, you will need an [OllamaFunctionCallingAdapter API key](https://platform.openai.com/account/api-keys) and a [Dappier API key](https://platform.dappier.com/profile/api-keys)
    
    We will import the relevant agents and tools and pass them our keys here:
    
    ## Installation
    
    First, install the `llama-index` and `llama-index-tools-dappier` packages
    """
    logger.info("## Building a Dappier Real Time Search Agent")
    
    # %pip install llama-index llama-index-tools-dappier
    
    """
    ## Setup API keys
    
    You'll need to set up your API keys for OllamaFunctionCallingAdapter and Dappier.
    
    You can go to [here](https://platform.openai.com/settings/organization/api-keys) to get API Key from Open AI.
    """
    logger.info("## Setup API keys")
    
    # from getpass import getpass
    
    # openai_api_key = getpass("Enter your API key: ")
    # os.environ["OPENAI_API_KEY"] = openai_api_key
    
    """
    You can go to [here](https://platform.dappier.com/profile/api-keys) to get API Key from Dappier with **free** credits.
    """
    logger.info("You can go to [here](https://platform.dappier.com/profile/api-keys) to get API Key from Dappier with **free** credits.")
    
    # dappier_api_key = getpass("Enter your API key: ")
    os.environ["DAPPIER_API_KEY"] = dappier_api_key
    
    """
    ## Setup Dappier Tool
    
    Initialize the Dappier Real-Time Search Tool, convert it to a tool list.
    """
    logger.info("## Setup Dappier Tool")
    
    
    dappier_tool = DappierRealTimeSearchToolSpec()
    
    dappier_tool_list = dappier_tool.to_tool_list()
    for tool in dappier_tool_list:
        logger.debug(tool.metadata.name)
    
    """
    ## Usage
    
    We've imported our OllamaFunctionCallingAdapter agent, set up the api key, and initialized our tool. Let's test out the tool before setting up our Agent.
    
    ### Search Real Time Data
    
    Access real-time google web search results including the latest news, weather, travel, deals and more.
    """
    logger.info("## Usage")
    
    logger.debug(dappier_tool.search_real_time_data("How is the weather in New York today?"))
    
    """
    ### Search Stock Market Data
    
    Access real-time financial news, stock prices, and trades from polygon.io, with AI-powered insights and up-to-the-minute updates to keep you informed on all your financial interests.
    """
    logger.info("### Search Stock Market Data")
    
    logger.debug(dappier_tool.search_stock_market_data("latest financial news on Meta"))
    
    """
    ### Using the Dappier Real Time Search tool in an Agent
    
    We can create an agent with access to the Dappier Real Time Search tool start testing it out:
    """
    logger.info("### Using the Dappier Real Time Search tool in an Agent")
    
    
    agent = FunctionAgent(
        tools=dappier_tool_list,
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    logger.debug(
        await agent.run(
            "Analyze next week's weather in New York and provide daily clothing recommendations in a markdown format."
        )
    )
    
    logger.debug(await agent.run("Last 24-hour activity of apple stock"))
    
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