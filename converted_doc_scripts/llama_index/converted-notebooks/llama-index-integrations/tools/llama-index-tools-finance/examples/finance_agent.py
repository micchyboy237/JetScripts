async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.finance import FinanceAgentToolSpec
    import os
    import shutil


    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Finance Agent Tool Spec
    
    This tool connects to various open finance apis and libraries to gather news, earnings information and doing fundamental analysis.
    
    To use this tool, you'll need a few API keys:
    
    - POLYGON_API_KEY -- <https://polygon.io/>
    - FINNHUB_API_KEY -- <https://finnhub.io/>
    - ALPHA_VANTAGE_API_KEY -- <https://www.alphavantage.co/>
    - NEWSAPI_API_KEY -- <https://newsapi.org/>
    
    ## Installation
    """
    logger.info("# Finance Agent Tool Spec")

    # !pip install llama-index-tools-finance

    """
    ## Usage
    """
    logger.info("## Usage")


    POLYGON_API_KEY = ""
    FINNHUB_API_KEY = ""
    ALPHA_VANTAGE_API_KEY = ""
    NEWSAPI_API_KEY = ""
    # OPENAI_API_KEY = ""

    GPT_MODEL_NAME = "gpt-4-0613"


    def create_agent(
        polygon_api_key: str,
        finnhub_api_key: str,
        alpha_vantage_api_key: str,
        newsapi_api_key: str,
        openai_api_key: str,
    ) -> FunctionAgent:
        tool_spec = FinanceAgentToolSpec(
            polygon_api_key, finnhub_api_key, alpha_vantage_api_key, newsapi_api_key
        )
        llm = OllamaFunctionCalling(temperature=0, model=GPT_MODEL_NAME, api_key=openai_api_key)
        return FunctionAgent(
            tools=tool_spec.to_tool_list(),
            llm=llm,
        )


    agent = create_agent(
        POLYGON_API_KEY,
        FINNHUB_API_KEY,
        ALPHA_VANTAGE_API_KEY,
        NEWSAPI_API_KEY,
    #     OPENAI_API_KEY,
    )

    response = await agent.run("What happened to AAPL stock on February 19th, 2024?")
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