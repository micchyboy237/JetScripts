async def main():
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.yahoo_finance.base import YahooFinanceToolSpec
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    ## Building a Yahoo Finance Agent
    
    This tutorial walks you through the process of building a Yahoo Finance Agent using the `yahoo_finance` tool. The agent will be able to retrieve stock data, financial statements, and other financial information from Yahoo Finance.
    """
    logger.info("## Building a Yahoo Finance Agent")

    # !pip install llama-index llama-index-tools-yahoo-finance

    # os.environ["OPENAI_API_KEY"] = "sk-your-key"

    finance_tool = YahooFinanceToolSpec()

    finance_tool_list = finance_tool.to_tool_list()
    for tool in finance_tool_list:
        logger.debug(tool.metadata.name)

    logger.debug(finance_tool.balance_sheet("AAPL"))

    agent = FunctionAgent(
        tools=finance_tool_list,
        llm=OllamaFunctionCalling(model="llama3.2"),
    )

    logger.debug(await agent.run("What are the analyst recommendations for AAPL?"))

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
