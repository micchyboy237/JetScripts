async def main():
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent import FunctionAgent
    from llama_index.tools.linkup_research.base import LinkupToolSpec
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Building a Linkup Data Agent
    
    This tutorial walks through using the LLM tools provided by the [Linkup API](https://app.linkup.so/) to allow LLMs to easily search and retrieve relevant content from the Internet.
    
    To get started, you will need an [OllamaFunctionCalling api key](https://platform.openai.com/account/api-keys) and a [Linkup API key](https://app.linkup.so)
    
    We will import the relevant agents and tools and pass them our keys here:
    """
    logger.info("# Building a Linkup Data Agent")

    # %pip install llama-index-tools-linkup-research llama-index

    # os.environ["OPENAI_API_KEY"] = "sk-..."

    linkup_tool = LinkupToolSpec(
        api_key="your Linkup API Key",
        # Choose (standard) for a faster result (deep) for a slower but more complete result.
        depth="",
        # Choose (searchResults) for a list of results relative to your query, (sourcedAnswer) for an answer and a list of sources, or (structured) if you want a specific shema.
        output_type="",
    )

    agent = FunctionAgent(
        tools=linkup_tool.to_tool_list(),
        llm=OllamaFunctionCalling(model="llama3.2"),
    )

    logger.debug(await agent.run("Can you tell me which women were awarded the Physics Nobel Prize"))

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
