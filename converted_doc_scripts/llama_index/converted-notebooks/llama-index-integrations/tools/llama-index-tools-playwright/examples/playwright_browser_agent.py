async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent import FunctionAgent
    from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
    )
    from llama_index.core.agent.workflow import AgentWorkflow
    from llama_index.tools.playwright.base import PlaywrightToolSpec
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # Building a Playwright Browser Agent
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-playwright/examples/playwright_browser_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    This tutorial walks through using the LLM tools provided by the [Playwright](https://playwright.dev/) to allow LLMs to easily navigate and scrape content from the Internet.
    
    ## Instaniation
    """
    logger.info("# Building a Playwright Browser Agent")
    
    # %pip install llama-index-tools-playwright llama-index
    
    # !playwright install
    
    # import nest_asyncio
    
    # nest_asyncio.apply()
    
    
    browser = await PlaywrightToolSpec.create_async_playwright_browser(headless=True)
    logger.success(format_json(browser))
    playwright_tool = PlaywrightToolSpec.from_async_browser(browser)
    
    """
    ## Testing the playwright tools
    
    ### Listing all tools
    """
    logger.info("## Testing the playwright tools")
    
    playwright_tool_list = playwright_tool.to_tool_list()
    for tool in playwright_tool_list:
        logger.debug(tool.metadata.name)
    
    """
    ### Navigating to playwright doc website
    """
    logger.info("### Navigating to playwright doc website")
    
    await playwright_tool.navigate_to("https://playwright.dev/python/docs/intro")
    
    logger.debug(await playwright_tool.get_current_page())
    
    """
    ### Extract all hyperlinks
    """
    logger.info("### Extract all hyperlinks")
    
    logger.debug(await playwright_tool.extract_hyperlinks())
    
    """
    ### Extract all text
    """
    logger.info("### Extract all text")
    
    logger.debug(await playwright_tool.extract_text())
    
    """
    ### Get element
    Get element attributes for navigating to the next page.
    You can retrieve the selector from google chrome dev tools.
    """
    logger.info("### Get element")
    
    element = await playwright_tool.get_elements(
            selector="#__docusaurus_skipToContent_fallback > div > div > main > div > div > div.col.docItemCol_VOVn > div > nav > a",
            attributes=["innerText"],
        )
    logger.success(format_json(element))
    logger.debug(element)
    
    """
    ### Click
    Click on the search bar
    """
    logger.info("### Click")
    
    await playwright_tool.click(
        selector="#__docusaurus > nav > div.navbar__inner > div.navbar__items.navbar__items--right > div.navbarSearchContainer_Bca1 > button"
    )
    
    """
    ### Fill
    Fill in the search bar with "Mouse click"
    """
    logger.info("### Fill")
    
    await playwright_tool.fill(selector="#docsearch-input", value="Mouse click")
    
    """
    Click on the first result, we should be redirected to the Mouse click page
    """
    logger.info("Click on the first result, we should be redirected to the Mouse click page")
    
    await playwright_tool.click(selector="#docsearch-hits0-item-0")
    logger.debug(await playwright_tool.get_current_page())
    
    """
    ## Using the playwright tool with agent
    To get started, you will need an [OllamaFunctionCallingAdapter api key](https://platform.openai.com/account/api-keys)
    """
    logger.info("## Using the playwright tool with agent")
    
    
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    
    
    playwright_tool_list = playwright_tool.to_tool_list()
    
    agent = FunctionAgent(
        tools=playwright_tool_list,
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    logger.debug(
        await agent.run(
            "Navigate to https://blog.samaltman.com/productivity, extract the text on this page and return a summary of the article."
        )
    )
    
    """
    ## Using the playwright tool with agent workflow
    """
    logger.info("## Using the playwright tool with agent workflow")
    
    
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)
    
    workflow = AgentWorkflow.from_tools_or_functions(
        playwright_tool_list,
        llm=llm,
        system_prompt="You are a helpful assistant that can do browser automation and data extraction",
    )
    
    handler = workflow.run(
        user_msg="Navigate to https://blog.samaltman.com/productivity, extract the text on this page and return a summary of the article."
    )
    
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            logger.debug(event.delta, end="", flush=True)
        elif isinstance(event, ToolCallResult):
            logger.debug(event.tool_name)  # the tool name
            logger.debug(event.tool_kwargs)  # the tool kwargs
            logger.debug(event.tool_output)  # the tool output
    
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