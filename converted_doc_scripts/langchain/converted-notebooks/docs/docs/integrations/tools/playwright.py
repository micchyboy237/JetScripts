from jet.transformers.formatters import format_json
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
from langgraph.prebuilt import create_react_agent
import os
import shutil

async def main():
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",	  },
    )
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger.basicConfig(filename=log_file)
    logger.info(f"Logs: {log_file}")
    
    PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    """
    # PlayWright Browser Toolkit
    
    >[Playwright](https://github.com/microsoft/playwright) is an open-source automation tool developed by `Microsoft` that allows you to programmatically control and automate web browsers. It is designed for end-to-end testing, scraping, and automating tasks across various web browsers such as `Chromium`, `Firefox`, and `WebKit`.
    
    This toolkit is used to interact with the browser. While other tools (like the `Requests` tools) are fine for static sites, `PlayWright Browser` toolkits let your agent navigate the web and interact with dynamically rendered sites. 
    
    Some tools bundled within the `PlayWright Browser` toolkit include:
    
    - `NavigateTool` (navigate_browser) - navigate to a URL
    - `NavigateBackTool` (previous_page) - wait for an element to appear
    - `ClickTool` (click_element) - click on an element (specified by selector)
    - `ExtractTextTool` (extract_text) - use beautiful soup to extract text from the current web page
    - `ExtractHyperlinksTool` (extract_hyperlinks) - use beautiful soup to extract hyperlinks from the current web page
    - `GetElementsTool` (get_elements) - select elements by CSS selector
    - `CurrentPageTool` (current_page) - get the current page URL
    """
    logger.info("# PlayWright Browser Toolkit")
    
    # %pip install --upgrade --quiet  playwright > /dev/null
    # %pip install --upgrade --quiet  lxml
    
    
    """
    Async function to create context and launch browser:
    """
    logger.info("Async function to create context and launch browser:")
    
    
    # import nest_asyncio
    
    # nest_asyncio.apply()
    
    """
    ## Instantiating a Browser Toolkit
    
    It's always recommended to instantiate using the from_browser method so that the browser context is properly initialized and managed, ensuring seamless interaction and resource optimization.
    """
    logger.info("## Instantiating a Browser Toolkit")
    
    async_browser = create_async_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()
    tools
    
    tools_by_name = {tool.name: tool for tool in tools}
    navigate_tool = tools_by_name["navigate_browser"]
    get_elements_tool = tools_by_name["get_elements"]
    
    await navigate_tool.arun(
        {"url": "https://web.archive.org/web/20230428133211/https://cnn.com/world"}
    )
    
    await get_elements_tool.arun(
        {"selector": ".container__headline", "attributes": ["innerText"]}
    )
    
    await tools_by_name["current_webpage"].arun({})
    
    """
    ## Use within an Agent
    
    Several of the browser tools are `StructuredTool`'s, meaning they expect multiple arguments. These aren't compatible (out of the box) with agents older than the `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION`
    """
    logger.info("## Use within an Agent")
    
    
    llm = ChatOllama(
        model_name="claude-3-haiku-20240307", temperature=0
    )  # or any other LLM, e.g., ChatOllama(model="llama3.2"), Ollama()
    
    agent_chain = create_react_agent(model=llm, tools=tools)
    
    result = await agent_chain.ainvoke(
            {"messages": [("user", "What are the headers on langchain.com?")]}
        )
    logger.success(format_json(result))
    logger.debug(result)
    
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