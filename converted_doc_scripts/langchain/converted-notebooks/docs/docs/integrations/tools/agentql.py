from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_agentql import AgentQLBrowserToolkit
from langchain_agentql.tools import ExtractWebDataBrowserTool
from langchain_agentql.tools import ExtractWebDataTool
from langchain_agentql.tools import GetWebElementBrowserTool
from langchain_agentql.utils import create_async_playwright_browser
from langchain_community.tools.playwright import ClickTool
from langchain_community.tools.playwright import ClickTool, NavigateTool
from langchain_community.tools.playwright import CurrentWebPageTool
from langchain_community.tools.playwright import NavigateTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langgraph.prebuilt import create_react_agent
import os
import shutil

async def main():
    
    
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
    # AgentQL
    
    [AgentQL](https://www.agentql.com/) tools provides web interaction and structured data extraction from any web page using an [AgentQL query](https://docs.agentql.com/agentql-query) or a Natural Language prompt. AgentQL can be used across multiple languages and web pages without breaking over time and change.
    
    ## Overview
    
    AgentQL provides the following three tools:
    
    - **`ExtractWebDataTool`** extracts structured data as JSON from a web page given a URL using either an [AgentQL query](https://docs.agentql.com/agentql-query/query-intro) or a Natural Language description of the data.
    
    The following two tools are also bundled as `AgentQLBrowserToolkit` and must be used with a `Playwright` browser or a remote browser instance via Chrome DevTools Protocal (CDP):
    
    - **`ExtractWebDataBrowserTool`** extracts structured data as JSON from the active web page in a browser using either an [AgentQL query](https://docs.agentql.com/agentql-query/query-intro) or a Natural Language description.
    
    - **`GetWebElementBrowserTool`** finds a web element on the active web page in a browser using a Natural Language description and returns its CSS selector for further interaction.
    
    ### Integration details
    
    | Class | Package | Serializable | [JS support](https://js.langchain.com/docs/integrations/tools/langchain_agentql) |  Package latest |
    | :--- | :--- | :---: | :---: | :---: |
    | AgentQL | langchain-agentql | ❌ | ❌ |  1.0.0 |
    
    ### Tool features
    
    | Tool | Web Data Extraction | Web Element Extraction | Use With Local Browser |
    | :--- | :---: | :---: | :---: |
    | ExtractWebDataTool | ✅ | ❌ | ❌
    | ExtractWebDataBrowserTool | ✅ | ❌ | ✅
    | GetWebElementBrowserTool | ❌ | ✅ | ✅
    
    ## Setup
    """
    logger.info("# AgentQL")
    
    # %pip install --quiet -U langchain-agentql
    
    """
    To run this notebook, install `Playwright` browser and configure Jupyter Notebook's `asyncio` loop.
    """
    logger.info("To run this notebook, install `Playwright` browser and configure Jupyter Notebook's `asyncio` loop.")
    
    # !playwright install
    
    # import nest_asyncio
    
    # nest_asyncio.apply()
    
    """
    ### Credentials
    
    To use the AgentQL tools, you will need to get your own API key from the [AgentQL Dev Portal](https://dev.agentql.com/) and set the AgentQL environment variable.
    """
    logger.info("### Credentials")
    
    
    os.environ["AGENTQL_API_KEY"] = "YOUR_AGENTQL_API_KEY"
    
    """
    ## Instantiation
    
    ### `ExtractWebDataTool`
    You can instantiate `ExtractWebDataTool` with the following params:
    - `api_key`: Your AgentQL API key from [dev.agentql.com](https://dev.agentql.com). **`Optional`.**
    - `timeout`: The number of seconds to wait for a request before timing out. Increase if data extraction times out. **Defaults to `900`.**
    - `is_stealth_mode_enabled`: Whether to enable experimental anti-bot evasion strategies. This feature may not work for all websites at all times. Data extraction may take longer to complete with this mode enabled. **Defaults to `False`.**
    - `wait_for`: The number of seconds to wait for the page to load before extracting data. **Defaults to `0`.**
    - `is_scroll_to_bottom_enabled`: Whether to scroll to bottom of the page before extracting data. **Defaults to `False`.**
    - `mode`: `"standard"` uses deep data analysis, while `"fast"` trades some depth of analysis for speed and is adequate for most usecases. [Learn more about the modes in this guide.](https://docs.agentql.com/accuracy/standard-mode) **Defaults to `"fast"`.**
    - `is_screenshot_enabled`: Whether to take a screenshot before extracting data. Returned in 'metadata' as a Base64 string. **Defaults to `False`.**
    
    `ExtractWebDataTool` is implemented with AgentQL's REST API, you can view more details about the parameters in the [API Reference docs](https://docs.agentql.com/rest-api/api-reference).
    """
    logger.info("## Instantiation")
    
    
    extract_web_data_tool = ExtractWebDataTool()
    
    """
    ### `ExtractWebDataBrowserTool`
    
    To instantiate **ExtractWebDataBrowserTool**, you need to connect the tool with a browser instance.
    
    You can set the following params:
    - `timeout`: The number of seconds to wait for a request before timing out. Increase if data extraction times out. **Defaults to `900`.**
    - `wait_for_network_idle`: Whether to wait until the network reaches a full idle state before executing. **Defaults to `True`.**
    - `include_hidden`: Whether to take into account visually hidden elements on the page. **Defaults to `True`.**
    - `mode`: `"standard"` uses deep data analysis, while `"fast"` trades some depth of analysis for speed and is adequate for most usecases. [Learn more about the modes in this guide.](https://docs.agentql.com/accuracy/standard-mode) **Defaults to `"fast"`.**
    
    `ExtractWebDataBrowserTool` is implemented with AgentQL's SDK. You can find more details about the parameters and the functions in AgentQL's [API References](https://docs.agentql.com/python-sdk/api-references/agentql-page#querydata).
    """
    logger.info("### `ExtractWebDataBrowserTool`")
    
    
    async_browser = await create_async_playwright_browser()
    logger.success(format_json(async_browser))
    
    extract_web_data_browser_tool = ExtractWebDataBrowserTool(async_browser=async_browser)
    
    """
    ### `GetWebElementBrowserTool`
    
    To instantiate **GetWebElementBrowserTool**, you need to connect the tool with a browser instance.
    
    You can set the following params:
    - `timeout`: The number of seconds to wait for a request before timing out. Increase if data extraction times out. **Defaults to `900`.**
    - `wait_for_network_idle`: Whether to wait until the network reaches a full idle state before executing. **Defaults to `True`.**
    - `include_hidden`: Whether to take into account visually hidden elements on the page. **Defaults to `False`.**
    - `mode`: `"standard"` uses deep data analysis, while `"fast"` trades some depth of analysis for speed and is adequate for most usecases. [Learn more about the modes in this guide.](https://docs.agentql.com/accuracy/standard-mode) **Defaults to `"fast"`.**
    
    `GetWebElementBrowserTool` is implemented with AgentQL's SDK. You can find more details about the parameters and the functions in AgentQL's [API References](https://docs.agentql.com/python-sdk/api-references/agentql-page#queryelements).`
    """
    logger.info("### `GetWebElementBrowserTool`")
    
    
    extract_web_element_tool = GetWebElementBrowserTool(async_browser=async_browser)
    
    """
    ## Invocation
    
    ### `ExtractWebDataTool`
    
    This tool uses AgentQL's REST API under the hood, sending the publically available web page's URL to AgentQL's endpoint. This will not work with private pages or logged in sessions. Use `ExtractWebDataBrowserTool` for those usecases.
    
    - `url`: The URL of the web page you want to extract data from.
    - `query`: The AgentQL query to execute. Use AgentQL query if you want to extract precisely structured data. Learn more about [how to write an AgentQL query in the docs](https://docs.agentql.com/agentql-query) or test one out in the [AgentQL Playground](https://dev.agentql.com/playground).
    - `prompt`: A Natural Language description of the data to extract from the page. AgentQL will infer the data’s structure from your prompt. Use `prompt` if you want to extract data defined by free-form language without defining a particular structure.  
    
    **Note:** You must define either a `query` or a `prompt` to use AgentQL.
    """
    logger.info("## Invocation")
    
    extract_web_data_tool.invoke(
        {
            "url": "https://www.agentql.com/blog",
            "query": "{ posts[] { title url date author } }",
        },
    )
    
    """
    ### `ExtractWebDataBrowserTool`
    - `query`: The AgentQL query to execute. Use AgentQL query if you want to extract precisely structured data. Learn more about [how to write an AgentQL query in the docs](https://docs.agentql.com/agentql-query) or test one out in the [AgentQL Playground](https://dev.agentql.com/playground).
    - `prompt`: A Natural Language description of the data to extract from the page. AgentQL will infer the data’s structure from your prompt. Use `prompt` if you want to extract data defined by free-form language without defining a particular structure.  
    
    **Note:** You must define either a `query` or a `prompt` to use AgentQL.
    
    To extract data, first you must navigate to a web page using LangChain's [Playwright](https://python.langchain.com/docs/integrations/tools/playwright/) tool.
    """
    logger.info("### `ExtractWebDataBrowserTool`")
    
    
    navigate_tool = NavigateTool(async_browser=async_browser)
    await navigate_tool.ainvoke({"url": "https://www.agentql.com/blog"})
    
    await extract_web_data_browser_tool.ainvoke(
        {"prompt": "the blog posts with title, url, date of post and author"}
    )
    
    """
    ### `GetWebElementBrowserTool`
    - `prompt`: A Natural Language description of the web element to find on the page.
    """
    logger.info("### `GetWebElementBrowserTool`")
    
    selector = await extract_web_element_tool.ainvoke({"prompt": "Next page button"})
    logger.success(format_json(selector))
    selector
    
    
    await ClickTool(async_browser=async_browser, visible_only=False).ainvoke(
        {"selector": selector}
    )
    
    
    await CurrentWebPageTool(async_browser=async_browser).ainvoke({})
    
    """
    ## Chaining
    
    You can use AgentQL tools in a chain by first binding one to a [tool-calling model](/docs/how_to/tool_calling/) and then calling it:
    
    ### Instantiate LLM
    """
    logger.info("## Chaining")
    
    
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    
    
    llm = init_chat_model(model="llama3.2", model_provider="ollama")
    
    """
    ### Execute Tool Chain
    """
    logger.info("### Execute Tool Chain")
    
    
    prompt = ChatPromptTemplate(
        [
            ("system", "You are a helpful assistant in extracting data from website."),
            ("human", "{user_input}"),
            ("placeholder", "{messages}"),
        ]
    )
    
    llm_with_tools = llm.bind_tools(
        [extract_web_data_tool], tool_choice="extract_web_data_with_rest_api"
    )
    
    llm_chain = prompt | llm_with_tools
    
    
    @chain
    def tool_chain(user_input: str, config: RunnableConfig):
        input_ = {"user_input": user_input}
        ai_msg = llm_chain.invoke(input_, config=config)
        tool_msgs = extract_web_data_tool.batch(ai_msg.tool_calls, config=config)
        return {"messages": tool_msgs}
    
    
    tool_chain.invoke(
        "Extract data from https://www.agentql.com/blog using the following agentql query: { posts[] { title url date author } }"
    )
    
    """
    ## Use within an agent
    
    You can use AgentQL tools with an AI Agent using the `AgentQLBrowserToolkit` . This toolkit includes `ExtractDataBrowserTool` and `GetWebElementBrowserTool`. Here's an example of agentic browser actions that combine AgentQL's toolkit with the Playwright tools.
    
    ### Instantiate Toolkit
    """
    logger.info("## Use within an agent")
    
    
    async_agent_browser = await create_async_playwright_browser()
    logger.success(format_json(async_agent_browser))
    
    
    agentql_toolkit = AgentQLBrowserToolkit(async_browser=async_agent_browser)
    agentql_toolkit.get_tools()
    
    
    playwright_toolkit = [
        NavigateTool(async_browser=async_agent_browser),
        ClickTool(async_browser=async_agent_browser, visible_only=False),
    ]
    playwright_toolkit
    
    """
    ### Use with a ReAct Agent
    """
    logger.info("### Use with a ReAct Agent")
    
    # %pip install --quiet -U langgraph
    
    
    agent_executor = create_react_agent(
        llm, agentql_toolkit.get_tools() + playwright_toolkit
    )
    
    prompt = """
    Navigate to https://news.ycombinator.com/,
    extract the news titles on the current page,
    show the current page url,
    find the button on the webpage that direct to the next page,
    click on the button,
    show the current page url,
    extract the news title on the current page
    extract the news titles that mention "AI" from the two pages.
    """
    
    events = agent_executor.astream(
        {"messages": [("user", prompt)]},
        stream_mode="values",
    )
    async for event in events:
        event["messages"][-1].pretty_logger.debug()
    
    """
    ## API reference
    
    For more information on how to use this integration, please refer to the [git repo](https://github.com/tinyfish-io/agentql-integrations/tree/main/langchain) or the [langchain integration documentation](https://docs.agentql.com/integrations/langchain)
    """
    logger.info("## API reference")
    
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