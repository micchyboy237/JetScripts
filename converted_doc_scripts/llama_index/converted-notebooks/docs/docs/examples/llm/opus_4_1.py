async def main():
    from jet.transformers.formatters import format_json
    from get_code_from_markdown import get_code_from_markdown
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.agent.workflow import ToolCall, ToolCallResult
    from llama_index.llms.anthropic import Anthropic
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # Using Opus 4.1 with LlamaIndex
    
    In this notebook we are going to exploit [Claude Opus 4.1 by Anthropic](https://www.anthropic.com/news/claude-opus-4-1) advanced coding capabilities to create a cute website, and we're going to do it within LlamaIndex!
    
    ## Build an LLM-based assistant with Opus 4.1
    
    **1. Install needed dependencies**
    """
    logger.info("# Using Opus 4.1 with LlamaIndex")
    
    # ! pip install -q llama-index-llms-anthropic get-code-from-markdown
    
    """
    Let's just define a helper function to help us fetch the code from Markdown:
    """
    logger.info("Let's just define a helper function to help us fetch the code from Markdown:")
    
    
    
    def fetch_code_from_markdown(markdown: str) -> str:
        return get_code_from_markdown(markdown, language="html")
    
    """
    Let's now initialize our LLM:
    """
    logger.info("Let's now initialize our LLM:")
    
    # import getpass
    
    # os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
    
    
    llm = Anthropic(model="claude-opus-4-1-20250805", max_tokens=12000)
    
    res = llm.complete(
        "Can you build a llama-themed static HTML page, with cute little bouncing animations and blue/white/indigo as theme colors?"
    )
    
    """
    Let's now get the code and write it to an HTML file!
    """
    logger.info("Let's now get the code and write it to an HTML file!")
    
    html_code = fetch_code_from_markdown(res.text)
    
    with open("index.html", "w") as f:
        for block in html_code:
            f.write(block)
    
    """
    You can now download `index.html` and take a look at the results :)
    
    ![Llama Paradise HTML](./llama_paradise.png)
    
    ## Build an agent with Opus 4.1
    
    We can also build a simple calculator agent using Claude Opus 4.1
    """
    logger.info("## Build an agent with Opus 4.1")
    
    
    
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and return an integer"""
        return a * b
    
    
    def add(a: int, b: int) -> int:
        """Sum two integers and return an integer"""
        return a + b
    
    
    agent = FunctionAgent(
        name="CalculatorAgent",
        description="Useful to perform basic arithmetic operations",
        system_prompt="You are a calculator agent, you should perform arithmetic operations using the tools available to you.",
        tools=[multiply, add],
        llm=llm,
    )
    
    """
    Let's now run the agent through and get the result for a multiplication:
    """
    logger.info("Let's now run the agent through and get the result for a multiplication:")
    
    
    handler = agent.run("What is 60 multiplied by 95?")
    
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            logger.debug(
                f"Result from calling tool {event.tool_name}:\n\n{event.tool_output}"
            )
        if isinstance(event, ToolCall):
            logger.debug(
                f"Calling tool {event.tool_name} with arguments:\n\n{event.tool_kwargs}"
            )
    
    response = await handler
    logger.success(format_json(response))
    
    logger.debug("Final response")
    logger.debug(response)
    
    """
    Let's also run it with a sum!
    """
    logger.info("Let's also run it with a sum!")
    
    
    handler = agent.run("What is 1234 plus 5678?")
    
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            logger.debug(
                f"Result from calling tool {event.tool_name}:\n\n{event.tool_output}"
            )
        if isinstance(event, ToolCall):
            logger.debug(
                f"Calling tool {event.tool_name} with arguments:\n\n{event.tool_kwargs}"
            )
    
    response = await handler
    logger.success(format_json(response))
    
    logger.debug("Final response")
    logger.debug(response)
    
    """
    If you want more content around Anthropic, make sure to check out our [general example notebook](./anthropic.ipynb)
    """
    logger.info("If you want more content around Anthropic, make sure to check out our [general example notebook](./anthropic.ipynb)")
    
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