from jet.transformers.formatters import format_json
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
import ChatModelTabs from "@theme/ChatModelTabs";
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
    # How to stream events from a tool
    
    :::info Prerequisites
    
    This guide assumes familiarity with the following concepts:
    - [LangChain Tools](/docs/concepts/tools)
    - [Custom tools](/docs/how_to/custom_tools)
    - [Using stream events](/docs/how_to/streaming/#using-stream-events)
    - [Accessing RunnableConfig within a custom tool](/docs/how_to/tool_configure/)
    
    :::
    
    If you have [tools](/docs/concepts/tools/) that call [chat models](/docs/concepts/chat_models/), [retrievers](/docs/concepts/retrievers/), or other [runnables](/docs/concepts/runnables/), you may want to access [internal events](https://python.langchain.com/docs/how_to/streaming/#event-reference) from those runnables or configure them with additional properties. This guide shows you how to manually pass parameters properly so that you can do this using the `astream_events()` method.
    
    :::caution Compatibility
    
    LangChain cannot automatically propagate configuration, including callbacks necessary for `astream_events()`, to child runnables if you are running `async` code in `python<=3.10`. This is a common reason why you may fail to see events being emitted from custom runnables or tools.
    
    If you are running `python<=3.10`, you will need to manually propagate the `RunnableConfig` object to the child runnable in async environments. For an example of how to manually propagate the config, see the implementation of the `bar` RunnableLambda below.
    
    If you are running `python>=3.11`, the `RunnableConfig` will automatically propagate to child runnables in async environment. However, it is still a good idea to propagate the `RunnableConfig` manually if your code may run in older Python versions.
    
    This guide also requires `langchain-core>=0.2.16`.
    :::
    
    Say you have a custom tool that calls a chain that condenses its input by prompting a chat model to return only 10 words, then reversing the output. First, define it in a naive way:
    
    
    <ChatModelTabs customVarName="model" />
    """
    logger.info("# How to stream events from a tool")
    
    # from getpass import getpass
    
    
    # if "ANTHROPIC_API_KEY" not in os.environ:
    #     os.environ["ANTHROPIC_API_KEY"] = getpass()
    
    model = ChatOllama(model="llama3.2")
    
    
    
    @tool
    async def special_summarization_tool(long_text: str) -> str:
        """A tool that summarizes input text using advanced techniques."""
        prompt = ChatPromptTemplate.from_template(
            "You are an expert writer. Summarize the following text in 10 words or less:\n\n{long_text}"
        )
    
        def reverse(x: str):
            return x[::-1]
    
        chain = prompt | model | StrOutputParser() | reverse
        summary = await chain.ainvoke({"long_text": long_text})
        logger.success(format_json(summary))
        return summary
    
    """
    Invoking the tool directly works just fine:
    """
    logger.info("Invoking the tool directly works just fine:")
    
    LONG_TEXT = """
    NARRATOR:
    (Black screen with text; The sound of buzzing bees can be heard)
    According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible.
    BARRY BENSON:
    (Barry is picking out a shirt)
    Yellow, black. Yellow, black. Yellow, black. Yellow, black. Ooh, black and yellow! Let's shake it up a little.
    JANET BENSON:
    Barry! Breakfast is ready!
    BARRY:
    Coming! Hang on a second.
    """
    
    await special_summarization_tool.ainvoke({"long_text": LONG_TEXT})
    
    """
    But if you wanted to access the raw output from the chat model rather than the full tool, you might try to use the [`astream_events()`](/docs/how_to/streaming/#using-stream-events) method and look for an `on_chat_model_end` event. Here's what happens:
    """
    logger.info("But if you wanted to access the raw output from the chat model rather than the full tool, you might try to use the [`astream_events()`](/docs/how_to/streaming/#using-stream-events) method and look for an `on_chat_model_end` event. Here's what happens:")
    
    stream = special_summarization_tool.astream_events({"long_text": LONG_TEXT})
    
    async for event in stream:
        if event["event"] == "on_chat_model_end":
            logger.debug(event)
    
    """
    You'll notice (unless you're running through this guide in `python>=3.11`) that there are no chat model events emitted from the child run!
    
    This is because the example above does not pass the tool's config object into the internal chain. To fix this, redefine your tool to take a special parameter typed as `RunnableConfig` (see [this guide](/docs/how_to/tool_configure) for more details). You'll also need to pass that parameter through into the internal chain when executing it:
    """
    logger.info("You'll notice (unless you're running through this guide in `python>=3.11`) that there are no chat model events emitted from the child run!")
    
    
    
    @tool
    async def special_summarization_tool_with_config(
        long_text: str, config: RunnableConfig
    ) -> str:
        """A tool that summarizes input text using advanced techniques."""
        prompt = ChatPromptTemplate.from_template(
            "You are an expert writer. Summarize the following text in 10 words or less:\n\n{long_text}"
        )
    
        def reverse(x: str):
            return x[::-1]
    
        chain = prompt | model | StrOutputParser() | reverse
        summary = await chain.ainvoke({"long_text": long_text}, config=config)
        logger.success(format_json(summary))
        return summary
    
    """
    And now try the same `astream_events()` call as before with your new tool:
    """
    logger.info("And now try the same `astream_events()` call as before with your new tool:")
    
    stream = special_summarization_tool_with_config.astream_events({"long_text": LONG_TEXT})
    
    async for event in stream:
        if event["event"] == "on_chat_model_end":
            logger.debug(event)
    
    """
    Awesome! This time there's an event emitted.
    
    For streaming, `astream_events()` automatically calls internal runnables in a chain with streaming enabled if possible, so if you wanted to a stream of tokens as they are generated from the chat model, you could simply filter to look for `on_chat_model_stream` events with no other changes:
    """
    logger.info("Awesome! This time there's an event emitted.")
    
    stream = special_summarization_tool_with_config.astream_events({"long_text": LONG_TEXT})
    
    async for event in stream:
        if event["event"] == "on_chat_model_stream":
            logger.debug(event)
    
    """
    ## Next steps
    
    You've now seen how to stream events from within a tool. Next, check out the following guides for more on using tools:
    
    - Pass [runtime values to tools](/docs/how_to/tool_runtime)
    - Pass [tool results back to a model](/docs/how_to/tool_results_pass_to_model)
    - [Dispatch custom callback events](/docs/how_to/callbacks_custom_events)
    
    You can also check out some more specific uses of tool calling:
    
    - Building [tool-using chains and agents](/docs/how_to#tools)
    - Getting [structured outputs](/docs/how_to/structured_output/) from models
    """
    logger.info("## Next steps")
    
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