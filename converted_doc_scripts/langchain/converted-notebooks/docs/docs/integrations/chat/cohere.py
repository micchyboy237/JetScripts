from jet.logger import logger
from langchain_cohere import ChatCohere
from langchain_core.messages import (
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import os
import shutil

async def main():
    HumanMessage,
    ToolMessage,
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
    ---
    sidebar_label: Cohere
    ---
    
    # Cohere
    
    This notebook covers how to get started with [Cohere chat models](https://cohere.com/chat).
    
    Head to the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.cohere.ChatCohere.html) for detailed documentation of all attributes and methods.
    
    ## Setup
    
    The integration lives in the `langchain-cohere` package. We can install these with:
    
    ```bash
    pip install -U langchain-cohere
    ```
    
    We'll also need to get a [Cohere API key](https://cohere.com/) and set the `COHERE_API_KEY` environment variable:
    """
    logger.info("# Cohere")
    
    # import getpass
    
    # os.environ["COHERE_API_KEY"] = getpass.getpass()
    
    """
    It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability
    """
    logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability")
    
    
    
    """
    ## Usage
    
    ChatCohere supports all [ChatModel](/docs/how_to#chat-models) functionality:
    """
    logger.info("## Usage")
    
    
    chat = ChatCohere()
    
    messages = [HumanMessage(content="1"), HumanMessage(content="2 3")]
    chat.invoke(messages)
    
    await chat.ainvoke(messages)
    
    for chunk in chat.stream(messages):
        logger.debug(chunk.content, end="", flush=True)
    
    chat.batch([messages])
    
    """
    ## Chaining
    
    You can also easily combine with a prompt template for easy structuring of user input. We can do this using [LCEL](/docs/concepts/lcel)
    """
    logger.info("## Chaining")
    
    
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    chain = prompt | chat
    
    chain.invoke({"topic": "bears"})
    
    """
    ## Tool calling
    
    Cohere supports tool calling functionalities!
    """
    logger.info("## Tool calling")
    
    
    @tool
    def magic_function(number: int) -> int:
        """Applies a magic operation to an integer
        Args:
            number: Number to have magic operation performed on
        """
        return number + 10
    
    
    def invoke_tools(tool_calls, messages):
        for tool_call in tool_calls:
            selected_tool = {"magic_function": magic_function}[tool_call["name"].lower()]
            tool_output = selected_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
        return messages
    
    
    tools = [magic_function]
    
    llm_with_tools = chat.bind_tools(tools=tools)
    messages = [HumanMessage(content="What is the value of magic_function(2)?")]
    
    res = llm_with_tools.invoke(messages)
    while res.tool_calls:
        messages.append(res)
        messages = invoke_tools(res.tool_calls, messages)
        res = llm_with_tools.invoke(messages)
    
    res
    
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