from jet.transformers.formatters import format_json
from dotenv import find_dotenv, load_dotenv
from jet.logger import logger
from langchain_community.chat_models import ChatDeepInfra
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel
import asyncio
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
    # DeepInfra
    
    [DeepInfra](https://deepinfra.com/?utm_source=langchain) is a serverless inference as a service that provides access to a [variety of LLMs](https://deepinfra.com/models?utm_source=langchain) and [embeddings models](https://deepinfra.com/models?type=embeddings&utm_source=langchain). This notebook goes over how to use LangChain with DeepInfra for chat models.
    
    ## Set the Environment API Key
    Make sure to get your API key from DeepInfra. You have to [Login](https://deepinfra.com/login?from=%2Fdash) and get a new token.
    
    You are given a 1 hour free of serverless GPU compute to test different models. (see [here](https://github.com/deepinfra/deepctl#deepctl))
    You can print your token with `deepctl auth token`
    """
    logger.info("# DeepInfra")
    
    # from getpass import getpass
    
    
    # DEEPINFRA_API_TOKEN = getpass()
    
    os.environ["DEEPINFRA_API_TOKEN"] = DEEPINFRA_API_TOKEN
    
    chat = ChatDeepInfra(model="meta-llama/Llama-2-7b-chat-hf")
    
    messages = [
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        )
    ]
    chat.invoke(messages)
    
    """
    ## `ChatDeepInfra` also supports async and streaming functionality:
    """
    logger.info("## `ChatDeepInfra` also supports async and streaming functionality:")
    
    
    await chat.agenerate([messages])
    
    chat = ChatDeepInfra(
        streaming=True,
        verbose=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    chat.invoke(messages)
    
    """
    # Tool Calling
    
    DeepInfra currently supports only invoke and async invoke tool calling.
    
    For a complete list of models that support tool calling, please refer to our [tool calling documentation](https://deepinfra.com/docs/advanced/function_calling).
    """
    logger.info("# Tool Calling")
    
    
    
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    
    _ = load_dotenv(find_dotenv())
    
    
    @tool
    def foo(something):
        """
        Called when foo
        """
        pass
    
    
    class Bar(BaseModel):
        """
        Called when Bar
        """
    
        pass
    
    
    llm = ChatDeepInfra(model=model_name)
    tools = [foo, Bar]
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        HumanMessage("Foo and bar, please."),
    ]
    
    response = llm_with_tools.invoke(messages)
    logger.debug(response.tool_calls)
    
    
    async def call_ainvoke():
        result = await llm_with_tools.ainvoke(messages)
        logger.success(format_json(result))
        logger.debug(result.tool_calls)
    
    
    asyncio.run(call_ainvoke())
    
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