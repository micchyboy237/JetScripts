from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List
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
    # How to use callbacks in async environments
    
    :::info Prerequisites
    
    This guide assumes familiarity with the following concepts:
    
    - [Callbacks](/docs/concepts/callbacks)
    - [Custom callback handlers](/docs/how_to/custom_callbacks)
    :::
    
    If you are planning to use the async APIs, it is recommended to use and extend [`AsyncCallbackHandler`](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.AsyncCallbackHandler.html) to avoid blocking the event.
    
    
    :::warning
    If you use a sync `CallbackHandler` while using an async method to run your LLM / Chain / Tool / Agent, it will still work. However, under the hood, it will be called with [`run_in_executor`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor) which can cause issues if your `CallbackHandler` is not thread-safe.
    :::
    
    :::danger
    
    If you're on `python<=3.10`, you need to remember to propagate `config` or `callbacks` when invoking other `runnable` from within a `RunnableLambda`, `RunnableGenerator` or `@tool`. If you do not do this,
    the callbacks will not be propagated to the child runnables being invoked.
    :::
    """
    logger.info("# How to use callbacks in async environments")
    
    # %pip install -qU langchain jet.adapters.langchain.chat_ollama
    
    # import getpass
    
    # os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
    
    
    
    
    class MyCustomSyncHandler(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            logger.debug(f"Sync handler being called in a `thread_pool_executor`: token: {token}")
    
    
    class MyCustomAsyncHandler(AsyncCallbackHandler):
        """Async callback handler that can be used to handle callbacks from langchain."""
    
        async def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
        ) -> None:
            """Run when chain starts running."""
            logger.debug("zzzz....")
            await asyncio.sleep(0.3)
            class_name = serialized["name"]
            logger.debug("Hi! I just woke up. Your llm is starting")
    
        async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
            """Run when chain ends running."""
            logger.debug("zzzz....")
            await asyncio.sleep(0.3)
            logger.debug("Hi! I just woke up. Your llm is ending")
    
    
    chat = ChatOllama(
        model="claude-3-7-sonnet-20250219",
        max_tokens=25,
        streaming=True,
        callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
    )
    
    await chat.agenerate([[HumanMessage(content="Tell me a joke")]])
    
    """
    ## Next steps
    
    You've now learned how to create your own custom callback handlers.
    
    Next, check out the other how-to guides in this section, such as [how to attach callbacks to a runnable](/docs/how_to/callbacks_attach).
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