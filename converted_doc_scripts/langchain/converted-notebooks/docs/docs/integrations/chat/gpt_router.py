from jet.logger import logger
from langchain_community.chat_models import GPTRouter
from langchain_community.chat_models.gpt_router import GPTRouterModel
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
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
    ---
    sidebar_label: GPTRouter
    ---
    
    # GPTRouter
    
    [GPTRouter](https://github.com/Writesonic/GPTRouter) is an open source LLM API Gateway that offers a universal API for 30+ LLMs, vision, and image models, with smart fallbacks based on uptime and latency, automatic retries, and streaming.
    
     
    This notebook covers how to get started with using Langchain + the GPTRouter I/O library. 
    
    * Set `GPT_ROUTER_API_KEY` environment variable
    * or use the `gpt_router_api_key` keyword argument
    """
    logger.info("# GPTRouter")
    
    # %pip install --upgrade --quiet  GPTRouter
    
    
    anthropic_claude = GPTRouterModel(name="claude-instant-1.2", provider_name="anthropic")
    
    chat = GPTRouter(models_priority_list=[anthropic_claude])
    
    messages = [
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        )
    ]
    chat(messages)
    
    """
    ## `GPTRouter` also supports async and streaming functionality:
    """
    logger.info("## `GPTRouter` also supports async and streaming functionality:")
    
    
    await chat.agenerate([messages])
    
    chat = GPTRouter(
        models_priority_list=[anthropic_claude],
        streaming=True,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    chat(messages)
    
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