from jet.logger import logger
from langchain_community.chat_models.friendli import ChatFriendli
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
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
    sidebar_label: ChatFriendli
    ---
    
    # ChatFriendli
    
    > [Friendli](https://friendli.ai/) enhances AI application performance and optimizes cost savings with scalable, efficient deployment options, tailored for high-demand AI workloads.
    
    This tutorial guides you through integrating `ChatFriendli` for chat applications using LangChain. `ChatFriendli` offers a flexible approach to generating conversational AI responses, supporting both synchronous and asynchronous calls.
    
    ## Setup
    
    Ensure the `langchain_community` and `friendli-client` are installed.
    
    ```sh
    pip install -U langchain-community friendli-client.
    ```
    
    Sign in to [Friendli Suite](https://suite.friendli.ai/) to create a Personal Access Token, and set it as the `FRIENDLI_TOKEN` environment.
    """
    logger.info("# ChatFriendli")
    
    # import getpass
    
    if "FRIENDLI_TOKEN" not in os.environ:
    #     os.environ["FRIENDLI_TOKEN"] = getpass.getpass("Friendi Personal Access Token: ")
    
    """
    You can initialize a Friendli chat model with selecting the model you want to use. The default model is `mixtral-8x7b-instruct-v0-1`. You can check the available models at [docs.friendli.ai](https://docs.periflow.ai/guides/serverless_endpoints/pricing#text-generation-models).
    """
    logger.info("You can initialize a Friendli chat model with selecting the model you want to use. The default model is `mixtral-8x7b-instruct-v0-1`. You can check the available models at [docs.friendli.ai](https://docs.periflow.ai/guides/serverless_endpoints/pricing#text-generation-models).")
    
    
    chat = ChatFriendli(model="meta-llama-3.1-8b-instruct", max_tokens=100, temperature=0)
    
    """
    ## Usage
    
    `FrienliChat` supports all methods of [`ChatModel`](/docs/how_to#chat-models) including async APIs.
    
    You can also use functionality of  `invoke`, `batch`, `generate`, and `stream`.
    """
    logger.info("## Usage")
    
    
    system_message = SystemMessage(content="Answer questions as short as you can.")
    human_message = HumanMessage(content="Tell me a joke.")
    messages = [system_message, human_message]
    
    chat.invoke(messages)
    
    chat.batch([messages, messages])
    
    chat.generate([messages, messages])
    
    for chunk in chat.stream(messages):
        logger.debug(chunk.content, end="", flush=True)
    
    """
    You can also use all functionality of async APIs: `ainvoke`, `abatch`, `agenerate`, and `astream`.
    """
    logger.info("You can also use all functionality of async APIs: `ainvoke`, `abatch`, `agenerate`, and `astream`.")
    
    await chat.ainvoke(messages)
    
    await chat.abatch([messages, messages])
    
    await chat.agenerate([messages, messages])
    
    for chunk in chat.stream(messages):
        logger.debug(chunk.content, end="", flush=True)
    
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