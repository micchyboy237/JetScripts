from jet.logger import logger
from langchain_community.llms.friendli import Friendli
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
    sidebar_label: Friendli
    ---
    
    # Friendli
    
    > [Friendli](https://friendli.ai/) enhances AI application performance and optimizes cost savings with scalable, efficient deployment options, tailored for high-demand AI workloads.
    
    This tutorial guides you through integrating `Friendli` with LangChain.
    
    ## Setup
    
    Ensure the `langchain_community` and `friendli-client` are installed.
    
    ```sh
    pip install -U langchain-community friendli-client
    ```
    
    Sign in to [Friendli Suite](https://suite.friendli.ai/) to create a Personal Access Token, and set it as the `FRIENDLI_TOKEN` environment.
    """
    logger.info("# Friendli")
    
    # import getpass
    
    if "FRIENDLI_TOKEN" not in os.environ:
    #     os.environ["FRIENDLI_TOKEN"] = getpass.getpass("Friendi Personal Access Token: ")
    
    """
    You can initialize a Friendli chat model with selecting the model you want to use.  
    The default model is `meta-llama-3.1-8b-instruct`. You can check the available models at [friendli.ai/docs](https://friendli.ai/docs/guides/serverless_endpoints/pricing#text-generation-models).
    """
    logger.info("You can initialize a Friendli chat model with selecting the model you want to use.")
    
    
    llm = Friendli(model="meta-llama-3.1-8b-instruct", max_tokens=100, temperature=0)
    
    """
    ## Usage
    
    `Frienli` supports all methods of [`LLM`](/docs/how_to#llms) including async APIs.
    
    You can use functionality of `invoke`, `batch`, `generate`, and `stream`.
    """
    logger.info("## Usage")
    
    llm.invoke("Tell me a joke.")
    
    llm.batch(["Tell me a joke.", "Tell me a joke."])
    
    llm.generate(["Tell me a joke.", "Tell me a joke."])
    
    for chunk in llm.stream("Tell me a joke."):
        logger.debug(chunk, end="", flush=True)
    
    """
    You can also use all functionality of async APIs: `ainvoke`, `abatch`, `agenerate`, and `astream`.
    """
    logger.info("You can also use all functionality of async APIs: `ainvoke`, `abatch`, `agenerate`, and `astream`.")
    
    await llm.ainvoke("Tell me a joke.")
    
    await llm.abatch(["Tell me a joke.", "Tell me a joke."])
    
    await llm.agenerate(["Tell me a joke.", "Tell me a joke."])
    
    for chunk in llm.stream("Tell me a joke."):
        logger.debug(chunk, end="", flush=True)
    
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