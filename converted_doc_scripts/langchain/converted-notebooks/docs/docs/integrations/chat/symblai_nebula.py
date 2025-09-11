from jet.logger import logger
from langchain_community.chat_models.symblai_nebula import ChatNebula
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
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
    sidebar_label: Nebula (Symbl.ai)
    ---
    
    # Nebula (Symbl.ai)
    
    This notebook covers how to get started with [Nebula](https://docs.symbl.ai/docs/nebula-llm) - Symbl.ai's chat model.
    
    ### Integration details
    Head to the [API reference](https://docs.symbl.ai/reference/nebula-chat) for detailed documentation.
    
    ### Model features: TODO
    
    ## Setup
    
    ### Credentials
    To get started, request a [Nebula API key](https://platform.symbl.ai/#/login) and set the `NEBULA_API_KEY` environment variable:
    """
    logger.info("# Nebula (Symbl.ai)")
    
    # import getpass
    
    # os.environ["NEBULA_API_KEY"] = getpass.getpass()
    
    """
    ### Installation
    The integration is set up in the `langchain-community` package.
    
    ## Instantiation
    """
    logger.info("### Installation")
    
    
    chat = ChatNebula(max_tokens=1024, temperature=0.5)
    
    """
    ## Invocation
    """
    logger.info("## Invocation")
    
    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers general knowledge questions."
        ),
        HumanMessage(content="What is the capital of France?"),
    ]
    chat.invoke(messages)
    
    """
    ### Async
    """
    logger.info("### Async")
    
    await chat.ainvoke(messages)
    
    """
    ### Streaming
    """
    logger.info("### Streaming")
    
    for chunk in chat.stream(messages):
        logger.debug(chunk.content, end="", flush=True)
    
    """
    ### Batch
    """
    logger.info("### Batch")
    
    chat.batch([messages])
    
    """
    ## Chaining
    """
    logger.info("## Chaining")
    
    
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    chain = prompt | chat
    
    chain.invoke({"topic": "cows"})
    
    """
    ## API reference
    
    Check out the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.symblai_nebula.ChatNebula.html) for more detail.
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