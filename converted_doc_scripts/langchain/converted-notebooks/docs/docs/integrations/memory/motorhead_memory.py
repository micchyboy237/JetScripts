from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.memory.motorhead_memory import MotorheadMemory
from langchain_core.prompts import PromptTemplate
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
    # Motörhead
    
    >[Motörhead](https://github.com/getmetal/motorhead) is a memory server implemented in Rust. It automatically handles incremental summarization in the background and allows for stateless applications.
    
    ## Setup
    
    See instructions at [Motörhead](https://github.com/getmetal/motorhead) for running the server locally.
    """
    logger.info("# Motörhead")

    """
    ## Example
    """
    logger.info("## Example")

    template = """You are a chatbot having a conversation with a human.
    
    {chat_history}
    Human: {human_input}
    AI:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    memory = MotorheadMemory(
        session_id="testing-1", url="http://localhost:8080", memory_key="chat_history"
    )

    await memory.init()

    llm_chain = LLMChain(
        llm=Ollama(),
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    llm_chain.run("hi im bob")

    llm_chain.run("whats my name?")

    llm_chain.run("whats for dinner?")

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
