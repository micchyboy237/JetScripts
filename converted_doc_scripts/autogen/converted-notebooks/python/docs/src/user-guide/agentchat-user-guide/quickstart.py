from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from jet.logger import logger
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
    # Quickstart
    
    Via AgentChat, you can build applications quickly using preset agents.
    To illustrate this, we will begin with creating a single agent that can
    use tools.
    
    First, we need to install the AgentChat and Extension packages.
    """
    logger.info("# Quickstart")

    # pip install -U "autogen-agentchat" "autogen-ext[ollama,azure]"

    """
    This example uses an Ollama model, however, you can use other models as well.
    Simply update the `model_client` with the desired model or model client class.
    
    To use Azure Ollama models and AAD authentication,
    you can follow the instructions [here](./tutorial/models.ipynb#azure-ollama).
    To use other models, see [Models](./tutorial/models.ipynb).
    """
    logger.info(
        "This example uses an Ollama model, however, you can use other models as well.")

    model_client = OllamaChatCompletionClient(
        model="llama3.2",
    )

    async def get_weather(city: str) -> str:
        """Get the weather for a given city."""
        return f"The weather in {city} is 73 degrees and Sunny."

    agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        tools=[get_weather],
        system_message="You are a helpful assistant.",
        reflect_on_tool_use=True,
        # Enable streaming tokens from the model client.
        model_client_stream=True,
    )

    async def main() -> None:
        await Console(agent.run_stream(task="What is the weather in New York?"))
        await model_client.close()

    await main()

    """
    ## What's Next?
    
    Now that you have a basic understanding of how to use a single agent, consider following the [tutorial](./tutorial/index.md) for a walkthrough on other features of AgentChat.
    """
    logger.info("## What's Next?")

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
