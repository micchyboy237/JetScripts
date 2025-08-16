import asyncio
from jet.transformers.formatters import format_json
from dotenv import load_dotenv
from jet.logger import CustomLogger
from openai import AsyncOllama
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OllamaChatCompletion
from semantic_kernel.functions import kernel_function
from typing import Annotated
import os
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Semantic Kernel 

In this code sample, you will use the [Semantic Kernel](https://aka.ms/ai-agents-beginners/semantic-kernel) AI Framework to create a basic agent. 

The goal of this sample is to show you the steps that we will later use in the additional code samples when implementing the different agentic patterns.

## Import the Needed Python Packages
"""
logger.info("# Semantic Kernel")




"""
## Creating the Client

In this sample, we will use [GitHub Models](https://aka.ms/ai-agents-beginners/github-models) for access to the LLM. 

The `ai_model_id` is defined as `llama3.1`. Try changing the model to another model available on the GitHub Models marketplace to see the different results. 

For us to use the `Azure Inference SDK` that is used for the `base_url` for GitHub Models, we will use the `OllamaChatCompletion` connector within Semantic Kernel. There are also other [available connectors](https://learn.microsoft.com/semantic-kernel/concepts/ai-services/chat-completion) to use Semantic Kernel for other model providers.
"""
logger.info("## Creating the Client")



class DestinationsPlugin:
    """A List of Random Destinations for a vacation."""

    def __init__(self):
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia"
        ]
        self.last_destination = None

    @kernel_function(description="Provides a random vacation destination.")
    def get_random_destination(self) -> Annotated[str, "Returns a random vacation destination."]:
        available_destinations = self.destinations.copy()
        if self.last_destination and len(available_destinations) > 1:
            available_destinations.remove(self.last_destination)

        destination = random.choice(available_destinations)

        self.last_destination = destination

        return destination

load_dotenv()
client = AsyncOllama(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/",
)

chat_completion_service = OllamaChatCompletion(
    ai_model_id="llama3.1",
    async_client=client,
)

"""
## Creating the Agent 

Below we create the Agent called `TravelAgent`.

For this example, we are using very simple instructions. You can change these instructions to see how the agent responds differently.
"""
logger.info("## Creating the Agent")

agent = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[DestinationsPlugin()],
    name="TravelAgent",
    instructions="You are a helpful AI Agent that can help plan vacations for customers at random destinations",
)

"""
## Running the Agent

Now we can run the Agent by defining a thread of type `ChatHistoryAgentThread`.  Any required system messages are provided to the agent's invoke_stream `messages` keyword argument.

After these are defined, we create a `user_inputs` that will be what the user is sending to the agent. In this case, we have set this message to `Plan me a sunny vacation`. 

Feel free to change this message to see how the agent responds differently.
"""
logger.info("## Running the Agent")

async def main():
    thread: ChatHistoryAgentThread | None = None

    user_inputs = [
        "Plan me a day trip.",
    ]

    for user_input in user_inputs:
        logger.debug(f"# User: {user_input}\n")
        first_chunk = True
        async for response in agent.invoke_stream(
            messages=user_input, thread=thread,
        ):
            if first_chunk:
                logger.debug(f"# {response.name}: ", end="", flush=True)
                first_chunk = False
            logger.debug(f"{response}", end="", flush=True)
            thread = response.thread
        logger.debug()

    async def run_async_code_8521c805():
        await thread.delete() if thread else None
        return 
     = asyncio.run(run_async_code_8521c805())
    logger.success(format_json())

async def run_async_code_ba09313d():
    await main()
    return 
 = asyncio.run(run_async_code_ba09313d())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)