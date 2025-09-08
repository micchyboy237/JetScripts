async def main():
    from IPython.display import display, HTML
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from openai import AsyncOllama
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
    from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent
    from semantic_kernel.functions import kernel_function
    from typing import Annotated
    import json
    import os
    import random
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"

    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")

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
    
    The `ai_model_id` is defined as `llama3.2`. Try changing the model to another model available on the GitHub Models marketplace to see the different results. 
    
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

    chat_completion_service = OllamaChatCompletion(ai_model_id="llama3.2")

    """
    ## Creating the Agent 
    
    Below we create the Agent called `TravelAgent`.
    
    For this example, we are using very simple instructions. You can change these instructions to see how the agent responds differently.
    """
    logger.info("## Creating the Agent")

    AGENT_INSTRUCTIONS = """You are a helpful AI Agent that can help plan vacations for customers.
    
    Important: When users specify a destination, always plan for that location. Only suggest random destinations when the user hasn't specified a preference.
    
    When the conversation begins, introduce yourself with this message:
    "Hello! I'm your TravelAgent assistant. I can help plan vacations and suggest interesting destinations for you. Here are some things you can ask me:
    1. Plan a day trip to a specific location
    2. Suggest a random vacation destination
    3. Find destinations with specific features (beaches, mountains, historical sites, etc.)
    4. Plan an alternative trip if you don't like my first suggestion
    
    What kind of trip would you like me to help you plan today?"
    
    Always prioritize user preferences. If they mention a specific destination like "Bali" or "Paris," focus your planning on that location rather than suggesting alternatives.
    """

    agent = ChatCompletionAgent(
        service=chat_completion_service,
        plugins=[DestinationsPlugin()],
        name="TravelAgent",
        instructions=AGENT_INSTRUCTIONS,
    )

    """
    ## Running the Agents 
    
    Now we can run the Agent by defining the `ChatHistory` and adding the `system_message` to it. We will use the `AGENT_INSTRUCTIONS` that we defined earlier. 
    
    After these are defined, we create a `user_inputs` that will be what the user is sending to the agent. In this case, we have set this message to `Plan me a sunny vacation`. 
    
    Feel free to change this message to see how the agent responds differently.
    """
    logger.info("## Running the Agents")

    user_inputs = [
        "Plan me a day trip.",
        "I don't like that destination. Plan me another vacation.",
    ]

    async def main():
        thread: ChatHistoryAgentThread | None = None

        for user_input in user_inputs:
            html_output = (
                f"<div style='margin-bottom:10px'>"
                f"<div style='font-weight:bold'>User:</div>"
                f"<div style='margin-left:20px'>{user_input}</div></div>"
            )

            agent_name = None
            full_response: list[str] = []
            function_calls: list[str] = []

            current_function_name = None
            argument_buffer = ""

            async for response in agent.invoke_stream(
                messages=user_input,
                thread=thread,
            ):
                thread = response.thread
                agent_name = response.name
                content_items = list(response.items)

                for item in content_items:
                    if isinstance(item, FunctionCallContent):
                        if item.function_name:
                            current_function_name = item.function_name

                        if isinstance(item.arguments, str):
                            argument_buffer += item.arguments
                    elif isinstance(item, FunctionResultContent):
                        if current_function_name:
                            formatted_args = argument_buffer.strip()
                            try:
                                parsed_args = json.loads(formatted_args)
                                formatted_args = json.dumps(parsed_args)
                            except Exception:
                                pass  # leave as raw string

                            function_calls.append(
                                f"Calling function: {current_function_name}({formatted_args})")
                            current_function_name = None
                            argument_buffer = ""

                        function_calls.append(
                            f"\nFunction Result:\n\n{item.result}")
                    elif isinstance(item, StreamingTextContent) and item.text:
                        full_response.append(item.text)

            if function_calls:
                html_output += (
                    "<div style='margin-bottom:10px'>"
                    "<details>"
                    "<summary style='cursor:pointer; font-weight:bold; color:#0066cc;'>Function Calls (click to expand)</summary>"
                    "<div style='margin:10px; padding:10px; background-color:#f8f8f8; "
                    "border:1px solid #ddd; border-radius:4px; white-space:pre-wrap; font-size:14px; color:#333;'>"
                    f"{chr(10).join(function_calls)}"
                    "</div></details></div>"
                )

            html_output += (
                "<div style='margin-bottom:20px'>"
                f"<div style='font-weight:bold'>{agent_name or 'Assistant'}:</div>"
                f"<div style='margin-left:20px; white-space:pre-wrap'>{''.join(full_response)}</div></div><hr>"
            )

            display(HTML(html_output))

    await main()

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
