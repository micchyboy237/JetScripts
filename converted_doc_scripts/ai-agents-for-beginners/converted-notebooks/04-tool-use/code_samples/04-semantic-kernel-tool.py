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
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Semantic Kernel Tool Use Example
    
    ## Import the Needed Packages
    """
    logger.info("# Semantic Kernel Tool Use Example")
    
    
    
    
    
    
    """
    ## Creating the Plugins    
    Semantic Kernel uses plugins as tools that can be called by the agent. A plugin can have multiple `kernel_functions` in it as a group. 
    
    In the example below, we create a `DestinationsPlugin` that has two functions: 
    1. Provides a list of destinations using the `get_destinations` function
    2. Provides a list of availability for each destination using the `get_availabilty` function,
    """
    logger.info("## Creating the Plugins")
    
    class DestinationsPlugin:
        """A List of Destinations for vacation."""
    
        @kernel_function(description="Provides a list of vacation destinations.")
        def get_destinations(self) -> Annotated[str, "Returns the vacation destinations."]:
            return """
            Barcelona, Spain
            Paris, France
            Berlin, Germany
            Tokyo, Japan
            New York, USA
            """
    
        @kernel_function(description="Provides the availability of a destination.")
        def get_availability(
            self, destination: Annotated[str, "The destination to check availability for."]
        ) -> Annotated[str, "Returns the availability of the destination."]:
            return """
            Barcelona - Unavailable
            Paris - Available
            Berlin - Available
            Tokyo - Unavailable
            New York - Available
            """
    
    """
    ## Creating the Client
    
    In this sample, we will use [GitHub Models](https://aka.ms/ai-agents-beginners/github-models) for access to the LLM. 
    
    The `ai_model_id` is defined as `llama3.2`. Try changing the model to another model available on the GitHub Models marketplace to see the different results. 
    
    For us to use the `Azure Inference SDK` that is used for the `base_url` for GitHub Models, we will use the `OllamaChatCompletion` connector within Semantic Kernel. There are also other [available connectors](https://learn.microsoft.com/semantic-kernel/concepts/ai-services/chat-completion) to use Semantic Kernel for other model providers.
    """
    logger.info("## Creating the Client")
    
    load_dotenv()
    client = AsyncOllama(
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com/",
    )
    
    chat_completion_service = OllamaChatCompletion(
        ai_model_id="llama3.2",
        async_client=client,
    )
    
    """
    ## Creating the Agent 
    Now we will create the Agent by using the Agent Name and Instructions that we can set. 
    
    You can change these settings to see how the differences in the agent's response.
    """
    logger.info("## Creating the Agent")
    
    agent = ChatCompletionAgent(
        service=chat_completion_service,
        name="TravelAgent",
        instructions="Answer questions about the travel destinations and their availability.",
        plugins=[DestinationsPlugin()],
    )
    
    """
    ## Running the Agent 
    
    Now we wil run the AI Agent. In this snippet, we can add two messages to the `user_input` to show how the agent responds to followup questions. 
    
    The agent should call the correct function to get the list of available destinations and confirm the availability of a certain location. 
    
    You can change the `user_inputs` to see how the agent responds.
    """
    logger.info("## Running the Agent")
    
    user_inputs = [
        "What destinations are available?",
        "Is Barcelona available?",
        "Are there any vacation destinations available not in Europe?",
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
    
                            function_calls.append(f"Calling function: {current_function_name}({formatted_args})")
                            current_function_name = None
                            argument_buffer = ""
    
                        function_calls.append(f"\nFunction Result:\n\n{item.result}")
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