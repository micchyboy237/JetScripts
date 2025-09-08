async def main():
    from jet.transformers.formatters import format_json
    from IPython.display import display, HTML
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.ui import Console
    from autogen_core import CancellationToken
    from autogen_core.models import UserMessage
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    from azure.core.credentials import AzureKeyCredential
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
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
    # AutoGen Basic Sample 
    
    In this code sample, you will use the [AutoGen](https://aka.ms/ai-agents/autogen) AI Framework to create a basic agent. 
    
    The goal of this sample is to show you the steps that we will later use in the additional code samples when implementing the different agentic patterns.
    
    ## Import the Needed Python Packages
    """
    logger.info("# AutoGen Basic Sample")

    """
    ## Create the Client 
    
    In this sample, we will use [GitHub Models](https://aka.ms/ai-agents-beginners/github-models) for access to the LLM. 
    
    The `model` is defined as `llama3.2`. Try changing the model to another model available on the GitHub Models marketplace to see the different results. 
    
    As a quick test, we will just run a simple prompt - `What is the capital of France`.
    """
    logger.info("## Create the Client")

    load_dotenv()
    client = OllamaChatCompletionClient(model="llama3.2")

    result = await client.create([UserMessage(content="What is the capital of France?", source="user")])
    logger.success(format_json(result))
    logger.debug(result)

    """
    ## Defining the Agent 
    
    Now that we have set up the `client` and confirmed that it is working, let us create an `AssistantAgent`. Each agent can be assigned a: 
    **name** - A short hand name that will be useful in referencing it in multi-agent flows. 
    **model_client** - The client that you created in the earlier step. 
    **tools** - Available tools that the Agent can use to complete a task.
    **system_message** - The metaprompt that defines the task, behavior and tone of the LLM. 
    
    You can change the system message to see how the LLM responds. We will cover `tools` in Lesson #4.
    """
    logger.info("## Defining the Agent")

    agent = AssistantAgent(
        name="assistant",
        model_client=client,
        tools=[],
        system_message="You are a travel agent that plans great vacations",
    )

    """
    ## Run the Agent 
    
    The below function will run the agent. We use the the `on_message` method to update the Agent's state with the new message. 
    
    In this case, we update the state with a new message from the user which is `"Plan me a great sunny vacation"`.
    
    You can change the message content to see how the LLM responds differently.
    """
    logger.info("## Run the Agent")

    async def assistant_run():
        user_query = "Plan me a great sunny vacation"

        html_output = "<div style='margin-bottom:10px'>"
        html_output += "<div style='font-weight:bold'>User:</div>"
        html_output += f"<div style='margin-left:20px'>{user_query}</div>"
        html_output += "</div>"

        response = await agent.on_messages(
            [TextMessage(content=user_query, source="user")],
            cancellation_token=CancellationToken(),
        )
        logger.success(format_json(response))

        html_output += "<div style='margin-bottom:20px'>"
        html_output += "<div style='font-weight:bold'>Assistant:</div>"
        html_output += f"<div style='margin-left:20px; white-space:pre-wrap'>{response.chat_message.content}</div>"
        html_output += "</div>"

        display(HTML(html_output))

    await assistant_run()

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
