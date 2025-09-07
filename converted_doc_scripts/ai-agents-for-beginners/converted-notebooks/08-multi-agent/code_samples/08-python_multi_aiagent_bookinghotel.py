async def main():
    from jet.transformers.formatters import format_json
    from azure.ai.projects.models import CodeInterpreterTool
    from azure.identity.aio import DefaultAzureCredential
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from semantic_kernel.agents import AgentGroupChat
    from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings
    from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
    from semantic_kernel.contents import AuthorRole
    from semantic_kernel.contents import ChatMessageContent
    from semantic_kernel.functions.kernel_function_decorator import kernel_function
    from typing import Annotated
    from typing import Annotated: This imports the Annotated type from the typing module. Annotated is used to add metadata to type hints, which can be useful for various purposes such as validation, documentation, or tooling
    import asyncio,os
    import asyncio: This imports the asyncio module, which provides support for asynchronous programming in Python. It allows you to write concurrent code using the async and await syntax.
    import os
    import requests
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    ## **Samples: Multi-AI Agents for Booking hotel**
    
    In today's fast-paced world, planning a business trip involves more than just booking a flight and a hotel room. It requires a level of coordination and efficiency that can be challenging to achieve. This is where Multi-AI Agents come into play, revolutionizing the way we manage our travel needs.
    
    Imagine having a team of intelligent agents at your disposal, working together to handle every aspect of your trip with precision and ease. With our advanced AI technology, we have created specialized agents for booking services and itinerary arrangement, ensuring a seamless and stress-free travel experience. 
    
    This is a basic scenario. When planning a business trip, we need to consult with a business travel agent to obtain air ticket information, hotel information, etc. Through AI Agents, we can build agents for booking services and agents for itinerary arrangement to collaborate and improve the level of intelligence.
    
    # Initialize the Azure AI Agent Service and get configuration information from **.env**
    
    ### **.env** 
    
    Create a .env file 
    
    **.env** contains the connection string of Azure AI Agent Service, the model used by AOAI, and the corresponding Google API Search service API, ENDPOINT, etc.
    
    - **AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME** = "Your Azure AI Agent Service Model Deployment Name"
    
    [**NOTE**] You will need a model with 100,000 Rate Limit (Tokens per minute)  Rate Limit of 600 (Request per minute)
    
      You can get model in Azure AI Foundry - Model and Endpoint. 
    
    
    - **AZURE_AI_AGENT_PROJECT_CONNECTION_STRING** = "Your Azure AI Agent Service Project Connection String"
    
      You can get the project connection string in your project overview in  AI ​​Foundry Portal Screen.
    
    - **SERPAPI_SEARCH_API_KEY** = "Your SERPAPI Search API KEY"
    - **SERPAPI_SEARCH_ENDPOINT** = "Your SERPAPI Search Endpoint"
    
    To get the Model Deployment Name and Project Connection String of Azure AI Agent Service, you need to create Azure AI Agent Service. It is recommended to use [this template](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Ffosteramanda%2Fazure-agent-quickstart-templates%2Frefs%2Fheads%2Fmaster%2Fquickstarts%2Fmicrosoft.azure-ai-agent-service%2Fstandard-agent%2Fazuredeploy.json) to create it directly （***Note:*** Azure AI Agent Service is currently set in a limited region. It is recommended that you refer to [this link](https://learn.microsoft.com/en-us/azure/ai-services/agents/concepts/model-region-support) to set the region)
    
    Agent needs to access SERPAPI. It is recommended to register using [this link](https://serpapi.com/searches). After registration, you can obtain a unique API KEY and ENDPOINT
    
    # Login to Azure 
    
    You Now need to login into Azure Open a terminal in VScode and run the `az login` command
    
    # Setup 
    
    To run this notebook, you will need to install the following libraries. Here is a list of the required libraries and the corresponding pip install commands:
    
    azure-identity: For Azure authentication.
    requests: For making HTTP requests.
    semantic-kernel: For the semantic kernel framework (assuming this is a custom or specific library, you might need to install it from a specific source or repository).
    """
    logger.info("## **Samples: Multi-AI Agents for Booking hotel**")
    
    # !pip install azure-identity
    # !pip install requests
    # !pip install semantic-kernel
    # !pip install --upgrade semantic_kernel
    # !pip install azure-cli
    
    """
    # Explanation: 
    """
    logger.info("# Explanation:")
    
    
    """
    # Explanation:
    By using from dotenv import load_dotenv and load_dotenv(), you can easily manage configuration settings and sensitive information (like API keys and database URLs) in a .env file, keeping them separate from your source code and making your application more secure and easier to configure.
    """
    logger.info("# Explanation:")
    
    
    load_dotenv()
    
    """
    # Explanation:
    
    Import Statement: from azure.identity.aio import DefaultAzureCredential: This imports the DefaultAzureCredential class from the azure.identity.aio module. The aio part of the module name indicates that it is designed for asynchronous operations.
    
    Purpose of DefaultAzureCredential: The DefaultAzureCredential class is part of the Azure SDK for Python. It provides a default way to authenticate with Azure services. It attempts to authenticate using multiple methods in a specific order, such as environment variables, managed identity, and Azure CLI credentials.
    
    Asynchronous Operations:The aio module indicates that the DefaultAzureCredential class supports asynchronous operations. This means you can use it with asyncio to perform non-blocking authentication requests.
    """
    logger.info("# Explanation:")
    
    
    """
    # Explanation:
    Imports various modules and classes from the semantic_kernel package. Here's a breakdown of each import:
    
    AgentGroupChat from semantic_kernel.agents: This class handles functionalities related to group chat for AI agents. AzureAIAgent and AzureAIAgentSettings from semantic_kernel.agents.azure_ai
    
    AzureAIAgent: This class is used to create and manage AI agents that utilize Azure AI services.
    
    AzureAIAgentSettings: This class is used to configure settings for the AzureAIAgent. TerminationStrategy from semantic_kernel.agents.strategies.termination.termination_strategy:
    
    This class defines strategies for terminating the execution of AI agents under certain conditions. ChatMessageContent from semantic_kernel.contents.chat_message_content:
    
    This class is used to handle the content of chat messages.
    AuthorRole from semantic_kernel.contents.utils.author_role:
    
    This class defines different roles for authors in the context of chat messages. 
    
    kernel_function from semantic_kernel.functions.kernel_function_decorator: This decorator is used to define kernel functions, which are functions that can be executed within the semantic kernel framework.
    These imports set up the necessary components for creating and managing AI agents that can interact in a group chat environment, possibly for tasks such as booking hotels or similar activities.
    """
    logger.info("# Explanation:")
    
    
    """
    # Explanation:
    Next we import the CodeInterpreterTool class from the azure.ai.projects.models module. 
    
    CodeInterpreterTool: This class is part of the Azure AI SDK and is used to interpret and execute code within the context of AI projects. It provides functionalities for running code snippets, analyzing code, or integrating code execution within AI workflows.
    This import sets up the necessary component for utilizing the CodeInterpreterTool in your project, which could be useful for tasks that involve interpreting and executing code dynamically.
    """
    logger.info("# Explanation:")
    
    
    """
    # Explanation: 
    The ApprovalTerminationStrategy class provides a specific strategy for terminating an AI agent's operation. The agent will terminate if the last message in its interaction history contains the word "saved". This could be useful in scenarios where the agent's task is considered complete once it receives confirmation that something has been "saved".Define the interaction method. After the reservation plan is saved, it can be stopped when receiving the saved signal
    """
    logger.info("# Explanation:")
    
    class ApprovalTerminationStrategy(TerminationStrategy):
        """A strategy for determining when an agent should terminate."""
    
        async def should_agent_terminate(self, agent, history):
            """Check if the agent should terminate."""
            return "saved" in history[-1].content.lower()
    
    """
    # Explanation:
    
    The line of code initializes an AzureAIAgentSettings object with default or predefined settings by calling the create() method. This settings object (ai_agent_settings) can then be used to configure and manage an AzureAIAgent instance.
    """
    logger.info("# Explanation:")
    
    ai_agent_settings = AzureAIAgentSettings.create()
    
    """
    # Explanation:
    By importing the requests library, you can easily make HTTP requests and interact with web services in your Python code.
    """
    logger.info("# Explanation:")
    
    
    """
    # Explanation:
    This is a variable that stores the API key for accessing a SERP (Search Engine Results Page) API service. An API key is a unique identifier used to authenticate requests associated with your account.
    
    'GOOGLE_SEARCH_API_KEY': This is a placeholder string. You need to replace ''GOOGLE_SEARCH_API_KEY' with your actual SERP API key.
    
    Purpose: The purpose of this line is to store the API key in a variable so that it can be used to authenticate requests to the SERP API service. The API key is required to access the service and perform searches.
    
    How to Get a SERP API Key: To get a SERP API key, follow these general steps at https://serpapi.com (the exact steps may vary depending on the specific SERP API service you are using):
    
    Choose a SERP API Service: There are several SERP API services available, such as SerpAPI, Google Custom Search JSON API, and others. Choose the one that best fits your needs.
    
    Sign Up for an Account:
    
    Go to the website of the chosen SERP API service https://www.serpapi.com and sign up for an account. You may need to provide some basic information and verify your email address.
    
    Create an API Key:
    
    After signing up, log in to your account and navigate to the API section or dashboard. Look for an option to create or generate a new API key.
    Copy the API Key:
    
    Once the API key is generated, copy it. This key will be used to authenticate your requests to the SERP API service.
    Replace the Placeholder:
    
    Replace the placeholder in your .env file
    """
    logger.info("# Explanation:")
    
    SERPAPI_SEARCH_API_KEY=os.getenv('SERPAPI_SEARCH_API_KEY')
    
    SERPAPI_SEARCH_ENDPOINT = os.getenv('SERPAPI_SEARCH_ENDPOINT')
    
    """
    # Explanation:
    The BookingPlugin class provides methods for booking hotels and flights using the Serpapi.com Google Search API. It constructs the necessary parameters, sends API requests, and processes the responses to return relevant booking information. The API key (SERPAPI_SEARCH_API_KEY) and endpoint (SERPAPI_SEARCH_ENDPOINT) are used to authenticate and send requests to the Google Search API.
    """
    logger.info("# Explanation:")
    
    class BookingPlugin:
        """Booking Plugin for customers"""
        @kernel_function(description="booking hotel")
        def booking_hotel(self,query: Annotated[str, "The name of the city"], check_in_date: Annotated[str, "Hotel Check-in Time"], check_out_date: Annotated[str, "Hotel Check-in Time"])-> Annotated[str, "Return the result of booking hotel infomation"]:
    
            params = {
                "engine": "google_hotels",
                "q": query,
                "check_in_date": check_in_date,
                "check_out_date": check_out_date,
                "adults": "2",
                "currency": "USD",
                "gl": "us",
                "hl": "en",
                "api_key": SERPAPI_SEARCH_API_KEY
            }
    
            response = requests.get(SERPAPI_SEARCH_ENDPOINT, params=params)
            if response.status_code == 200:
                response = response.json()
                return response["properties"]
            else:
                return None
    
    
        @kernel_function(description="booking fight")
        def  booking_fight(self,origin: Annotated[str, "The name of Departure"], destination: Annotated[str, "The name of Destination"], outbound_date: Annotated[str, "The date of outbound"], return_date: Annotated[str, "The date of Return_date"])-> Annotated[str, "Return the result of booking fight infomation"]:
    
            go_params = {
                "engine": "google_flights",
                "departure_id": origin,
                "arrival_id": destination,
                "outbound_date": outbound_date,
                "return_date": return_date,
                "currency": "USD",
                "hl": "en",
                "api_key": SERPAPI_SEARCH_API_KEY
            }
    
            logger.debug(go_params)
    
            go_response = requests.get(SERPAPI_SEARCH_ENDPOINT, params=go_params)
    
    
            result = ''
    
            if go_response.status_code == 200:
                response = go_response.json()
    
                result += "# outbound \n " + str(response)
            else:
                logger.debug('error!!!')
    
    
            back_params = {
                "engine": "google_flights",
                "departure_id": destination,
                "arrival_id": origin,
                "outbound_date": return_date,
                "return_date": return_date,
                "currency": "USD",
                "hl": "en",
                "api_key": SERPAPI_SEARCH_API_KEY
            }
    
    
            logger.debug(back_params)
    
    
            back_response = requests.get(SERPAPI_SEARCH_ENDPOINT, params=back_params)
    
    
    
            if back_response.status_code == 200:
                response = back_response.json()
    
                result += "\n # return \n"  + str(response)
    
            else:
                logger.debug('error!!!')
    
            logger.debug(result)
    
            return result
    
    """
    # Explanation:
    The SavePlugin class provides a method saving_plan to save trip plans using Azure AI services. It sets up Azure credentials, creates an AI agent, processes user inputs to generate and save the trip plan content, and handles file saving and cleanup operations. The method returns "Saved" upon successful completion.
    """
    logger.info("# Explanation:")
    
    class SavePlugin:
        """Save Plugin for customers"""
        @kernel_function(description="saving plan")
        async def saving_plan(self,tripplan: Annotated[str, "The content of trip plan"])-> Annotated[str, "Return status of save content"]:
    
            async with (
                    DefaultAzureCredential() as creds,
                    AzureAIAgent.create_client(
                        credential=creds,
                        conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
                    ) as client,
            logger.success(format_json(result))
            ):
    
                code_interpreter = CodeInterpreterTool()
    
                agent_definition = await client.agents.create_agent(
                        model=ai_agent_settings.model_deployment_name,
                        tools=code_interpreter.definitions,
                        tool_resources=code_interpreter.resources,
                    )
                logger.success(format_json(agent_definition))
    
    
                agent = AzureAIAgent(
                    client=client,
                    definition=agent_definition,
                )
    
                thread = await client.agents.create_thread()
                logger.success(format_json(thread))
    
    
                user_inputs = [
                    """
    
                            You are my Python programming assistant. Generate code,save """+ tripplan +
    
                        """
                            and execute it according to the following requirements
    
                            1. Save blog content to trip-{YYMMDDHHMMSS}.md
    
                            2. give me the download this file link
                        """
                ]
    
    
    
                try:
                    for user_input in user_inputs:
                        await agent.add_chat_message(
                            thread_id=thread.id, message=ChatMessageContent(role=AuthorRole.USER, content=user_input)
                        )
                        logger.debug(f"# User: '{user_input}'")
                        async for content in agent.invoke(thread_id=thread.id):
                            if content.role != AuthorRole.TOOL:
                                logger.debug(f"# Agent: {content.content}")
    
    
                        messages = await client.agents.list_messages(thread_id=thread.id)
                        logger.success(format_json(messages))
    
    
    
                        for file_path_annotation in messages.file_path_annotations:
    
                                file_name = os.path.basename(file_path_annotation.text)
    
                                await client.agents.save_file(file_id=file_path_annotation.file_path.file_id, file_name=file_name,target_dir="./trip")
    
    
                finally:
                    await client.agents.delete_thread(thread.id)
                    await client.agents.delete_agent(agent.id)
    
    
            return "Saved"
    
    """
    # Explanation:
    This code sets up Azure AI agents to handle booking flights and hotels, and saving trip plans based on user inputs. It uses Azure credentials to create and configure the agents, processes user inputs through a group chat, and ensures proper cleanup after the tasks are completed. The agents use specific plugins (BookingPlugin and SavePlugin) to perform their respective tasks.
    """
    logger.info("# Explanation:")
    
    async with (
            DefaultAzureCredential() as creds,
            AzureAIAgent.create_client(
                credential=creds,
                conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
            ) as client,
    logger.success(format_json(result))
    ):
        BOOKING_AGENT_NAME = "BookingAgent"
        BOOKING_AGENT_INSTRUCTIONS = """
        You are a booking agent. Help me book flights or hotels.
    
        Thought: Please understand the user's intention and confirm whether to use the reservation system to complete the task.
    
        Actions:
        - For flight bookings, convert the departure and destination names into airport codes.
        - Use the appropriate API for hotel or flight bookings. Verify that all necessary parameters are available. If any parameters are missing, ask the user to provide them. If all parameters are complete, call the corresponding function.
        - If the task is not related to hotel or flight booking, respond with the final answer only.
        - Output the results using a markdown table:
          - For flight bookings, output separate outbound and return contents in the order of:
            Departure Airport | Airline | Flight Number | Departure Time | Arrival Airport | Arrival Time | Duration | Airplane | Travel Class | Price (USD) | Legroom | Extensions | Carbon Emissions (kg).
          - For hotel bookings, output in the order of:
            Property Name | Property Description | Check-in Time | Check-out Time | Prices | Nearby Places | Hotel Class | GPS Coordinates.
        """
    
        SAVE_AGENT_NAME = "SaveAgent"
        SAVE_AGENT_INSTRUCTIONS = """
        You are a save tool agent. Help me to save the trip plan.
        """
    
        booking_agent_definition = await client.agents.create_agent(
                model=ai_agent_settings.model_deployment_name,
                name=BOOKING_AGENT_NAME,
                instructions=BOOKING_AGENT_INSTRUCTIONS,
            )
        logger.success(format_json(booking_agent_definition))
    
        booking_agent = AzureAIAgent(
            client=client,
            definition=booking_agent_definition,
        )
    
        booking_agent.kernel.add_plugin(BookingPlugin(), plugin_name="booking")
    
        save_agent_definition = await client.agents.create_agent(
                model=ai_agent_settings.model_deployment_name,
                name=SAVE_AGENT_NAME,
                instructions=SAVE_AGENT_INSTRUCTIONS
            )
        logger.success(format_json(save_agent_definition))
    
        save_agent = AzureAIAgent(
            client=client,
            definition=save_agent_definition,
        )
    
        save_agent.kernel.add_plugin(SavePlugin(), plugin_name="saving")
    
        user_inputs = [
            "I have a business trip from London to New York in Feb 20 2025 to Feb 27 2025 ,help me to book a hotel and fight tickets and save it"
        ]
    
        chat = AgentGroupChat(
            agents=[booking_agent, save_agent],
            termination_strategy=ApprovalTerminationStrategy(agents=[save_agent], maximum_iterations=10),
        )
    
        try:
            for user_input in user_inputs:
                await chat.add_chat_message(
                    ChatMessageContent(role=AuthorRole.USER, content=user_input)
                )
                logger.debug(f"# User: '{user_input}'")
    
                async for content in chat.invoke():
                    logger.debug(f"# {content.role} - {content.name or '*'}: '{content.content}'")
    
                logger.debug(f"# IS COMPLETE: {chat.is_complete}")
    
                logger.debug("*" * 60)
                logger.debug("Chat History (In Descending Order):\n")
                async for message in chat.get_chat_messages(agent=save_agent):
                    logger.debug(f"# {message.role} - {message.name or '*'}: '{message.content}'")
        finally:
            await chat.reset()
            await client.agents.delete_agent(save_agent.id)
            await client.agents.delete_agent(booking_agent.id)
    
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