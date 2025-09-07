async def main():
    from jet.transformers.formatters import format_json
    from azure.identity.aio import DefaultAzureCredential
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from semantic_kernel import __version__
    from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
    from semantic_kernel.functions import kernel_function
    from typing import Annotated
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
    # Example Sample Hotel and Flight Booker Agent 
    
    This solution will help you book flight tickets and hotel.  The scenario is a trip London Heathrow LHR Feb 20th 2024 to New York JFK returning Feb 27th 2025 flying economy with British Airways only. I want a stay in a Hilton hotel in New York please provide costs for the flight and hotel.
    
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
    
    # Setup 
    
    To run this notebook, you will need to make sure you've installed the required libraries by running `pip install -r requirements.txt`.
    """
    logger.info("# Example Sample Hotel and Flight Booker Agent")
    
    
    __version__
    
    """
    Your Semantic Kernel version should be at least 1.27.2.
    
    Load your .env file setting and resources please ensure you have added your keys and setting and created a local .env file.
    """
    logger.info("Your Semantic Kernel version should be at least 1.27.2.")
    
    
    load_dotenv()
    
    """
    # Log in to Azure
    
    You now need to log in to Azure. Open a terminal and run the following command:
    
    ```bash
    az login
    ```
    
    This command will prompt you to enter your Azure credentials, enabling the Azure AI Agent service to function correctly.
    
    # Explanation:
    This is a variable that stores the API key for accessing a SERP (Search Engine Results Page) API service. An API key is a unique identifier used to authenticate requests associated with your account.
    
    Purpose: The purpose of this line is to store the API key in a variable so that it can be used to authenticate requests to the SERP API service. The API key is required to access the service and perform searches.
    How to Get a SERP API Key: To get a SERP API key, follow these general steps at https://serpapi.com (the exact steps may vary depending on the specific SERP API service you are using):
    
    Choose a SERP API Service: There are several SERP API services available, such as SerpAPI, Google Custom Search JSON API, and others. Choose the one that best fits your needs.
    
    Sign Up for an Account: Go to the website of the chosen SERP API service and sign up for an account. You may need to provide some basic information and verify your email address.
    
    Create an API Key: After signing up, log in to your account and navigate to the API section or dashboard. Look for an option to create or generate a new API key.
    Copy the API Key to your .env file.
    """
    logger.info("# Log in to Azure")
    
    SERP_API_KEY='SERPAPI_SEARCH_API_KEY'
    
    """
    # Explanation:
    BASE_URL: This is a variable that stores the base URL for the SERP API endpoint. The variable name BASE_URL is a convention used to indicate that this URL is the starting point for making API requests.
    'https://serpapi.com/search':
    
    This is the actual URL string assigned to the BASE_URL variable. It represents the endpoint for performing search queries using the SERP API.
    
    # Purpose:
    The purpose of this line is to define a constant that holds the base URL for the SERP API. This URL will be used as the starting point for constructing API requests to perform search operations.
    
    # Usage:
    By defining the base URL in a variable, you can easily reuse it throughout your code whenever you need to make requests to the SERP API. This makes your code more maintainable and reduces the risk of errors from hardcoding the URL in multiple places. The current example is https://serpapi.com/search?engine=bing which is using Bing search API. Different API can be selected at https://Serpapi.com
    """
    logger.info("# Explanation:")
    
    BASE_URL = 'https://serpapi.com/search?engine=bing'
    
    """
    # Explanation:
    
    This is where your plugin code is located.
    
    Class Definition: `class BookingPlugin`: Defines a class named BookingPlugin that contains methods for booking hotels and flights.
    
    Hotel Booking Method:
    
    - `@kernel_function(description="booking hotel")`: A decorator that describes the function as a kernel function for booking hotels.
    - `def booking_hotel(self, query: Annotated[str, "The name of the city"], check_in_date: Annotated[str, "Hotel Check-in Time"], check_out_date: Annotated[str, "Hotel Check-out Time"]) -> Annotated[str, "Return the result of booking hotel information"]:`: Defines a method for booking hotels with annotated parameters and return type.
    
    The method constructs a dictionary of parameters for the hotel booking request and sends a GET request to the SERP API. It checks the response status and returns the hotel properties if successful, or None if the request failed.
    
    Flight Booking Method: 
    
    - `@kernel_function(description="booking flight")`: A decorator that describes the function as a kernel function for booking flights.
    - `def booking_flight(self, origin: Annotated[str, "The name of Departure"], destination: Annotated[str, "The name of Destination"], outbound_date: Annotated[str, "The date of outbound"], return_date: Annotated[str, "The date of Return_date"]) -> Annotated[str, "Return the result of booking flight information"]:`: Defines a method for booking flights with annotated parameters and return type.
    
    The method constructs dictionaries of parameters for the outbound and return flight requests and sends GET requests to the SERP API. It checks the response status and appends the flight information to the result string if successful, or prints an error message if the request failed. The method returns the result string containing the flight information.
    """
    logger.info("# Explanation:")
    
    
    
    
    class BookingPlugin:
        """Booking Plugin for customers"""
    
        @kernel_function(description="booking hotel")
        def booking_hotel(
            self,
            query: Annotated[str, "The name of the city"],
            check_in_date: Annotated[str, "Hotel Check-in Time"],
            check_out_date: Annotated[str, "Hotel Check-out Time"],
        ) -> Annotated[str, "Return the result of booking hotel information"]:
            """
            Function to book a hotel.
            Parameters:
            - query: The name of the city
            - check_in_date: Hotel Check-in Time
            - check_out_date: Hotel Check-out Time
            Returns:
            - The result of booking hotel information
            """
    
            params = {
                "engine": "google_hotels",
                "q": query,
                "check_in_date": check_in_date,
                "check_out_date": check_out_date,
                "adults": "1",
                "currency": "GBP",
                "gl": "uk",
                "hl": "en",
                "api_key": SERP_API_KEY
            }
    
            response = requests.get(BASE_URL, params=params)
    
            if response.status_code == 200:
                response = response.json()
                return response["properties"]
            else:
                return None
    
        @kernel_function(description="booking flight")
        def booking_flight(
            self,
            origin: Annotated[str, "The name of Departure"],
            destination: Annotated[str, "The name of Destination"],
            outbound_date: Annotated[str, "The date of outbound"],
            return_date: Annotated[str, "The date of Return_date"],
        ) -> Annotated[str, "Return the result of booking flight information"]:
            """
            Function to book a flight.
            Parameters:
            - origin: The name of Departure
            - destination: The name of Destination
            - outbound_date: The date of outbound
            - return_date: The date of Return_date
            - airline: The preferred airline carrier
            - hotel_brand: The preferred hotel brand
            Returns:
            - The result of booking flight information
            """
    
            go_params = {
                "engine": "google_flights",
                "departure_id": "destination",
                "arrival_id": "origin",
                "outbound_date": "outbound_date",
                "return_date": "return_date",
                "currency": "GBP",
                "hl": "en",
                "airline": "airline",
                "hotel_brand": "hotel_brand",
                "api_key": "SERP_API_KEY"
            }
    
            logger.debug(go_params)
    
            go_response = requests.get(BASE_URL, params=go_params)
    
            result = ''
    
            if go_response.status_code == 200:
                response = go_response.json()
                result += "# outbound \n " + str(response)
            else:
                logger.debug('error!!!')
    
            back_params = {
                "departure_id": destination,
                "arrival_id": origin,
                "outbound_date": outbound_date,
                "return_date": return_date,
                "currency": "GBP",
                "hl": "en",
                "api_key": SERP_API_KEY
            }
    
            back_response = requests.get(BASE_URL, params=back_params)
    
            if back_response.status_code == 200:
                response = back_response.json()
                result += "\n # return \n" + str(response)
            else:
                logger.debug('error!!!')
    
            logger.debug(result)
    
            return result
    
    """
    # Explanation:
    Import Statements: Import necessary modules for Azure credentials, AI agent, chat message content, author role, and kernel function decorator.
    
    Asynchronous Context Manager: async with (DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds, conn_str="...") as client,): This sets up an asynchronous context manager to handle Azure credentials and create an AI agent client.
    
    Agent Name and Instructions: 
    - `AGENT_NAME = "BookingAgent"`: Defines the name of the agent.
    - `AGENT_INSTRUCTIONS = """
    logger.info("# Explanation:")..."""`: Provides detailed instructions for the agent on how to handle booking requests.
    
    Create Agent Definition: `agent_definition = await client.agents.create_agent(...)`: Creates an agent definition with the specified model, name, and instructions.
    
    Create AzureAI Agent: `agent = AzureAIAgent(...)`: Creates an AzureAI agent using the client, agent definition, and the defined plugin.
    
    Create Thread: `thread: AzureAIAgentThread | None = None`: Create a thread for the agent. It isn't required to first create a thread - if the value of `None` is provided, a new thread will be created during the first invocation and returned as part of the response.
    
    User Inputs: `user_inputs = ["..."]`: Defines a list of user inputs for the agent to process.
    
    In the finally block, delete the thread and agent to clean up resources.
    
    # Authentication
    
    The `DefaultAzureCredential` class is part of the Azure SDK for Python. It provides a default way to authenticate with Azure services. It attempts to authenticate using multiple methods in a specific order, such as environment variables, managed identity, and Azure CLI credentials.
    
    Asynchronous Operations: The aio module indicates that the DefaultAzureCredential class supports asynchronous operations. This means you can use it with asyncio to perform non-blocking authentication requests.
    """
    logger.info("# Authentication")
    
    
    ai_agent_settings = AzureAIAgentSettings.create()
    
    async with (
             DefaultAzureCredential() as creds,
            AzureAIAgent.create_client(
                credential=creds,
                conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
            ) as client,
    logger.success(format_json(result))
    ):
    
        AGENT_NAME = "BookingAgent"
        AGENT_INSTRUCTIONS = """
        You are a booking agent, help me to book flights or hotels.
    
        Thought: Understand the user's intention and confirm whether to use the reservation system to complete the task.
    
        Action:
        - If booking a flight, convert the departure name and destination name into airport codes.
        - If booking a hotel or flight, use the corresponding API to call. Ensure that the necessary parameters are available. If any parameters are missing, use default values or assumptions to proceed.
        - If it is not a hotel or flight booking, respond with the final answer only.
        - Output the results using a markdown table:
        - For flight bookings, separate the outbound and return contents and list them in the order of Departure_airport Name | Airline | Flight Number | Departure Time | Arrival_airport Name | Arrival Time | Duration | Airplane | Travel Class | Price (USD) | Legroom | Extensions | Carbon Emissions (kg).
        - For hotel bookings, list them in the order of Properties Name | Properties description | check_in_time | check_out_time | prices | nearby_places | hotel_class | gps_coordinates.
        """
    
        agent_definition = await client.agents.create_agent(
                model=ai_agent_settings.model_deployment_name,
                name=AGENT_NAME,
                instructions=AGENT_INSTRUCTIONS,
            )
        logger.success(format_json(agent_definition))
    
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
            plugins=[BookingPlugin()]
        )
    
        thread: AzureAIAgentThread | None = None
    
        user_inputs = [
            "Help me book flight tickets and hotel for the following trip London Heathrow LHR Feb 20th 2025 to New York JFK returning Feb 27th 2025 flying economy with British Airways only. I want a stay in a Hilton hotel in New York please provide costs for the flight and hotel"
        ]
    
        try:
            for user_input in user_inputs:
                logger.debug(f"# User: '{user_input}'")
                response = await agent.get_response(
                        messages=user_input,
                        thread=thread,
                    )
                logger.success(format_json(response))
                thread = response.thread
                logger.debug(f"{response.name}: '{response.content}'")
        finally:
            await thread.delete() if thread else None
            await client.agents.delete_agent(agent.id)
    
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