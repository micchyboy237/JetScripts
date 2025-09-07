async def main():
    from IPython.display import display, HTML
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from openai import AsyncOllama
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
    from semantic_kernel.contents import FunctionCallContent,FunctionResultContent, StreamingTextContent
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
    
    This document provides an overview and explanation of the code used to create a Semantic Kernel-based tool that integrates with Azure AI Search for Retrieval-Augmented Generation (RAG). The example demonstrates how to build an AI agent that retrieves travel documents from an Azure AI Search index, augments user queries with semantic search results, and streams detailed travel recommendations.
    
    ## Initializing the Environment
    
    ### Importing Packages
    The following code imports the necessary packages:
    """
    logger.info("# Semantic Kernel Tool Use Example")
    
    
    
    
    
    
    
    
    """
    ### Creating the Semantic Kernel and AI Service
    
    A Semantic Kernel instance is created and configured with an asynchronous Ollama chat completion service. The service is added to the kernel for use in generating responses.
    """
    logger.info("### Creating the Semantic Kernel and AI Service")
    
    load_dotenv()
    client = AsyncOllama(
        api_key=os.environ["GITHUB_TOKEN"],
        base_url="https://models.inference.ai.azure.com/"
    )
    
    chat_completion_service = OllamaChatCompletion(
        ai_model_id="llama3.2",
        async_client=client,
    )
    
    """
    ### Defining the Prompt Plugin
    
    The PromptPlugin is a native plugin that defines a function to build an augmented prompt using retrieval context
    """
    logger.info("### Defining the Prompt Plugin")
    
    class SearchPlugin:
    
        def __init__(self, search_client: SearchClient):
            self.search_client = search_client
    
        @kernel_function(
            name="build_augmented_prompt",
            description="Build an augmented prompt using retrieval context or function results.",
        )
        def build_augmented_prompt(self, query: str, retrieval_context: str) -> str:
            return (
                f"Retrieved Context:\n{retrieval_context}\n\n"
                f"User Query: {query}\n\n"
                "First review the retrieved context, if this does not answer the query, try calling an available plugin functions that might give you an answer. If no context is available, say so."
            )
    
        @kernel_function(
            name="retrieve_documents",
            description="Retrieve documents from the Azure Search service.",
        )
        def get_retrieval_context(self, query: str) -> str:
            results = self.search_client.search(query)
            context_strings = []
            for result in results:
                context_strings.append(f"Document: {result['content']}")
            return "\n\n".join(context_strings) if context_strings else "No results found"
    
    class WeatherInfoPlugin:
        """A Plugin that provides the average temperature for a travel destination."""
    
        def __init__(self):
            self.destination_temperatures = {
                "maldives": "82°F (28°C)",
                "swiss alps": "45°F (7°C)",
                "african safaris": "75°F (24°C)"
            }
    
        @kernel_function(description="Get the average temperature for a specific travel destination.")
        def get_destination_temperature(self, destination: str) -> Annotated[str, "Returns the average temperature for the destination."]:
            """Get the average temperature for a travel destination."""
            normalized_destination = destination.lower()
    
            if normalized_destination in self.destination_temperatures:
                return f"The average temperature in {destination} is {self.destination_temperatures[normalized_destination]}."
            else:
                return f"Sorry, I don't have temperature information for {destination}. Available destinations are: Maldives, Swiss Alps, and African safaris."
    
    """
    ## Vector Database Initialization
    
    We initialize Azure AI Search with persistent storage and add enhanced sample documents. Azure AI Search will be used to store and retrieve documents that provide context for generating accurate responses.
    """
    logger.info("## Vector Database Initialization")
    
    search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = "travel-documents"
    
    search_client = SearchClient(
        endpoint=search_service_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_api_key)
    )
    
    index_client = SearchIndexClient(
        endpoint=search_service_endpoint,
        credential=AzureKeyCredential(search_api_key)
    )
    
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String)
    ]
    
    index = SearchIndex(name=index_name, fields=fields)
    
    try:
        existing_index = index_client.get_index(index_name)
        logger.debug(f"Index '{index_name}' already exists, using the existing index.")
    except Exception:
        logger.debug(f"Creating new index '{index_name}'...")
        index_client.create_index(index)
    
    
    documents = [
        {"id": "1", "content": "Contoso Travel offers luxury vacation packages to exotic destinations worldwide."},
        {"id": "2", "content": "Our premium travel services include personalized itinerary planning and 24/7 concierge support."},
        {"id": "3", "content": "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage."},
        {"id": "4", "content": "Popular destinations include the Maldives, Swiss Alps, and African safaris."},
        {"id": "5", "content": "Contoso Travel provides exclusive access to boutique hotels and private guided tours."}
    ]
    
    search_client.upload_documents(documents)
    
    agent = ChatCompletionAgent(
        service=chat_completion_service,
        plugins=[SearchPlugin(search_client=search_client), WeatherInfoPlugin()],
        name="TravelAgent",
        instructions="Answer travel queries using the provided tools and context. If context is provided, do not say 'I have no context for that.'",
    )
    
    """
    ### Running the Agent with Streaming Invocation
    
    The main asynchronous loop creates a thread for the conversation and, for each user input,  so that the agent sees the retrieval context. The user message is also added, and then the agent is invoked using streaming. The output is printed as it streams in.
    """
    logger.info("### Running the Agent with Streaming Invocation")
    
    async def main():
        thread: ChatHistoryAgentThread | None = None
    
        user_inputs = [
            "Can you explain Contoso's travel insurance coverage?",
            "What is the average temperature of the Maldives?",
            "What is a good cold destination offered by Contoso and what is it average temperature?",
        ]
    
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
    
    """
    You should see output similar to the following:
    
    ```text
    User: 
    Can you explain Contoso's travel insurance coverage?
    
    Function Calls (click to expand)
    
    Calling function: retrieve_documents({"query": "Contoso travel insurance coverage"})
    
    Function Result:
    
    Document: Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage.
    
    Document: Contoso Travel offers luxury vacation packages to exotic destinations worldwide.
    
    Document: Contoso Travel provides exclusive access to boutique hotels and private guided tours.
    
    Document: Our premium travel services include personalized itinerary planning and 24/7 concierge support.
    
    TravelAgent:
    
    Contoso's travel insurance coverage includes the following:
    
    1. **Medical Emergencies**: Coverage for unforeseen medical issues that may arise while traveling.
    2. **Trip Cancellations**: Protection in case you need to cancel your trip for covered reasons.
    3. **Lost Baggage**: Compensation for baggage that is lost during your trip.
    
    If you need more specific details about the policy, it would be best to contact Contoso directly or refer to their official documentation.
    ```
    """
    logger.info("You should see output similar to the following:")
    
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