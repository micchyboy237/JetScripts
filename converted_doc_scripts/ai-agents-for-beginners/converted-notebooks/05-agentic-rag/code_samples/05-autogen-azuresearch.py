async def main():
    from jet.transformers.formatters import format_json
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken
    from autogen_ext.models.azure import AzureAIChatCompletionClient
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from typing import List, Dict
    import asyncio
    import os
    import shutil
    import time
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Agentic RAG with Autogen using Azure AI Services
    
    This notebook demonstrates implementing Retrieval-Augmented Generation (RAG) using Autogen agents with enhanced evaluation capabilities.
    """
    logger.info("# Agentic RAG with Autogen using Azure AI Services")
    
    
    
    
    
    load_dotenv()
    
    """
    ## Create the Client 
    
    First, we initialize the Azure AI Chat Completion Client. This client will be used to interact with the Azure Ollama service to generate responses to user queries.
    """
    logger.info("## Create the Client")
    
    client = AzureAIChatCompletionClient(
        model="llama3.2",
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
        model_info={
            "json_output": True,
            "function_calling": True,
            "vision": True,
            "family": "unknown",
        },
    )
    
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
    
    index_client.create_index(index)
    
    documents = [
        {"id": "1", "content": "Contoso Travel offers luxury vacation packages to exotic destinations worldwide."},
        {"id": "2", "content": "Our premium travel services include personalized itinerary planning and 24/7 concierge support."},
        {"id": "3", "content": "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage."},
        {"id": "4", "content": "Popular destinations include the Maldives, Swiss Alps, and African safaris."},
        {"id": "5", "content": "Contoso Travel provides exclusive access to boutique hotels and private guided tours."}
    ]
    
    search_client.upload_documents(documents)
    
    def get_retrieval_context(query: str) -> str:
        results = search_client.search(query)
        context_strings = []
        for result in results:
            context_strings.append(f"Document: {result['content']}")
        return "\n\n".join(context_strings) if context_strings else "No results found"
    
    def get_weather_data(location: str) -> str:
        """
        Simulates retrieving weather data for a given location.
        In a real-world scenario, this would call a weather API.
        """
        weather_database = {
            "new york": {"temperature": 72, "condition": "Partly Cloudy", "humidity": 65, "wind": "10 mph"},
            "london": {"temperature": 60, "condition": "Rainy", "humidity": 80, "wind": "15 mph"},
            "tokyo": {"temperature": 75, "condition": "Sunny", "humidity": 50, "wind": "5 mph"},
            "sydney": {"temperature": 80, "condition": "Clear", "humidity": 45, "wind": "12 mph"},
            "paris": {"temperature": 68, "condition": "Cloudy", "humidity": 70, "wind": "8 mph"},
        }
    
        location_key = location.lower()
    
        if location_key in weather_database:
            data = weather_database[location_key]
            return f"Weather for {location.title()}:\n" \
                   f"Temperature: {data['temperature']}°F\n" \
                   f"Condition: {data['condition']}\n" \
                   f"Humidity: {data['humidity']}%\n" \
                   f"Wind: {data['wind']}"
        else:
            return f"No weather data available for {location}."
    
    """
    ## Agent Configuration
    
    We configure the retrieval and assistant agents. The retrieval agent is specialized in finding relevant information using semantic search, while the assistant generates detailed responses based on the retrieved information.
    """
    logger.info("## Agent Configuration")
    
    assistant = AssistantAgent(
        name="assistant",
        model_client=client,
        system_message=(
            "You are a helpful AI assistant that provides answers using ONLY the provided context. "
            "Do NOT include any external information. Base your answer entirely on the context given below."
        ),
    )
    
    """
    ## RAGEvaluator Class
    
    We define the `RAGEvaluator` class to evaluate the response based on various metrics like response length, source citations, response time, and context relevance.
    """
    logger.info("## RAGEvaluator Class")
    
    class RAGEvaluator:
        def __init__(self):
            self.responses: List[Dict] = []
    
        def evaluate_response(self, query: str, response: str, context: List[Dict]) -> Dict:
            start_time = time.time()
            metrics = {
                'response_length': len(response),
                'source_citations': sum(1 for doc in context if doc["content"] in response),
                'evaluation_time': time.time() - start_time,
                'context_relevance': self._calculate_relevance(query, context)
            }
            self.responses.append({
                'query': query,
                'response': response,
                'metrics': metrics
            })
            return metrics
    
        def _calculate_relevance(self, query: str, context: List[Dict]) -> float:
            return sum(1 for c in context if query.lower() in c["content"].lower()) / len(context)
    
    """
    ## Query Processing with RAG
    
    We define the `ask_rag` function to send the query to the assistant, process the response, and evaluate it. This function handles the interaction with the assistant and uses the evaluator to measure the quality of the response.
    """
    logger.info("## Query Processing with RAG")
    
    async def ask_unified_rag(query: str, evaluator: RAGEvaluator, location: str = None):
        """
        A unified RAG function that combines both document retrieval and weather data
        based on the query and optional location parameter.
    
        Args:
            query: The user's question
            evaluator: The RAG evaluator to measure response quality
            location: Optional location for weather queries
        """
        try:
            retrieval_context = get_retrieval_context(query)
    
            weather_context = ""
            if location:
                weather_context = get_weather_data(location)
                weather_intro = f"\nWeather Information for {location}:\n"
            else:
                weather_intro = ""
    
            augmented_query = (
                f"Retrieved Context:\n{retrieval_context}\n\n"
                f"{weather_intro}{weather_context}\n\n"
                f"User Query: {query}\n\n"
                "Based ONLY on the above context, please provide the answer."
            )
    
            start_time = time.time()
            response = await assistant.on_messages(
                    [TextMessage(content=augmented_query, source="user")],
                    cancellation_token=CancellationToken(),
                )
            logger.success(format_json(response))
            processing_time = time.time() - start_time
    
            combined_context = documents.copy()  # Start with travel documents
    
            if location and weather_context:
                combined_context.append({"id": f"weather-{location}", "content": weather_context})
    
            metrics = evaluator.evaluate_response(
                query=query,
                response=response.chat_message.content,
                context=combined_context
            )
    
            result = {
                'response': response.chat_message.content,
                'processing_time': processing_time,
                'metrics': metrics,
            }
    
            if location:
                result['location'] = location
    
            return result
        except Exception as e:
            logger.debug(f"Error processing unified query: {e}")
            return None
    
    """
    # Example usage
    
    We initialize the evaluator and define the queries that we want to process and evaluate.
    """
    logger.info("# Example usage")
    
    async def main():
        evaluator = RAGEvaluator()
    
        user_inputs = [
            {"query": "Can you explain Contoso's travel insurance coverage?"},
    
            {"query": "What's the current weather condition in London?", "location": "london"},
    
            {"query": "What is a good cold destination offered by Contoso and what is its temperature?", "location": "london"},
        ]
    
        logger.debug("Processing Queries:")
        for query_data in user_inputs:
            query = query_data["query"]
            location = query_data.get("location")
    
            if location:
                logger.debug(f"\nProcessing Query for {location}: {query}")
            else:
                logger.debug(f"\nProcessing Query: {query}")
    
            retrieval_context = get_retrieval_context(query)
            weather_context = get_weather_data(location) if location else ""
    
            logger.debug("\n--- RAG Context ---")
            logger.debug(retrieval_context)
            if weather_context:
                logger.debug(f"\n--- Weather Context for {location} ---")
                logger.debug(weather_context)
            logger.debug("-------------------\n")
    
            result = await ask_unified_rag(query, evaluator, location)
            logger.success(format_json(result))
            if result:
                logger.debug("Response:", result['response'])
                logger.debug("\nMetrics:", result['metrics'])
            logger.debug("\n" + "="*60 + "\n")
    
    """
    ## Run the Script
    
    We check if the script is running in an interactive environment or a standard script, and run the main function accordingly.
    """
    logger.info("## Run the Script")
    
    if __name__ == "__main__":
        if asyncio.get_event_loop().is_running():
            await main()
        else:
            asyncio.run(main())
    
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