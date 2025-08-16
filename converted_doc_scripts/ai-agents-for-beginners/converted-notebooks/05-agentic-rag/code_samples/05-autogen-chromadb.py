import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from jet.logger import CustomLogger
from typing import List, Dict
import asyncio
import chromadb
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agentic RAG with Autogen

This notebook demonstrates implementing Retrieval-Augmented Generation (RAG) using Autogen agents with enhanced evaluation capabilities.

SQLite Version Fix
If you encounter the error:
```
RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
```

Uncomment this code block at the start of your notebook:
"""
logger.info("# Agentic RAG with Autogen")






load_dotenv()

"""
## Create the Client 

First, we initialize the Azure AI Chat Completion Client. This client will be used to interact with the Azure Ollama service to generate responses to user queries.
"""
logger.info("## Create the Client")

client = AzureAIChatCompletionClient(
    model="llama3.1",
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

We initialize ChromaDB with persistent storage and add enhanced sample documents. ChromaDB will be used to store and retrieve documents that provide context for generating accurate responses.
"""
logger.info("## Vector Database Initialization")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.create_collection(
    name="travel_documents",
    metadata={"description": "travel_service"},
    get_or_create=True
)

documents = [
    "Contoso Travel offers luxury vacation packages to exotic destinations worldwide.",
    "Our premium travel services include personalized itinerary planning and 24/7 concierge support.",
    "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage.",
    "Popular destinations include the Maldives, Swiss Alps, and African safaris.",
    "Contoso Travel provides exclusive access to boutique hotels and private guided tours."
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": "training", "type": "explanation"} for _ in documents]
)

"""
## Context Provider Implementation

The `ContextProvider` class handles context retrieval and integration from multiple sources:

1. **Vector Database Retrieval**: Uses ChromaDB to perform semantic search on travel documents
2. **Weather Information**: Maintains a simulated weather database for major cities
3. **Unified Context**: Combines both document and weather data into comprehensive context

Key methods:
- `get_retrieval_context()`: Retrieves relevant documents based on query
- `get_weather_data()`: Provides weather information for specified locations
- `get_unified_context()`: Combines both document and weather contexts for enhanced responses
"""
logger.info("## Context Provider Implementation")

class ContextProvider:
    def __init__(self, collection):
        self.collection = collection
        self.weather_database = {
            "new york": {"temperature": 72, "condition": "Partly Cloudy", "humidity": 65, "wind": "10 mph"},
            "london": {"temperature": 60, "condition": "Rainy", "humidity": 80, "wind": "15 mph"},
            "tokyo": {"temperature": 75, "condition": "Sunny", "humidity": 50, "wind": "5 mph"},
            "sydney": {"temperature": 80, "condition": "Clear", "humidity": 45, "wind": "12 mph"},
            "paris": {"temperature": 68, "condition": "Cloudy", "humidity": 70, "wind": "8 mph"},
        }

    def get_retrieval_context(self, query: str) -> str:
        """Retrieves relevant documents from vector database based on query."""
        results = self.collection.query(
            query_texts=[query],
            include=["documents", "metadatas"],
            n_results=2
        )
        context_strings = []
        if results and results.get("documents") and len(results["documents"][0]) > 0:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                context_strings.append(f"Document: {doc}\nMetadata: {meta}")
        return "\n\n".join(context_strings) if context_strings else "No relevant documents found"

    def get_weather_data(self, location: str) -> str:
        """Simulates retrieving weather data for a given location."""
        if not location:
            return ""

        location_key = location.lower()
        if location_key in self.weather_database:
            data = self.weather_database[location_key]
            return f"Weather for {location.title()}:\n" \
                   f"Temperature: {data['temperature']}°F\n" \
                   f"Condition: {data['condition']}\n" \
                   f"Humidity: {data['humidity']}%\n" \
                   f"Wind: {data['wind']}"
        else:
            return f"No weather data available for {location}."

    def get_unified_context(self, query: str, location: str = None) -> str:
        """Returns a unified context combining both document retrieval and weather data."""
        retrieval_context = self.get_retrieval_context(query)

        weather_context = ""
        if location:
            weather_context = self.get_weather_data(location)
            weather_intro = f"\nWeather Information for {location}:\n"
        else:
            weather_intro = ""

        return f"Retrieved Context:\n{retrieval_context}\n\n{weather_intro}{weather_context}"

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
## Query Processing with RAG

We define the `ask_rag` function to send the query to the assistant, process the response, and evaluate it. This function handles the interaction with the assistant and uses the evaluator to measure the quality of the response.
"""
logger.info("## Query Processing with RAG")

async def ask_rag_agent(query: str, context_provider: ContextProvider, location: str = None):
    """
    Sends a query to the assistant agent with context from the provider.

    Args:
        query: The user's question
        context_provider: The context provider instance
        location: Optional location for weather queries
    """
    try:
        context = context_provider.get_unified_context(query, location)

        augmented_query = (
            f"{context}\n\n"
            f"User Query: {query}\n\n"
            "Based ONLY on the above context, please provide a helpful answer."
        )

        start_time = time.time()
        async def async_func_19():
            response = await assistant.on_messages(
                [TextMessage(content=augmented_query, source="user")],
                cancellation_token=CancellationToken(),
            )
            return response
        response = asyncio.run(async_func_19())
        logger.success(format_json(response))
        processing_time = time.time() - start_time

        return {
            'query': query,
            'response': response.chat_message.content,
            'processing_time': processing_time,
            'location': location
        }
    except Exception as e:
        logger.debug(f"Error processing query: {e}")
        return None

"""
# Example usage

We initialize the evaluator and define the queries that we want to process and evaluate.
"""
logger.info("# Example usage")

async def main():
    context_provider = ContextProvider(collection)

    queries = [
        {"query": "What does Contoso's travel insurance cover?"},
        {"query": "What's the weather like in London?", "location": "london"},
        {"query": "What luxury destinations does Contoso offer and what's the weather in Paris?", "location": "paris"},
    ]

    logger.debug("=== Autogen RAG Demo ===")
    for query_data in queries:
        query = query_data["query"]
        location = query_data.get("location")

        logger.debug(f"\n\nQuery: {query}")
        if location:
            logger.debug(f"Location: {location}")

        context = context_provider.get_unified_context(query, location)
        logger.debug("\n--- Context Used ---")
        logger.debug(context)
        logger.debug("-------------------")

        async def run_async_code_3da414ec():
            async def run_async_code_aa92cf90():
                result = await ask_rag_agent(query, context_provider, location)
                return result
            result = asyncio.run(run_async_code_aa92cf90())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_3da414ec())
        logger.success(format_json(result))
        if result:
            logger.debug(f"\nResponse: {result['response']}")
        logger.debug("\n" + "="*50)

"""
## Run the Script

We check if the script is running in an interactive environment or a standard script, and run the main function accordingly.
"""
logger.info("## Run the Script")

if __name__ == "__main__":
    if asyncio.get_event_loop().is_running():
        async def run_async_code_42b014ad():
            await main()
            return 
         = asyncio.run(run_async_code_42b014ad())
        logger.success(format_json())
    else:
        asyncio.run(main())

logger.info("\n\n[DONE]", bright=True)