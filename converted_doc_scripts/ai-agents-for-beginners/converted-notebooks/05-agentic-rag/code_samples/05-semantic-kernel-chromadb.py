async def main():
    from IPython.display import display, HTML
    from chromadb.api.models.Collection import Collection
    from jet.logger import CustomLogger
    from openai import AsyncOllama
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
    from semantic_kernel.contents import FunctionCallContent,FunctionResultContent, StreamingTextContent
    from semantic_kernel.functions import kernel_function
    from typing import Annotated, TYPE_CHECKING
    import chromadb
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
    
    This document provides an overview and explanation of the code used to create a Semantic Kernel-based tool that integrates with ChromaDB for Retrieval-Augmented Generation (RAG). The example demonstrates how to build an AI agent that retrieves travel documents from a ChromaDB collection, augments user queries with semantic search results, and streams detailed travel recommendations.
    
    ## Initializing the Environment
    
    SQLite Version Fix
    If you encounter the error:
    ```
    RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
    ```
    
    Uncomment this code block at the start of your notebook:
    """
    logger.info("# Semantic Kernel Tool Use Example")
    
    
    
    """
    ### Importing Packages
    The following code imports the necessary packages:
    """
    logger.info("### Importing Packages")
    
    
    
    
    
    if TYPE_CHECKING:
    
    """
    ### Creating the Semantic Kernel and AI Service
    
    A Semantic Kernel instance is created and configured with an asynchronous Ollama chat completion service. The service is added to the kernel for use in generating responses.
    """
    logger.info("### Creating the Semantic Kernel and AI Service")
    
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
    
    class PromptPlugin:
    
        def __init__(self, collection: "Collection"):
            self.collection = collection
    
        @kernel_function(
            name="build_augmented_prompt",
            description="Build an augmented prompt using retrieval context."
        )
        def build_augmented_prompt(self, query: str, retrieval_context: str) -> str:
            return (
                f"Retrieved Context:\n{retrieval_context}\n\n"
                f"User Query: {query}\n\n"
                "Based ONLY on the above context, please provide your answer."
            )
    
        @kernel_function(name="retrieve_context", description="Retrieve context from the database.")
        def get_retrieval_context(self, query: str) -> str:
            results = self.collection.query(
                query_texts=[query],
                include=["documents", "metadatas"],
                n_results=2
            )
            context_entries = []
            if results and results.get("documents") and results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    context_entries.append(f"Document: {doc}\nMetadata: {meta}")
            return "\n\n".join(context_entries) if context_entries else "No retrieval context found."
    
    """
    ### Defining Weather Information Plugin
    
    The WeatherInfoPlugin is a native plugin that provides temperature information for specific travel destinations.
    """
    logger.info("### Defining Weather Information Plugin")
    
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
    ### Defining Destinations Information Plugin
    
    The DestinationsPlugin is a native plugin that provides detailed information about popular travel destinations.
    """
    logger.info("### Defining Destinations Information Plugin")
    
    class DestinationsPlugin:
        DESTINATIONS = {
            "maldives": {
                "name": "The Maldives",
                "description": "An archipelago of 26 atolls in the Indian Ocean, known for pristine beaches and overwater bungalows.",
                "best_time": "November to April (dry season)",
                "activities": ["Snorkeling", "Diving", "Island hopping", "Spa retreats", "Underwater dining"],
                "avg_cost": "$400-1200 per night for luxury resorts"
            },
            "swiss alps": {
                "name": "The Swiss Alps",
                "description": "Mountain range spanning across Switzerland with picturesque villages and world-class ski resorts.",
                "best_time": "December to March for skiing, June to September for hiking",
                "activities": ["Skiing", "Snowboarding", "Hiking", "Mountain biking", "Paragliding"],
                "avg_cost": "$250-500 per night for alpine accommodations"
            },
            "safari": {
                "name": "African Safari",
                "description": "Wildlife viewing experiences across various African countries including Kenya, Tanzania, and South Africa.",
                "best_time": "June to October (dry season) for optimal wildlife viewing",
                "activities": ["Game drives", "Walking safaris", "Hot air balloon rides", "Cultural village visits"],
                "avg_cost": "$400-800 per person per day for luxury safari packages"
            },
            "bali": {
                "name": "Bali, Indonesia",
                "description": "Island paradise known for lush rice terraces, beautiful temples, and vibrant culture.",
                "best_time": "April to October (dry season)",
                "activities": ["Surfing", "Temple visits", "Rice terrace trekking", "Yoga retreats", "Beach relaxation"],
                "avg_cost": "$100-500 per night depending on accommodation type"
            },
            "santorini": {
                "name": "Santorini, Greece",
                "description": "Stunning volcanic island with white-washed buildings and blue domes overlooking the Aegean Sea.",
                "best_time": "Late April to early November",
                "activities": ["Sunset watching in Oia", "Wine tasting", "Boat tours", "Beach hopping", "Ancient ruins exploration"],
                "avg_cost": "$200-600 per night for caldera view accommodations"
            }
        }
    
        @kernel_function(
            name="get_destination_info",
            description="Provides detailed information about specific travel destinations."
        )
        def get_destination_info(self, query: str) -> str:
            query_lower = query.lower()
            matching_destinations = []
    
            for key, details in DestinationsPlugin.DESTINATIONS.items():
                if key in query_lower or details["name"].lower() in query_lower:
                    matching_destinations.append(details)
    
            if not matching_destinations:
                return (f"User Query: {query}\n\n"
                        f"I couldn't find specific destination information in our database. "
                        f"Please use the general retrieval system for this query.")
    
            destination_info = "\n\n".join([
                f"Destination: {dest['name']}\n"
                f"Description: {dest['description']}\n"
                f"Best time to visit: {dest['best_time']}\n"
                f"Popular activities: {', '.join(dest['activities'])}\n"
                f"Average cost: {dest['avg_cost']}" for dest in matching_destinations
            ])
    
            return (f"Destination Information:\n{destination_info}\n\n"
                    f"User Query: {query}\n\n"
                    "Based on the above destination details, provide a helpful response "
                    "that addresses the user's query about this location.")
    
    """
    ## Setting Up ChromaDB
    
    To facilitate Retrieval-Augmented Generation, a persistent ChromaDB client is instantiated and a collection named `"travel_documents"` is created (or retrieved if it exists). This collection is then populated with sample travel documents and metadata.
    """
    logger.info("## Setting Up ChromaDB")
    
    collection = chromadb.PersistentClient(path="./chroma_db").create_collection(
        name="travel_documents",
        metadata={"description": "travel_service"},
        get_or_create=True,
    )
    
    documents = [
        "Contoso Travel offers luxury vacation packages to exotic destinations worldwide.",
        "Our premium travel services include personalized itinerary planning and 24/7 concierge support.",
        "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage.",
        "Popular destinations include the Maldives, Swiss Alps, and African safaris.",
        "Contoso Travel provides exclusive access to boutique hotels and private guided tours.",
    ]
    
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": "training", "type": "explanation"} for _ in documents]
    )
    
    agent = ChatCompletionAgent(
        service=chat_completion_service,
        plugins=[DestinationsPlugin(), WeatherInfoPlugin(), PromptPlugin(collection)],
        name="TravelAgent",
        instructions="Answer travel queries using the provided context. If context is provided, do not say 'I have no context for that.'",
    )
    
    """
    ### Running the Agent with Streaming Chat History
    The main asynchronous loop creates a chat history for the conversation and, for each user input, first adds the augmented prompt (as a system message) to the chat history so that the agent sees the retrieval context. The user message is also added, and then the agent is invoked using streaming. The output is printed as it streams in.
    """
    logger.info("### Running the Agent with Streaming Chat History")
    
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