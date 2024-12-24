import asyncio
import json
from jet.llm.similarity import get_similar_texts
from jet.llm.memory import Memory
from jet.logger import logger


def mock_event_emitter(event: dict):
    """
    Mock event emitter to simulate real-world usage of event handling.
    """
    print(f"Event Emitted: {event}")


def main():
    """
    Main function to demonstrate and test Memory functionalities.
    """
    # Sample memory id for testing
    memory_id = "test1"

    # Mock memories list with unique and similar entries
    overwrite = False
    mock_memories = [
        "User asked about AI-generated art tools",
        "User requested image editing software recommendations",
        "User inquired about AI-generated animation tools",
        "Discussion on SEO best practices",
        "Explored e-commerce platforms for product launches",
        "Asked for step-by-step deployment guides",
        "Searched for low-competition Instagram hashtags",
        "User discussed creating a Shopify store for a cosmetics brand",
        "SEO optimization for blogs was discussed",
        "Requested information on integrating Stripe with Webflow"
    ]
    queries = [
        "AI-generated art"
    ]

    # Initialize the Memory class
    memory_instance = Memory(memory_id, settings={
        "collections_settings": {
            "initial_data": mock_memories,
            "overwrite": overwrite,
        }
    })

    # Simulate getting similar memories
    similar_results = memory_instance.search(queries)
    logger.log("Similar Memories Results:")
    logger.success(json.dumps(similar_results, indent=2))


# Run the main function
if __name__ == "__main__":
    main()
