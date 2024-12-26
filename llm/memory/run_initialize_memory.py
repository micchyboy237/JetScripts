from datetime import datetime
from jet.llm.memory import Memory
from jet.logger import logger


def setup_initial_memory():
    """Sets up initial memory data."""
    initial_data = [
        {
            "id": "1",
            "document": "User is a software developer specializing in AI",
            "metadata": {"created_at": datetime.utcfromtimestamp(1731464051).isoformat()}
        },
        {
            "id": "2",
            "document": "User enjoys playing the guitar on weekends",
            "metadata": {"created_at": datetime.utcfromtimestamp(1731459051).isoformat()}
        },
        {
            "id": "3",
            "document": "User frequently hikes in the mountains",
            "metadata": {"created_at": datetime.utcfromtimestamp(1731463151).isoformat()}
        },
    ]
    return initial_data


def usage_example_identify_memories(memory_instance: Memory):
    """Example: Identifying memories based on input text."""
    input_text = "I enjoy painting and sketching in my free time."
    identified_memories = memory_instance.identify_memories(input_text)
    print("Identified Memories:", identified_memories)


def usage_example_consolidate_memories(memory_instance: Memory):
    """Example: Consolidating overlapping or similar memories."""
    existing_memories = '[{"document": "User enjoys sketching", "metadata": {"created_at": "2024-12-24T12:34:11Z"}}, {"document": "User enjoys painting and sketching", "metadata": {"created_at": "2024-12-24T12:35:08Z"}}]'
    consolidated_memories = memory_instance.consolidate_memories(
        existing_memories)
    print("Consolidated Memories:", consolidated_memories)


async def usage_example_store_memory(memory_instance: Memory):
    """Example: Storing new memories."""
    memory_text = "User recently started learning French."
    await memory_instance.store_memory(memory_text)
    print(f"Memory '{memory_text}' stored successfully.")


async def main():
    """Main function demonstrating the Memory class usage."""
    settings = {
        "model": "llama3.1",
        "db_path": "data/vector_db",
        "related_memories_n": 10,
        "related_memories_dist": 0.7,
        "collections_settings": {
            "overwrite": True,
            "initial_data": setup_initial_memory(),
        },
    }
    memory_instance = Memory(memory_id="test1", settings=settings)

    # Demonstrate usage examples
    usage_example_identify_memories(memory_instance)
    usage_example_consolidate_memories(memory_instance)
    await usage_example_store_memory(memory_instance)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
