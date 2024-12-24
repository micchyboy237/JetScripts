from jet.llm.memory import Memory
from jet.logger import logger


def setup_initial_memory():
    """Sets up initial memory data."""
    initial_data = [
        {"fact": "User is a software developer specializing in AI",
            "created_at": 1731464051},
        {"fact": "User enjoys playing the guitar on weekends", "created_at": 1731459051},
        {"fact": "User frequently hikes in the mountains", "created_at": 1731463151},
    ]
    return initial_data


def usage_example_identify_memories(memory_instance: Memory):
    """Example: Identifying memories based on input text."""
    input_text = "I enjoy painting and sketching in my free time."
    identified_memories = memory_instance.identify_memories(input_text)
    print("Identified Memories:", identified_memories)


def usage_example_consolidate_memories(memory_instance: Memory):
    """Example: Consolidating overlapping or similar memories."""
    existing_memories = '[{"fact": "User enjoys sketching", "created_at": 1731464051}, {"fact": "User enjoys painting and sketching", "created_at": 1731464108}]'
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
        "model": "llama3.1:latest",
        "db_path": "data/vector_db",
        "related_memories_n": 10,
        "related_memories_dist": 0.7,
        "collections_settings": {
            "overwrite": True,
            "metadata": {"project": "new_memory_project"},
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
