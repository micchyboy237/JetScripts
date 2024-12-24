import json
from jet.llm.memory import Memory
from jet.logger import logger


def create_memory_instance(memory_id):
    """Create an instance of Memory class with default settings."""
    memory_instance = Memory(memory_id)
    return memory_instance


def identify_user_preferences(memory_instance):
    """Example of identifying user preferences."""
    user_input = "I enjoy playing chess and reading science fiction novels."
    identified_memories = memory_instance.identify_memories(user_input)
    return identified_memories


def consolidate_existing_memories(memory_instance):
    """Example of consolidating overlapping or redundant memories."""
    existing_memories = json.dumps([
        {"fact": "User likes chess", "created_at": 1731464051},
        {"fact": "User enjoys science fiction novels", "created_at": 1731464108},
        {"fact": "User enjoys reading novels", "created_at": 1731464500}
    ])
    consolidated_memories = memory_instance.consolidate_memories(
        existing_memories)
    return consolidated_memories


def main():
    # Step 1: Create an instance of Memory
    memory_id = "test1"
    memory_instance = create_memory_instance(memory_id)
    logger.log("Memory instance created:", memory_id,
               colors=["DEBUG", "BRIGHT_SUCCESS"])
    logger.success(identified_memories)

    # Step 2: Identify user preferences based on input
    identified_memories = identify_user_preferences(memory_instance)
    logger.log("Identified Memories Result:")
    logger.success(identified_memories)

    # Step 3: Consolidate overlapping or redundant memories
    consolidated_memories = consolidate_existing_memories(memory_instance)
    logger.log("Consolidated Memories Result:")
    logger.success(consolidated_memories)


if __name__ == "__main__":
    main()
