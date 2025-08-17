import os
import shutil
import asyncio
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.models import UserMessage

from jet.llm.mlx.memory import ConcreteChatCompletionContext, MemoryList

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main():
    # Initialize MemoryList
    memory = MemoryList(name="user_memory")

    # Example 1: Adding text memories with metadata
    print("\nExample 1: Adding text memories")
    await memory.add(
        MemoryContent(
            content="User likes dark mode UI.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"context": "ui"}
        )
    )
    await memory.add(
        MemoryContent(
            content="User prefers light mode on mobile.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"context": "mobile"}
        )
    )
    await memory.add(
        MemoryContent(
            content="User enjoys Python programming.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"context": "programming"}
        )
    )
    print("Memories added successfully")

    # Example 2: Querying memories with filter
    print("\nExample 2: Querying memories with 'dark mode' filter")
    results = await memory.query("dark mode")
    for i, result in enumerate(results.results, 1):
        print(f"Result {i}: {result.content} (Metadata: {result.metadata})")

    # Example 3: Querying memories with no filter
    print("\nExample 3: Querying all memories (no filter)")
    results = await memory.query("")
    for i, result in enumerate(results.results, 1):
        print(f"Result {i}: {result.content} (Metadata: {result.metadata})")

    # Example 4: Updating context with relevant memories
    print("\nExample 4: Updating context")
    context = ConcreteChatCompletionContext(
        messages=[UserMessage(
            content="What are my UI preferences?", source="test_user")]
    )
    update_result = await memory.update_context(context)
    for i, result in enumerate(update_result.memories.results, 1):
        print(
            f"Context Memory {i}: {result.content} (Metadata: {result.metadata})")

    # Example 5: Adding JSON memory
    print("\nExample 5: Adding JSON memory")
    json_content = {"settings": {"theme": "dark", "notifications": "enabled"}}
    await memory.add(
        MemoryContent(
            content=json_content,
            mime_type=MemoryMimeType.JSON,
            metadata={"type": "configuration"}
        )
    )
    json_results = await memory.query("theme")
    for i, result in enumerate(json_results.results, 1):
        print(
            f"JSON Result {i}: {result.content} (Metadata: {result.metadata})")

    # Example 6: Clearing memories
    print("\nExample 6: Clearing memories")
    await memory.clear()
    cleared_results = await memory.query("dark mode")
    print(f"After clear, found {len(cleared_results.results)} memories")

    # Example 7: Closing memory
    print("\nExample 7: Closing memory")
    await memory.close()
    print("Memory closed successfully")

if __name__ == "__main__":
    asyncio.run(main())
