import os
import shutil
import asyncio
from jet.llm.mlx.memory import MemoryManager
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.models import UserMessage

from jet.llm.mlx.memory import ConcreteChatCompletionContext, MemoryList

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main():
    # Initialize MemoryManager and MemoryList
    memory_manager = MemoryManager(
        user_id="test_user", limit=5, log_dir=f"{OUTPUT_DIR}/chats")
    memory_list = MemoryList(name="recent_memory",
                             memory_manager=memory_manager)

    # Example 1: Adding text memories to MemoryList
    print("\nExample 1: Adding text memories to MemoryList")
    await memory_list.add(
        MemoryContent(
            content="User prefers dark mode for web apps.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"context": "ui", "priority": "high"}
        )
    )
    await memory_list.add(
        MemoryContent(
            content="User is learning Python for data science.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"context": "learning", "language": "python"}
        )
    )
    print("Memories added to MemoryList successfully")

    # Example 2: Querying MemoryList with local filter
    print("\nExample 2: Querying MemoryList with 'dark mode' filter (local)")
    list_results = await memory_list.query(
        MemoryContent(content="dark mode", mime_type=MemoryMimeType.TEXT)
    )
    for i, result in enumerate(list_results.results, 1):
        print(
            f"Local Result {i}: {result.content} (Metadata: {result.metadata})")

    # Example 3: Querying MemoryList with MemoryManager (vector search)
    print("\nExample 3: Querying MemoryList with 'dark mode' (via MemoryManager)")
    manager_results = await memory_list.query("dark mode")
    for i, result in enumerate(manager_results.results, 1):
        print(
            f"Vector Result {i}: {result.content} (Metadata: {result.metadata})")

    # Example 4: Updating context with MemoryList
    print("\nExample 4: Updating context with MemoryList")
    context = ConcreteChatCompletionContext(
        messages=[UserMessage(
            content="What are my UI preferences?", source="test_user")]
    )
    update_result = await memory_list.update_context(context)
    for i, result in enumerate(update_result.memories.results, 1):
        print(
            f"Context Memory {i}: {result.content} (Metadata: {result.metadata})")

    # Example 5: Adding JSON memory to MemoryList
    print("\nExample 5: Adding JSON memory to MemoryList")
    json_content = {"settings": {"theme": "dark", "notifications": "enabled"}}
    await memory_list.add(
        MemoryContent(
            content=json_content,
            mime_type=MemoryMimeType.JSON,
            metadata={"type": "configuration", "priority": "high"}
        )
    )
    json_results = await memory_list.query("theme")
    for i, result in enumerate(json_results.results, 1):
        print(
            f"JSON Result {i}: {result.content} (Metadata: {result.metadata})")

    # Example 6: Clearing MemoryList
    print("\nExample 6: Clearing MemoryList")
    await memory_list.clear()
    cleared_results = await memory_list.query("dark mode")
    print(
        f"After clearing MemoryList, found {len(cleared_results.results)} memories")

    # Example 7: Closing MemoryList and MemoryManager
    print("\nExample 7: Closing MemoryList and MemoryManager")
    await memory_list.close()
    await memory_manager.close()
    print("MemoryList and MemoryManager closed successfully")

if __name__ == "__main__":
    asyncio.run(main())
