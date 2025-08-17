from pydantic import BaseModel
from autogen_core.models import LLMMessage
from autogen_core.memory._base_memory import ChatCompletionContext
from typing import List
import os
import shutil
import asyncio
from jet.llm.mlx.memory import MemoryManager
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.models import UserMessage

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


class ConcreteChatCompletionContext(ChatCompletionContext):
    """Concrete implementation of ChatCompletionContext."""

    def __init__(self, messages: List[LLMMessage]):
        super().__init__(messages)

    def get_messages(self) -> List[LLMMessage]:
        """Return the list of messages in the context."""
        return self._messages


async def main():
    # Initialize MemoryManager
    manager = MemoryManager(user_id="test_user", limit=5,
                            log_dir=f"{OUTPUT_DIR}/chats")

    # Example 1: Adding text memories with metadata
    print("\nExample 1: Adding text memories")
    await manager.add(
        MemoryContent(
            content="User prefers dark mode for apps.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"app": "example_app", "preference": "ui"}
        )
    )
    await manager.add(
        MemoryContent(
            content="User frequently searches for Python tutorials.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "learning", "language": "python"}
        )
    )
    print("Text memories added successfully")

    # Example 2: Querying memories
    print("\nExample 2: Querying memories")
    results = await manager.query("user preferences")
    for i, result in enumerate(results.results, 1):
        print(f"Result {i}: {result.content} (Metadata: {result.metadata})")

    # Example 3: Updating context with relevant memories
    print("\nExample 3: Updating context")
    context = ConcreteChatCompletionContext(
        messages=[UserMessage(
            content="What are my app preferences?", source="test_user")]
    )
    update_result = await manager.update_context(context)
    for i, memory in enumerate(update_result.memories.results, 1):
        print(
            f"Context Memory {i}: {memory.content} (Metadata: {memory.metadata})")

    # Example 4: Adding JSON memory
    print("\nExample 4: Adding JSON memory")
    json_content = {"settings": {"theme": "dark", "notifications": "enabled"}}
    await manager.add(
        MemoryContent(
            content=json_content,
            mime_type=MemoryMimeType.JSON,
            metadata={"type": "configuration"}
        )
    )
    json_results = await manager.query("app settings")
    for i, result in enumerate(json_results.results, 1):
        print(
            f"JSON Result {i}: {result.content} (Metadata: {result.metadata})")

    # Example 5: Clearing memories
    print("\nExample 5: Clearing memories")
    await manager.clear()
    cleared_results = await manager.query("user preferences")
    print(f"After clear, found {len(cleared_results.results)} memories")

    # Example 6: Closing memory manager
    print("\nExample 6: Closing memory manager")
    await manager.close()
    print("Memory manager closed successfully")

if __name__ == "__main__":
    asyncio.run(main())
