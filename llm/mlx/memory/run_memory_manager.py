import os
import shutil
import asyncio

from jet.llm.mlx.memory import MemoryManager

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main():
    # Initialize the manager with local model paths
    manager = MemoryManager(user_id="test_user", limit=5,
                            log_dir=f"{OUTPUT_DIR}/chats")

    # Add a memory
    await manager.add("User prefers dark mode for apps.", {"app": "example_app"})
    print("Memory added successfully")

    # Search memories
    results = await manager.search("user preferences")
    for i, result in enumerate(results, 1):
        print(
            f"Result {i}: {result['content']} (Score: {result['metadata'].get('score', 0):.4f})")

if __name__ == "__main__":
    asyncio.run(main())
