import logging
import os

from mem0 import Memory
from mem0.configs.base import MemoryConfig

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════

# Your environment variables (with defaults for local llama.cpp)
LLM_BASE_URL = os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:8080/v1")
LLM_MODEL = os.getenv("LLAMA_CPP_LLM_MODEL", "llama-3-8b")
EMBED_MODEL = os.getenv("LLAMA_CPP_EMBED_MODEL", "nomic-embed-text")
EMBED_BASE_URL = os.getenv("LLAMA_CPP_EMBED_URL", "http://localhost:8080/v1")
EMBED_DIMS = int(os.getenv("LLAMA_CPP_EMBED_DIMS", "768"))

# Build the config
config_dict = {
    "llm": {
        "provider": "openai",  # llama.cpp is OpenAI-compatible
        "config": {
            "model": LLM_MODEL,
            "api_key": "not-needed",  # llama.cpp doesn't require auth
            "openai_base_url": LLM_BASE_URL,  # Points to your llama.cpp server
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": EMBED_MODEL,
            "api_key": "not-needed",
            "openai_base_url": EMBED_BASE_URL,
            "embedding_dims": EMBED_DIMS,
        },
    },
    "vector_store": {
        "provider": "faiss",  # Local file-based vector store (no server needed)
        "config": {
            "collection_name": "my_memories",
            "embedding_model_dims": EMBED_DIMS,
            "path": "./mem0_faiss_store",
        },
    },
}

# Create memory instance with custom config
config = MemoryConfig(**config_dict)
m = Memory(config)

# ═══════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════

USER_ID = "demo_user"


def print_section(title: str):
    """Helper to print formatted section headers."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def display_results(results):
    """Helper to display memory results cleanly."""
    if not results.get("results"):
        print("  (no results)")
        return
    for item in results["results"]:
        print(
            f"  [{item.get('event', 'N/A')}] {item.get('id', 'N/A')[:8]}... → {item.get('memory', 'N/A')}"
        )


# --- 1. Add Memories ---
print_section("1. ADDING MEMORIES")
print("  Adding travel preferences...")
add_result = m.add(
    [
        {
            "role": "user",
            "content": "I love visiting Japan. The food and culture are amazing.",
        },
        {
            "role": "assistant",
            "content": "Japan is wonderful! What's your favorite dish?",
        },
        {
            "role": "user",
            "content": "Definitely sushi and ramen. I also enjoy hiking near Mt. Fuji.",
        },
    ],
    user_id=USER_ID,
)
display_results(add_result)

print("\n  Adding work information...")
m.add(
    "I'm a software engineer working remotely, and I use Python daily.",
    user_id=USER_ID,
)

# --- 2. Get All Memories ---
print_section("2. GET ALL MEMORIES")
all_memories = m.get_all(filters={"user_id": USER_ID})
print(f"  Found {len(all_memories['results'])} memories for {USER_ID}:")
display_results(all_memories)

# --- 3. Get Single Memory ---
print_section("3. GET SINGLE MEMORY")
if all_memories["results"]:
    first_id = all_memories["results"][0]["id"]
    print(f"  Fetching memory: {first_id}")
    single = m.get(first_id)
    if single:
        print(f"  Memory: {single['memory']}")
        print(f"  Created: {single['created_at']}")

# --- 4. Search Memories ---
print_section("4. SEARCHING MEMORIES")

print("\n  Search: 'food and travel'")
search_result = m.search(
    "food and travel",
    filters={"user_id": USER_ID},
    top_k=5,
)
display_results(search_result)

print("\n  Search: 'programming skills'")
search_result = m.search(
    "programming skills",
    filters={"user_id": USER_ID},
    top_k=5,
)
display_results(search_result)

# --- 5. Update a Memory ---
print_section("5. UPDATING A MEMORY")
if all_memories["results"]:
    target = all_memories["results"][0]
    print(f"  Before: {target['memory']}")
    m.update(
        target["id"],
        data=f"{target['memory']} I'm also learning Rust.",
    )
    updated = m.get(target["id"])
    print(f"  After:  {updated['memory']}")

# --- 6. History ---
print_section("6. MEMORY HISTORY")
if all_memories["results"]:
    history = m.history(all_memories["results"][0]["id"])
    print("  Change history for first memory:")
    for entry in history:
        print(f"    [{entry['event']}] {entry['new_memory']}")

# --- 7. Delete a Memory ---
print_section("7. DELETE A MEMORY")
if len(all_memories["results"]) > 1:
    to_delete = all_memories["results"][1]["id"]
    print(f"  Deleting: {to_delete}")
    m.delete(to_delete)
    remaining = m.get_all(filters={"user_id": USER_ID})
    print(f"  Remaining memories: {len(remaining['results'])}")

# --- Cleanup ---
print_section("CLEANUP")
print("  Resetting memory store...")
m.reset()
print("  Done! All memories cleared.")
m.close()

print(f"\n{'=' * 60}")
print("  DEMO COMPLETE")
print(f"{'=' * 60}")
