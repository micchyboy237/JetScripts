import json
import logging
from pathlib import Path
# JetScripts/converted_doc_scripts/context_engineering/converted-notebooks/llama_cpp/1_write_context.py
import shutil
from typing import TypedDict
from jet.transformers.formatters import format_json
from jet.transformers.object import make_serializable
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from jet.visualization.terminal import display_iterm2_image
import os
from jet.logger import logger

BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)

# === State definition ===
class State(TypedDict):
    """State schema for the joke generator workflow.
    Attributes:
        topic: The topic for joke generation
        joke: The generated joke content
    """
    topic: str
    joke: str

# === Imports and LLM setup ===
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    verbosity="high",
)

# === Simple joke generation node ===
def generate_joke(state: State) -> dict[str, str]:
    """Generate a joke about the specified topic.
    Args:
        state: Current state containing the topic
    Returns:
        Dictionary with the generated joke
    """
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

# === Memory store setup with EmbedLlamaCpp ===
from langgraph.store.memory import InMemoryStore
from jet.adapters.langchain.embed_llama_cpp import EmbedLlamaCpp
from jet.models.utils import get_embedding_size

embeddings = EmbedLlamaCpp(model="embeddinggemma")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": get_embedding_size("embeddinggemma"),
    }
)

namespace = ("rlm", "joke_generator")

# === Memory-aware joke generation with checkpointer ===
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore

checkpointer = InMemorySaver()
memory_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": get_embedding_size("embeddinggemma"),
    }
)

def generate_joke_with_memory(state: State, store: BaseStore) -> dict[str, str]:
    """Generate a joke with memory awareness.
    Checks for existing jokes in memory before generating new ones.
    Args:
        state: Current state containing the topic
        store: Memory store for persistent context
    Returns:
        Dictionary with the generated joke
    """
    existing_jokes = list(store.search(namespace))
    if existing_jokes:
        existing_joke = existing_jokes[0].value
        logger.debug(f"Existing joke: {existing_joke['joke']}")
    else:
        logger.debug("Existing joke: No existing joke")

    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    store.put(namespace, "last_joke", {"joke": msg.content})
    return {"joke": msg.content}

def example_1_basic_joke_generation():
    """Example 1: Simple joke generation with graph visualization."""
    example_dir = Path(BASE_OUTPUT_DIR) / "example_1_basic_joke_generation"
    example_dir.mkdir(parents=True, exist_ok=True)
    log_file = example_dir / "main.log"
    logger.basicConfig(filename=log_file, level=logging.INFO, force=True)
    logger.orange(f"Example 1 logs: {log_file}")

    workflow = StateGraph(State)
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_edge(START, "generate_joke")
    workflow.add_edge("generate_joke", END)
    chain = workflow.compile()

    png_path = example_dir / "joke_generator_graph.png"
    png_data = render_mermaid_graph(chain, output_filename=str(png_path))
    display_iterm2_image(png_data)

    result = chain.invoke({"topic": "cats"})
    (example_dir / "result.json").write_text(json.dumps(result, indent=2))

    logger.blue("\nExample 1 - Joke Generator State:")
    logger.success(format_json(result))

def example_2_memory_store_write_read():
    """Example 2: Write joke to memory store and retrieve it."""
    example_dir = Path(BASE_OUTPUT_DIR) / "example_2_memory_store_write_read"
    example_dir.mkdir(parents=True, exist_ok=True)
    log_file = example_dir / "main.log"
    logger.basicConfig(filename=log_file, level=logging.INFO, force=True)
    logger.orange(f"Example 2 logs: {log_file}")

    # Use result from example 1
    prior_result_path = Path(BASE_OUTPUT_DIR) / "example_1_basic_joke_generation" / "result.json"
    if prior_result_path.exists():
        prior_joke = json.loads(prior_result_path.read_text())["joke"]
    else:
        prior_joke = "Why did the cat sit on the computer? Because it wanted to keep an eye on the mouse!"

    store.put(namespace, "last_joke", {"joke": prior_joke})
    stored_items = list(store.search(namespace))

    (example_dir / "stored_items.json").write_text(json.dumps([item.value for item in stored_items], indent=2))

    logger.green("\nExample 2 - Stored Items in Memory:")
    logger.success(format_json(stored_items))

def example_3_thread_isolated_memory():
    """Example 3: Thread-isolated joke generation with memory."""
    example_dir = Path(BASE_OUTPUT_DIR) / "example_3_thread_isolated_memory"
    example_dir.mkdir(parents=True, exist_ok=True)
    log_file = example_dir / "main.log"
    logger.basicConfig(filename=log_file, level=logging.INFO, force=True)
    logger.orange(f"Example 3 logs: {log_file}")

    workflow = StateGraph(State)
    workflow.add_node("generate_joke", generate_joke_with_memory)
    workflow.add_edge(START, "generate_joke")
    workflow.add_edge("generate_joke", END)
    chain = workflow.compile(checkpointer=checkpointer, store=memory_store)

    png_path = example_dir / "joke_memory_graph.png"
    png_data = render_mermaid_graph(chain, output_filename=str(png_path))
    display_iterm2_image(png_data)

    config1 = {"configurable": {"thread_id": "1"}}
    result1 = chain.invoke({"topic": "cats"}, config1)
    (example_dir / "thread1_result.json").write_text(json.dumps(result1, indent=2))

    latest_state = chain.get_state(config1)
    (example_dir / "thread1_latest_state.json").write_text(json.dumps(make_serializable(latest_state), indent=2, default=str))

    config2 = {"configurable": {"thread_id": "2"}}
    result2 = chain.invoke({"topic": "cats"}, config2)
    (example_dir / "thread2_result.json").write_text(json.dumps(result2, indent=2))

    logger.cyan("\nExample 3 - Thread 1:")
    logger.success(format_json(result1))
    logger.magenta("\nExample 3 - Thread 1 Latest State:")
    logger.success(format_json(latest_state))
    logger.yellow("\nExample 3 - Thread 2:")
    logger.success(format_json(result2))

if __name__ == "__main__":
    logger.magenta("Running 1_write_context.py examples...")
    example_1_basic_joke_generation()
    example_2_memory_store_write_read()
    example_3_thread_isolated_memory()
    logger.green("All examples completed.")