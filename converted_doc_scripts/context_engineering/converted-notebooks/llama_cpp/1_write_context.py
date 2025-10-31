# JetScripts/converted_doc_scripts/context_engineering/converted-notebooks/llama_cpp/1_write_context.py
from typing import TypedDict
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from rich.console import Console
from rich.pretty import pprint
from jet.visualization.terminal import display_iterm2_image
import os
import shutil
from jet.logger import logger

# === Setup output and logging ===
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_file = f"{OUTPUT_DIR}/main.log"
logger.basicConfig(filename=log_file)
logger.orange(f"Main logs: {log_file}")

console = Console()

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

# === Build and visualize basic workflow ===
workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)
chain = workflow.compile()

png_data = render_mermaid_graph(
    chain, output_filename=f"{OUTPUT_DIR}/joke_generator_graph.png")
display_iterm2_image(png_data)

# === Run basic workflow ===
joke_generator_state = chain.invoke({"topic": "cats"})
console.print("\n[bold blue]Joke Generator State:[/bold blue]")
pprint(joke_generator_state)

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
store.put(namespace, "last_joke", {"joke": joke_generator_state["joke"]})

stored_items = list(store.search(namespace))
console.print("\n[bold green]Stored Items in Memory:[/bold green]")
pprint(stored_items)

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

def generate_joke(state: State, store: BaseStore) -> dict[str, str]:
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
        console.print(f"[dim]Existing joke:[/dim] {existing_joke['joke']}")
    else:
        console.print("[dim]Existing joke:[/dim] No existing joke")

    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    store.put(namespace, "last_joke", {"joke": msg.content})
    return {"joke": msg.content}

# === Build memory-aware workflow ===
workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)
chain = workflow.compile(checkpointer=checkpointer, store=memory_store)

png_data = render_mermaid_graph(
    chain, output_filename=f"{OUTPUT_DIR}/joke_memory_graph.png")
display_iterm2_image(png_data)

# === Run with thread isolation ===
config1 = {"configurable": {"thread_id": "1"}}
joke_generator_state1 = chain.invoke({"topic": "cats"}, config1)
console.print("\n[bold cyan]Workflow Result (Thread 1):[/bold cyan]")
pprint(joke_generator_state1)

latest_state = chain.get_state(config1)
console.print("\n[bold magenta]Latest Graph State (Thread 1):[/bold magenta]")
pprint(latest_state)

config2 = {"configurable": {"thread_id": "2"}}
joke_generator_state2 = chain.invoke({"topic": "cats"}, config2)
console.print("\n[bold yellow]Workflow Result (Thread 2):[/bold yellow]")
pprint(joke_generator_state2)