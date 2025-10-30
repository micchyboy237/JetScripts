# %% [markdown]
# # Selecting Context in LangGraph
# 
# *Selecting context means pulling it into the context window to help an agent perform a task.*
# 
# ![Screenshot 2025-07-09 at 2.28.01 PM.png](attachment:da8d31d0-8a43-45bc-9784-570e68eca4e7.png)
# 
# ## Scratchpad
# 
# The mechanism for selecting context from a scratchpad depends upon how the scratchpad is implemented. If it’s a [tool](https://www.anthropic.com/engineering/claude-think-tool), then an agent can simply read it by making a tool call. If it’s part of the agent’s runtime state, then the developer can choose what parts of state to expose to an agent each step. This provides a fine-grained level of control for exposing context to an agent.
# 
# ### Scratchpad selecting in LangGraph
# 
# In `1_write_context.ipynb`, we saw how to write to the LangGraph state object. Now, we'll see how to select context from state and present it to an LLM call in a downstream node. This ability to select from state gives us control over what context we present to LLM calls. 

# %%
from typing import TypedDict

from rich.console import Console
from rich.pretty import pprint

from jet.visualization.terminal import display_iterm2_image

# Initialize console for rich formatting
console = Console()


class State(TypedDict):
    """State schema for the context selection workflow.
    
    Attributes:
        topic: The topic for joke generation
        joke: The generated joke content
    """
    topic: str
    joke: str

# %%

from langgraph.graph import END, START, StateGraph

# from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp
from langchain_openai import ChatOpenAI


# def _set_env(var: str) -> None:
#     """Set environment variable if not already set."""
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# # Set up environment and initialize model
# _set_env("ANTHROPIC_API_KEY")
# llm = ChatLlamaCpp(model="qwen3-instruct-2507:4b", temperature=0.0)
llm = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    verbosity="high",
)


def generate_joke(state: State) -> dict[str, str]:
    """Generate an initial joke about the topic.
    
    Args:
        state: Current state containing the topic
        
    Returns:
        Dictionary with the generated joke
    """
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def improve_joke(state: State) -> dict[str, str]:
    """Improve an existing joke by adding wordplay.
    
    This demonstrates selecting context from state - we read the existing
    joke from state and use it to generate an improved version.
    
    Args:
        state: Current state containing the original joke
        
    Returns:
        Dictionary with the improved joke
    """
    print(f"Initial joke: {state['joke']}")
    
    # Select the joke from state to present it to the LLM
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


# Build the workflow with two sequential nodes
workflow = StateGraph(State)

# Add both joke generation nodes
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)

# Connect nodes in sequence
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", "improve_joke")
workflow.add_edge("improve_joke", END)

# Compile the workflow
chain = workflow.compile()

# Display the workflow visualization
# display(Image(chain.get_graph().draw_mermaid_png()))

# Render and display graph in iTerm2
png_data = chain.get_graph().draw_mermaid_png()
display_iterm2_image(png_data)


# %% [markdown]
# ## Memory
# 
# If agents have the ability to save memories, they also need the ability to select memories relevant to the task they are performing. This can be useful for a few reasons. Agents might select few-shot examples ([episodic](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) [memories](https://arxiv.org/pdf/2309.02427)) for examples of desired behavior, instructions ([procedural](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) [memories](https://arxiv.org/pdf/2309.02427)) to steer behavior, or facts ([semantic](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) [memories](https://arxiv.org/pdf/2309.02427)) give the agent task-relevant context.
# 
# ![image (1).webp](attachment:2fc5dc77-8eba-4a80-8e38-ad00688adc3c.webp)
# 
# One challenge is ensure that relevant memories are selected. Some popular agents simply use a narrow set of files to store memories. For example, many code agent use “rules” files to save instructions (”procedural” memories) or, in some cases, examples (”episodic” memories). Claude Code uses [`CLAUDE.md`](http://CLAUDE.md). [Cursor](https://docs.cursor.com/context/rules) and [Windsurf](https://windsurf.com/editor/directory) use rules files. These are always pulled into context.
# 
# But, if an agent is storing a larger [collection](https://langchain-ai.github.io/langgraph/concepts/memory/#collection) of facts and / or relationships ([semantic](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) memories), selection is harder. [ChatGPT](https://help.openai.com/en/articles/8590148-memory-faq) is a good example of this. At the AIEngineer World’s Fair, [Simon Willison shared](https://simonwillison.net/2025/Jun/6/six-months-in-llms/) a good example of memory selection gone wrong: ChatGPT fetched his location and injected it into an image that he requested. This type of erroneous memory retrieval can make users feel like the context winder “*no longer belongs to them*”! Use of embeddings and / or [knowledge](https://arxiv.org/html/2501.13956v1#:~:text=In%20Zep%2C%20memory%20is%20powered,subgraph%2C%20and%20a%20community%20subgraph) [graphs](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/#:~:text=changes%20since%20updates%20can%20trigger,and%20holistic%20memory%20for%20agentic) for indexing of memories have been used to assist with selection.
# 
# ### Memory selecting in LangGraph
# 
# In `1_write_context.ipynb`, we saw how to write to `InMemoryStore` in graph nodes. Now let's select state from it. We can use the [get](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-storage) method to select context from state.

# %%
from langgraph.store.memory import InMemoryStore
from jet.models.utils import get_embedding_size
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding

embeddings = LlamacppEmbedding(model="embeddinggemma")

# Initialize the memory store
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": get_embedding_size("embeddinggemma"),
    }
)

# Define namespace for organizing memories
namespace = ("rlm", "joke_generator")

# %%
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# Initialize storage components
checkpointer = InMemorySaver()
memory_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": get_embedding_size("embeddinggemma"),
    }
)


def generate_joke(state: State, store: BaseStore) -> dict[str, str]:
    """Generate a joke with memory-aware context selection.
    
    This function demonstrates selecting context from memory before
    generating new content, ensuring consistency and avoiding duplication.
    
    Args:
        state: Current state containing the topic
        store: Memory store for persistent context
        
    Returns:
        Dictionary with the generated joke
    """
    # Select prior joke from memory if it exists
    prior_joke = store.get(namespace, "last_joke")
    if prior_joke:
        prior_joke_text = prior_joke.value["joke"]
        print(f"Prior joke: {prior_joke_text}")
    else:
        print("Prior joke: None!")

    # Generate a new joke that differs from the prior one
    prompt = (
        f"Write a short joke about {state['topic']}, "
        f"but make it different from any prior joke you've written: {prior_joke_text if prior_joke else 'None'}"
    )
    msg = llm.invoke(prompt)

    # Store the new joke in memory for future context selection
    store.put(namespace, "last_joke", {"joke": msg.content})

    return {"joke": msg.content}


# Build the memory-aware workflow
workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)

# Connect the workflow
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)

# Compile with both checkpointing and memory store
chain = workflow.compile(checkpointer=checkpointer, store=memory_store)

# %%
# Execute the workflow with the first thread
config = {"configurable": {"thread_id": "1"}}
joke_generator_state = chain.invoke({"topic": "cats"}, config)

# %%
# Get the latest state of the graph
latest_state = chain.get_state(config)

console.print("\n[bold magenta]Latest Graph State:[/bold magenta]")
pprint(latest_state)

# %% [markdown]
# We fetch the prior joke from memory and pass it to an LLM to improve it!


# %% [markdown]
# ## Tools
# 
# Agents use tools, but can become overloaded if they are provided with too many. This is often because the tool descriptions can overlap, causing model confusion about which tool to use. One approach is to apply RAG to tool descriptions in order to fetch the most relevant tools for a task based upon semantic similarity, an idea that Drew Breunig calls “[tool loadout](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html).” Some [recent papers](https://arxiv.org/abs/2505.03275) have shown that this improve tool selection accuracy by 3-fold.
# 
# ### Tool selecting in LangGraph
# 
# For tool selection, the [LangGraph Bigtool](https://github.com/langchain-ai/langgraph-bigtool) library is a great way to apply semantic similarity search over tool descriptions for selection of the most relevant tools for a task. It leverages LangGraph's long-term memory store to allow an agent to search for and retrieve relevant tools for a given problem. Lets demonstrate `langgraph-bigtool` by equipping an agent with all functions from Python's built-in math library.

# %%
import math
import types
import uuid

from langgraph.store.base import PutOp
from langgraph.store.memory import InMemoryStore

# from langgraph_bigtool import create_agent
# from langgraph_bigtool.utils import (
#     convert_positional_only_function_to_tool
# )
from langchain_core.tools import StructuredTool

from jet.models.utils import get_embedding_size

# _set_env("OPENAI_API_KEY")

import inspect

from langgraph.store.memory import InMemoryStore

# --- SAFE TOOL WRAPPER ---
def safe_tool_from_function(func):
    """Create StructuredTool with sanitized defaults to avoid schema errors."""
    try:
        sig = inspect.signature(func)
    except ValueError:
        return None

    # Skip *args / **kwargs
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in sig.parameters.values()):
        return None

    params = {}
    for name, param in sig.parameters.items():
        default = param.default
        if default is not param.empty:
            if default is None:
                default = None
            elif isinstance(default, float) and (default == 0.0 or abs(default) < 1e-6):
                default = float(default)
            elif isinstance(default, (int, float, str, bool)):
                default = default
            else:
                default = None
        params[name] = (param.annotation if param.annotation != param.empty else str, default)

    import pydantic
    fields = {
        name: (annotation, pydantic.Field(default=default) if default is not None else ...)
        for name, (annotation, default) in params.items()
    }
    ArgsSchema = pydantic.create_model(f"{func.__name__.capitalize()}Args", **fields)

    # ---- NEW: positional-only wrapper ----
    def positional_wrapper(**kwargs):
        # Order must match the original signature
        bound = sig.bind(**kwargs)
        bound.apply_defaults()
        return func(*bound.args)

    return StructuredTool(
        name=func.__name__,
        description=getattr(func, "__doc__", "") or f"Call {func.__name__}",
        args_schema=ArgsSchema,
        func=positional_wrapper,          # <-- use wrapper
    )

# --- BUILD TOOLS SAFELY ---
all_tools = []
for function_name in dir(math):
    function = getattr(math, function_name)
    if not isinstance(function, types.BuiltinFunctionType):
        continue
    tool = safe_tool_from_function(function)
    if tool is not None:
        all_tools.append(tool)

# Create registry of tools. This is a dict mapping
# identifiers to tool instances.
tool_registry = {str(uuid.uuid4()): tool for tool in all_tools}

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": get_embedding_size("embeddinggemma"),
        "fields": ["description"],
    }
)

console.print(f"[bold pink]Tool Registry ({len(tool_registry)}):[/bold pink]")
put_ops = []
for tool_idx, (tool_id, tool) in enumerate(tool_registry.items()):
    pprint(f"{tool_idx + 1}: {tool.name}: {tool.description}")
    put_ops.append(
        PutOp(
            namespace=("tools",),
            key=tool_id,
            value={"description": f"{tool.name}: {tool.description}"},
        )
    )

batch_results = store.batch(put_ops)

# %%
from jet.adapters.langchain.chat_agent_utils import build_agent

# Initialize agent
# builder = create_agent(llm, tool_registry)
# agent = builder.compile(store=store)
agent = build_agent(all_tools, llm)

# Display the agent visualization
# display(Image(agent.get_graph().draw_mermaid_png()))

# Render and display graph in iTerm2
png_data = agent.get_graph().draw_mermaid_png()
display_iterm2_image(png_data)

# %%
query = "Use available tools to calculate arc cosine of 0.5."
result = agent.invoke({"messages": query})

console.print("\n[bold blue]Agent tool result:[/bold blue]")
pprint(result)
