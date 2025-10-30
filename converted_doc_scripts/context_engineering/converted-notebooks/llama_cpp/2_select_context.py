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

from jet.visualization.terminal import display_iterm2_image
from rich.console import Console
from rich.pretty import pprint

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
png_data = chain.get_graph().draw_mermaid_png()
display_iterm2_image(png_data)

# %%
# # Execute the workflow to see context selection in action
# joke_generator_state = chain.invoke({"topic": "cats"})

# # Display the final state with rich formatting
# console.print("\n[bold pink]Joke Generator State:[/bold pink]")
# pprint(joke_generator_state)

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
from jet.adapters.langchain.embed_llama_cpp import EmbedLlamaCpp

embeddings = EmbedLlamaCpp(model="embeddinggemma")

# Initialize the memory store
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": get_embedding_size("embeddinggemma"),
    }
)

# Define namespace for organizing memories
namespace = ("rlm", "joke_generator")

# # Store the generated joke in memory
# store.put(
#     namespace,                             # namespace for organization
#     "last_joke",                          # key identifier
#     {"joke": joke_generator_state["joke"]} # value to store
# )

# # Select (retrieve) the joke from memory
# retrieved_joke = store.get(namespace, "last_joke").value

# # Display the retrieved context
# console.print("\n[bold green]Retrieved Context from Memory:[/bold green]")
# pprint(retrieved_joke)

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
# # Execute the workflow with the first thread
# config = {"configurable": {"thread_id": "1"}}
# joke_generator_state = chain.invoke({"topic": "cats"}, config)

# %%
# # Get the latest state of the graph
# latest_state = chain.get_state(config)

# console.print("\n[bold magenta]Latest Graph State:[/bold magenta]")
# pprint(latest_state)

# %% [markdown]
# We fetch the prior joke from memory and pass it to an LLM to improve it!

# %%
# # Execute the workflow with a second thread to demonstrate memory persistence
# config = {"configurable": {"thread_id": "2"}}
# joke_generator_state = chain.invoke({"topic": "cats"}, config)
# console.print("\n[bold pink]Memory persistence demo result:[/bold pink]")
# pprint(joke_generator_state)

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
import inspect
import pydantic
from langchain_core.tools import StructuredTool
from typing import Any

def safe_tool_from_function(func) -> StructuredTool | None:
    """Create StructuredTool from math builtin, safely handling missing signatures."""
    # Skip non-functions
    if not isinstance(func, types.BuiltinFunctionType):
        return None

    # --- Critical: Skip functions with no inspectable signature ---
    try:
        sig = inspect.signature(func)
    except ValueError:
        # Common for math.hypot, math.ldexp, math.frexp, etc.
        return None

    # Build parameter fields
    fields: dict[str, tuple[Any, Any]] = {}
    for param in sig.parameters.values():
        name = param.name
        anno = param.annotation if param.annotation is not param.empty else float

        if param.default is param.empty:
            fields[name] = (anno, ...)
        else:
            default = param.default
            # Sanitize defaults to JSON-serializable
            if default is None:
                default = None
            elif isinstance(default, (int, str, bool)):
                default = default
            elif isinstance(default, float):
                default = float(default)
            else:
                default = None  # e.g., math.nan → None
            fields[name] = (anno, pydantic.Field(default=default))

    # Create schema
    try:
        ArgsSchema = pydantic.create_model(
            f"{func.__name__.capitalize()}Args",
            **fields
        )
    except Exception as e:
        print(f"[WARN] Failed to create schema for {func.__name__}: {e}")
        return None

    # Wrapper that respects original call order
    def wrapper(**kwargs):
        # Keep only params that exist in the signature
        clean_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        args_list = []
        kwargs_dict = {}

        for param in sig.parameters.values():
            name = param.name
            kind = param.kind

            if kind is inspect.Parameter.POSITIONAL_ONLY:
                # positional-only: must be passed positionally
                if name in clean_kwargs:
                    args_list.append(clean_kwargs.pop(name))
                elif param.default is not param.empty:
                    args_list.append(param.default)
                else:
                    # required positional-only missing
                    raise TypeError(f"Missing required positional-only argument: {name}")

            elif kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                # prefer positional if provided in clean_kwargs
                if name in clean_kwargs:
                    args_list.append(clean_kwargs.pop(name))
                elif param.default is not param.empty:
                    args_list.append(param.default)
                else:
                    raise TypeError(f"Missing required argument: {name}")

            elif kind is inspect.Parameter.VAR_POSITIONAL:
                # accept a provided iterable under the param name, else nothing
                if name in clean_kwargs:
                    var_val = clean_kwargs.pop(name)
                    if not isinstance(var_val, (list, tuple)):
                        raise TypeError(f"VAR_POSITIONAL param '{name}' must be a list/tuple")
                    args_list.extend(var_val)

            elif kind is inspect.Parameter.KEYWORD_ONLY:
                if name in clean_kwargs:
                    kwargs_dict[name] = clean_kwargs.pop(name)
                elif param.default is not param.empty:
                    kwargs_dict[name] = param.default
                else:
                    raise TypeError(f"Missing required keyword-only argument: {name}")

            elif kind is inspect.Parameter.VAR_KEYWORD:
                # capture any remaining extras later
                pass

        # Any leftover keys should go into **kwargs if function accepts VAR_KEYWORD
        accepts_varkw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if clean_kwargs:
            if accepts_varkw:
                kwargs_dict.update(clean_kwargs)
            else:
                # unknown/extra parameters provided
                unexpected = ", ".join(clean_kwargs.keys())
                raise TypeError(f"Got unexpected keyword arguments: {unexpected}")

        return func(*args_list, **kwargs_dict)

    return StructuredTool(
        name=func.__name__,
        description=getattr(func, "__doc__", "") or f"Call {func.__name__}",
        args_schema=ArgsSchema,
        func=wrapper,
    )

# %%

from tqdm import tqdm

# --- BUILD TOOLS SAFELY ---
all_tools = []
for function_name in tqdm(dir(math), desc="Building tools"):
    func = getattr(math, function_name)
    if isinstance(func, types.BuiltinFunctionType):
        tool = safe_tool_from_function(func)
        if tool:
            all_tools.append(tool)
# TODO: Improve all_tools limit logic
all_tools = all_tools[:30]

console.print("\n[bold pink]All Tools:[/bold pink]")
pprint(all_tools)

# %%
# import math
# import types
import uuid

from langgraph.store.base import PutOp
from langgraph.store.memory import InMemoryStore


from jet.models.utils import get_embedding_size

# _set_env("OPENAI_API_KEY")

# # Collect functions from `math` built-in
# all_tools = []
# for function_name in dir(math):
#     function = getattr(math, function_name)
#     if not isinstance(
#         function, types.BuiltinFunctionType
#     ):
#         continue
#     # This is an idiosyncrasy of the `math` library
#     if tool := convert_positional_only_function_to_tool(
#         function
#     ):
#         all_tools.append(tool)

# Create registry of tools. This is a dict mapping
# identifiers to tool instances.
tool_registry = {
    str(uuid.uuid4()): tool
    for tool in all_tools
}

# Index tool names and descriptions in the LangGraph
# Store. Here we use a simple in-memory store.
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": get_embedding_size("embeddinggemma"),
        "fields": ["description"],
    }
)

console.print(f"[bold pink]Tool Registry ({len(tool_registry)}):[/bold pink]")
# Prepare batch operations
put_ops = []
for tool_idx, (tool_id, tool) in enumerate(tool_registry.items()):
    pprint(f"{tool_idx + 1}: {tool.name}: {tool.description}")
    put_ops.append(
        PutOp(
            namespace=("tools",),
            key=tool_id,
            value={
                "description": f"{tool.name}: {tool.description}",
            },
        )
    )

# Execute all put operations in a single batch
batch_results = store.batch(put_ops)

# %%
# # Initialize agent
# builder = create_agent(llm, tool_registry)
# agent = builder.compile(store=store)

from jet.adapters.langchain.chat_agent_utils import build_agent
agent = build_agent(all_tools, llm)

# Display the agent visualization
# display(Image(agent.get_graph().draw_mermaid_png()))
png_data = agent.get_graph().draw_mermaid_png()
display_iterm2_image(png_data)

# %%
query = "Use available tools to calculate arc cosine of 0.5."
result = agent.invoke({"messages": query})

console.print("\n[bold pink]Agent tool result:[/bold pink]")
pprint(result)

# %% [markdown]
# ### Learn more
# 
# * **Toolshed: Scale Tool-Equipped Agents with Advanced RAG-Tool Fusion** - Lumer, E., Subbiah, V.K., Burke, J.A., Basavaraju, P.H. & Huber, A. (2024). arXiv:2410.14594.
# 
# The paper introduces Toolshed Knowledge Bases and Advanced RAG-Tool Fusion to address challenges in scaling tool-equipped AI agents. The Toolshed Knowledge Base is a vector database designed to store enhanced tool representations and optimize tool selection for large-scale tool-equipped agents. The Advanced RAG-Tool Fusion technique applies retrieval-augmented generation across three phases: pre-retrieval (tool document enhancement), intra-retrieval (query planning and transformation), and post-retrieval (document refinement and self-reflection). The researchers demonstrated significant performance improvements, achieving 46%, 56%, and 47% absolute improvements on different benchmark datasets (Recall@5), all without requiring model fine-tuning.
# 
# * **Graph RAG-Tool Fusion** - Lumer, E., Basavaraju, P.H., Mason, M., Burke, J.A. & Subbiah, V.K. (2025). arXiv:2502.07223.
# 
# This paper addresses limitations in current RAG approaches for tool selection by introducing Graph RAG-Tool Fusion, which combines vector-based retrieval with graph traversal to capture tool dependencies. Traditional RAG methods fail to capture structured dependencies between tools (e.g., a "get stock price" API requiring a "stock ticker" parameter from another API). The authors present ToolLinkOS, a benchmark dataset with 573 fictional tools across 15 industries, each averaging 6.3 tool dependencies. Graph RAG-Tool Fusion achieved absolute improvements of 71.7% and 22.1% over naïve RAG on ToolLinkOS and ToolSandbox benchmarks, respectively, by understanding and navigating interconnected tool relationships within a predefined knowledge graph.
# 
# * **LLM-Tool-Survey** - https://github.com/quchangle1/LLM-Tool-Survey
# 
# This comprehensive survey repository explores Tool Learning with Large Language Models, presenting a systematic examination of how AI models can effectively use external tools to enhance their capabilities. The repository covers key aspects including benefits of tools (knowledge acquisition, expertise enhancement, interaction improvement) and technical workflows. It provides an extensive collection of research papers categorized by tool types, reasoning methods, and technological approaches, ranging from mathematical tools and programming interpreters to multi-modal and domain-specific applications. The repository serves as a valuable collaborative resource for researchers and practitioners interested in the evolving landscape of AI tool integration.
# 
# * **Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval** - Shi, Z., Wang, Y., Yan, L., Ren, P., Wang, S., Yin, D. & Ren, Z. arXiv:2503.01763.
# 
# The paper introduces ToolRet, a benchmark for evaluating tool retrieval capabilities of information retrieval (IR) models in LLM contexts. Unlike existing benchmarks that manually pre-annotate small sets of relevant tools, ToolRet comprises 7.6k diverse retrieval tasks and a corpus of 43k tools from existing datasets. The research found that even IR models with strong performance in conventional benchmarks exhibit poor performance on ToolRet, directly impacting task success rates of tool-using LLMs. As a solution, the researchers contributed a large-scale training dataset with over 200k instances that substantially optimizes tool retrieval ability, bridging the gap between existing approaches and real-world tool-learning scenarios.
# 
# ## Knowledge 
# 
# [RAG](https://github.com/langchain-ai/rag-from-scratch) (retrieval augmented generation) is an extremely rich topic. Code agents are some of the best examples of agentic RAG in large-scale production. [In practice, RAG is can be a central context engineering challenge](https://x.com/_mohansolo/status/1899630246862966837). Varun from Windsurf captures some of these challenges well:
# 
# > Indexing code ≠ context retrieval … [We are doing indexing & embedding search … [with] AST parsing code and chunking along semantically meaningful boundaries … embedding search becomes unreliable as a retrieval heuristic as the size of the codebase grows … we must rely on a combination of techniques like grep/file search, knowledge graph based retrieval, and … a re-ranking step where [context] is ranked in order of relevance. 
# 
# ### RAG in LangGraph
# 
# There are several [tutorials and videos](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/) that show how to use RAG with LangGraph. When combining RAG with agents in LangGraph, it's common to build a retrieval tool. Note that this tool could incorporate any combination of RAG techniques, as mentioned above.
# 
# Fetch documents to use in our RAG system. We will use three of the most recent pages from Lilian Weng's excellent blog. We'll start by fetching the content of the pages using WebBaseLoader utility.

# %%
from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]

# %% [markdown]
# Split the fetched documents into smaller chunks for indexing into our vectorstore.

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# %% [markdown]
# Now that we have our split documents, we can index them into a vector store that we'll use for semantic search.

# %%
from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=embeddings
)
retriever = vectorstore.as_retriever()

# %% [markdown]
# Create a retriever tool that we can use in our agent.

# %%
from langchain_core.tools import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

# retriever_tool.invoke({"query": "types of reward hacking"})

# %% [markdown]
# Now, implement an agent that can select context from the tool.

# %%
# Augment the LLM with tools
tools = [retriever_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# %%
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from typing_extensions import Literal

from jet.adapters.llama_cpp.tokens import count_tokens
from jet.logger import logger

rag_prompt = """You are a helpful assistant tasked with retrieving information from a series of technical blog posts by Lilian Weng. 
Clarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and
proceed until you have sufficient context to answer the user's research request."""

# Nodes
def llm_call(state: MessagesState):
    messages = state["messages"]
    user_query = messages[-1].content

    # ------------------------------------------------------------------
    # 1. Retrieve & **count** tokens of raw docs *before* any concatenation
    # ------------------------------------------------------------------
    retrieved = retriever.invoke(user_query)
    doc_texts = [doc.page_content for doc in retrieved[:4]]

    # Count tokens for each document (exact, model-specific)
    doc_token_counts = count_tokens(
        doc_texts,
        model="qwen3-instruct-2507:4b",   # same model used for generation
        prevent_total=True,
        add_special_tokens=False,
    )   # -> List[int]

    # ------------------------------------------------------------------
    # 2. Build history part (always needed)
    # ------------------------------------------------------------------
    history_lines = []
    for msg in messages[:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content}")
    history_text = "\n".join(history_lines)

    # ------------------------------------------------------------------
    # 3. Determine how many docs we can safely include
    # ------------------------------------------------------------------
    MAX_CTX = 3000
    SAFETY_BUFFER = 600
    BUDGET = MAX_CTX - SAFETY_BUFFER          # ~2400 tokens

    # tokens already consumed by static parts (system prompt + question + delimiters)
    static_parts = (
        rag_prompt
        + "\n\nContext:\n"
        + "Documents:\n"
        + "\n\nHistory:\n"
        + history_text
        + "\n\nQuestion: "
        + user_query
    )
    static_tokens = count_tokens(static_parts, model="qwen3-instruct-2507:4b")

    available_for_docs = BUDGET - static_tokens - 200   # 200-token cushion for tool JSON

    # Greedily add docs until we run out of budget
    selected_docs = []
    consumed = 0
    for txt, cnt in zip(doc_texts, doc_token_counts):
        if consumed + cnt <= available_for_docs:
            selected_docs.append(txt)
            consumed += cnt
        else:
            break

    doc_text = "\n\n".join(selected_docs) if selected_docs else ""

    # ------------------------------------------------------------------
    # 4. Final prompt – guaranteed ≤ BUDGET
    # ------------------------------------------------------------------
    final_context = f"Documents:\n{doc_text}\n\nHistory:\n{history_text}"
    final_prompt = f"{rag_prompt}\n\nContext:\n{final_context}\nQuestion: {user_query}"

    # (optional) sanity-check
    total_tokens = count_tokens(final_prompt, model="qwen3-instruct-2507:4b")
    logger.log("final_prompt_tokens: ", total_tokens, colors=["INFO", "DEBUG"])

    response = llm_with_tools.invoke([SystemMessage(content=final_prompt)])
    return {"messages": [response]}
    
def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
png_data = agent.get_graph(xray=True).draw_mermaid_png()
display_iterm2_image(png_data)

# %%
query = "What are the types of reward hacking discussed in the blogs?"
result = agent.invoke({"messages": query})

console.print("\n[bold pink]Agent query result:[/bold pink]")
pprint(result)

# %%



