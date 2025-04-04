import networkx as nx
import matplotlib.pyplot as plt
from typing_extensions import TypedDict
from IPython.display import Image, display
import uuid
from jet.token.token_utils import get_tokenizer
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
import tiktoken
from typing import List, Literal, Optional
import json
import os
from llama_index.core.schema import Document as LlamaDocument
from jet.search import search_searxng
from jet.llm.query import setup_index, FUSION_MODES
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# A Long-Term Memory Agent
#
# This tutorial shows how to implement an agent with long-term memory capabilities using LangGraph. The agent can store, retrieve, and use memories to enhance its interactions with users.
#
# Inspired by papers like [MemGPT](https://memgpt.ai/) and distilled from our own works on long-term memory, the graph extracts memories from chat interactions and persists them to a database. "Memory" in this tutorial will be represented in two ways:
# * a piece of text information that is generated by the agent
# * structured information about entities extracted by the agent in the shape of `(subject, predicate, object)` knowledge triples.
#
# This information can later be read or queried semantically to provide personalized context when your bot is responding to a particular user.
#
# The KEY idea is that by saving memories, the agent persists information about users that is SHARED across multiple conversations (threads), which is different from memory of a single conversation that is already enabled by LangGraph's [persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/).
#
# ![memory_graph.png](attachment:a2b70d8c-dd71-41d0-9c6d-d3ed922c29cc.png)
#
# You can also check out a full implementation of this agent in [this repo](https://github.com/langchain-ai/lang-memgpt).

# Install dependencies

# %pip install -U --quiet langgraph langchain-openai langchain-community tiktoken

# import getpass


# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("OPENAI_API_KEY")
# _set_env("TAVILY_API_KEY")


# Define vectorstore for memories

# First, let's define the vectorstore where we will be storing our memories. Memories will be stored as embeddings and later looked up based on the conversation context. We will be using an in-memory vectorstore.

recall_vector_store = InMemoryVectorStore(
    OllamaEmbeddings(model="nomic-embed-text"))

# Define tools

# Next, let's define our memory tools. We will need a tool to store the memories and another tool to search them to find the most relevant memory.


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id


@tool
def save_recall_memory(memories: List[str], config: RunnableConfig) -> str:
    """Save memories to vectorstore for later semantic retrieval.

    Args:
        memories (List[str]): The generated list of phrases or simple sentences that describe the current user derived from conversations. Can update previous memories if new info is relevant.

    Returns:
        (str): Passed memories from args.
    """
    user_id = get_user_id(config)
    documents = []
    for memory in memories:
        document = Document(
            page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
        )
        documents.append(document)
    recall_vector_store.add_documents(documents)
    return memories


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]


@tool
def search(query: str, config: RunnableConfig) -> list[str]:
    """
    A search engine optimized for comprehensive, accurate, and trusted results.
    Useful for when you need to answer questions about current events.
    Input should be a search query.
    """

    results = search_searxng(
        query_url="http://searxng.local:8080/search",
        query=query,
        min_score=0,
        engines=["google"],
    )

    documents = [LlamaDocument(text=result['content']) for result in results]

    query_nodes = setup_index(documents)

    logger.newline()
    result = query_nodes(
        query, FUSION_MODES.RELATIVE_SCORE, score_threshold=0.3)

    return result['texts']


# Additionally, let's give our agent ability to search the web using [Tavily](https://tavily.com/).

# search = TavilySearchResults(max_results=1)
tools = [save_recall_memory, search_recall_memories, search]

# Define state, nodes and edges

# Our graph state will contain just two channels -- `messages` for keeping track of the chat history and `recall_memories` -- contextual memories that will be pulled in before calling the agent and passed to the agent's system prompt.


class State(MessagesState):
    recall_memories: List[str]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_recall_memory, search_recall_memories, search)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)

model = ChatOllama(model="llama3.2")
model_with_tools = model.bind_tools(tools)

tokenizer = get_tokenizer("llama3.2")


def agent(state: State) -> State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" +
        "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {
        "messages": [prediction],
    }


def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": recall_memories,
    }


def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"

    return END

# Build the graph

# Our agent graph is going to be very similar to simple [ReAct agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent). The only important modification is adding a node to load memories BEFORE calling the agent for the first time.


builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools, ["tools", END])
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


display(Image(graph.get_graph().draw_mermaid_png()))

# Run the agent!

# Let's run the agent for the first time and tell it some information about the user!


def pretty_print_stream_chunk(chunk):
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            updates["messages"][-1].pretty_print()
        else:
            print(updates)

        print("\n")


config = {"configurable": {"user_id": "1", "thread_id": "1"}}

for chunk in graph.stream({"messages": [("user", "my name is John")]}, config=config):
    pretty_print_stream_chunk(chunk)

# You can see that the agent saved the memory about user's name. Let's add some more information about the user!

for chunk in graph.stream({"messages": [("user", "i love pizza")]}, config=config):
    pretty_print_stream_chunk(chunk)

for chunk in graph.stream(
    {"messages": [("user", "yes -- pepperoni!")]},
    config={"configurable": {"user_id": "1", "thread_id": "1"}},
):
    pretty_print_stream_chunk(chunk)

for chunk in graph.stream(
    {"messages": [("user", "i also just moved to new york")]},
    config={"configurable": {"user_id": "1", "thread_id": "1"}},
):
    pretty_print_stream_chunk(chunk)

# Now we can use the saved information about our user on a different thread. Let's try it out:

config = {"configurable": {"user_id": "1", "thread_id": "2"}}

for chunk in graph.stream(
    {"messages": [("user", "where should i go for dinner?")]}, config=config
):
    pretty_print_stream_chunk(chunk)

# Notice how the agent is loading the most relevant memories before answering, and in our case suggests the dinner recommendations based on both the food preferences as well as location.
#
# Finally, let's use the search tool together with the rest of the conversation context and memory to find location of a pizzeria:

for chunk in graph.stream(
    {"messages": [
        ("user", "what's the address for joe's in greenwich village?")]},
    config=config,
):
    pretty_print_stream_chunk(chunk)

# If you were to pass a different user ID, the agent's response will not be personalized as we haven't saved any information about the other user:

# Adding structured memories
#
# So far we've represented memories as strings, e.g., `"John loves pizza"`. This is a natural representation when persisting memories to a vector store. If your use-case would benefit from other persistence backends-- such as a graph database-- we can update our application to generate memories with additional structure.
#
# Below, we update the `save_recall_memory` tool to accept a list of "knowledge triples", or 3-tuples with a `subject`, `predicate`, and `object`, suitable for storage in a knolwedge graph. Our model will then generate these representations as part of its tool calls.
#
# For simplicity, we use the same vector database as before, but the `save_recall_memory` and `search_recall_memories` tools could be further updated to interact with a graph database. For now, we only need to update the `save_recall_memory` tool:

recall_vector_store = InMemoryVectorStore(
    OllamaEmbeddings(model="nomic-embed-text"))


class KnowledgeTriple(TypedDict):
    subject: str
    predicate: str
    object: str


@tool
def save_recall_memory(memories: str | List[KnowledgeTriple], config: RunnableConfig) -> List[KnowledgeTriple]:
    """Save memory to vectorstore for later semantic retrieval.

    class KnowledgeTriple(TypedDict):
        subject: str # Current user name only
        predicate: str
        object: str # noun derived from predicate

    Args:
        memories (List[KnowledgeTriple]): The generated list of memories with each having a subject and predicate derived from conversations. Each are in the 3rd person perspective of the user.

    Returns:
        (List[KnowledgeTriple]): Passed memories from args.
    """
    user_id = get_user_id(config)
    documents = []
    if isinstance(memories, str):
        try:
            # Attempt to parse the string directly
            memories = json.loads(memories)
        except json.JSONDecodeError:
            # Replace single quotes with double quotes
            memories = memories.replace("'", '"')
            memories = json.loads(memories)  # Retry loading
    for memory in memories:
        serialized = " ".join(memory.values())
        document = Document(
            serialized,
            id=str(uuid.uuid4()),
            metadata={
                "user_id": user_id,
                **memory,
            },
        )
        documents.append(document)
    recall_vector_store.add_documents(documents)
    return memories

# We can then compile the graph exactly as before:


tools = [save_recall_memory, search_recall_memories, search]
model_with_tools = model.bind_tools(tools)


builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools, ["tools", END])
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"user_id": "3", "thread_id": "1"}}

for chunk in graph.stream({"messages": [("user", "Hi, I'm Alice.")]}, config=config):
    pretty_print_stream_chunk(chunk)

# Note that the application elects to extract knowledge-triples from the user's statements:

for chunk in graph.stream(
    {"messages": [("user", "My friend John likes Pizza.")]}, config=config
):
    pretty_print_stream_chunk(chunk)

# As before, the memories generated from one thread are accessed in another thread from the same user:

config = {"configurable": {"user_id": "3", "thread_id": "2"}}

for chunk in graph.stream(
    {"messages": [("user", "What food should I bring to John's party?")]}, config=config
):
    pretty_print_stream_chunk(chunk)

# Optionally, for illustrative purposes we can visualize the knowledge graph extracted by the model:

# %pip install -U --quiet matplotlib networkx


records = recall_vector_store.similarity_search(
    "Alice", k=2, filter=lambda doc: doc.metadata["user_id"] == "3"
)


plt.figure(figsize=(6, 4), dpi=80)
G = nx.DiGraph()

for record in records:
    G.add_edge(
        record.metadata["subject"],
        record.metadata["object"],
        label=record.metadata["predicate"],
    )

pos = nx.spring_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    font_weight="bold",
    arrows=True,
)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
plt.show()

logger.info("\n\n[DONE]", bright=True)
