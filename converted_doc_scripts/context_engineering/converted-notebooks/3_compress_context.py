from IPython.display import Image, display
from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from rich.markdown import Markdown
from typing_extensions import Literal
from utils import format_messages
from utils import format_messages, format_message
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


"""
# Compressing Context in LangGraph

*Compressing context involves retaining only the tokens required to perform a task.*

![Screenshot 2025-07-09 at 2.28.10 PM.png](attachment:81bde857-21da-4464-b4ff-156e6f7d7079.png)

## Summarization 

Agent interactions can span [hundreds of turns](https://www.anthropic.com/engineering/built-multi-agent-research-system) and use token-heavy tool calls. Summarization is one common way to manage these challenges. If you’ve used Claude Code, you’ve seen this in action. Claude Code runs “[auto-compact](https://docs.anthropic.com/en/docs/claude-code/costs)” after you exceed 95% of the context window and it will summarize the full trajectory of user-agent interactions. This type of compression across an [agent trajectory](https://langchain-ai.github.io/langgraph/concepts/memory/#manage-short-term-memory) can use various strategies such as [recursive](https://arxiv.org/pdf/2308.15022#:~:text=the%20retrieved%20utterances%20capture%20the,based%203) or [hierarchical](https://alignment.anthropic.com/2025/summarization-for-monitoring/#:~:text=We%20addressed%20these%20issues%20by,of%20our%20computer%20use%20capability) summarization.

It can also be useful to [add summarization](https://github.com/langchain-ai/open_deep_research/blob/e5a5160a398a3699857d00d8569cb7fd0ac48a4f/src/open_deep_research/utils.py#L1407) at points in an agent’s trajectory. For example, it can be used to post-process certain tool calls (e.g., token-heavy search tools). As a second example, [Cognition](https://cognition.ai/blog/dont-build-multi-agents#a-theory-of-building-long-running-agents) mentioned summarization at agent-agent boundaries to knowledge hand-off. They also the challenge if specific events or decisions to be captured. They use a fine-tuned model for this in Devin, which underscores how much work can go into this step.

![image (2).webp](attachment:756744cf-234a-4a7a-bd44-4a47801af657.webp)

### Summarization in LangGraph

Because LangGraph is a low [is a low-level orchestration framework](https://blog.langchain.com/how-to-think-about-agent-frameworks/), you can [lay out your agent as a set of nodes](https://www.youtube.com/watch?v=aHCDrAbH_go), [explicitly define](https://blog.langchain.com/how-to-think-about-agent-frameworks/) the logic within each one, and define an state object that is passed between them. This low-level control gives several ways to compress context.

You can use a message list as your agent state and [summarize](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#manage-short-term-memory) using [a few built-in utilities](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#manage-short-term-memory).

#### Summarize Messages

Let's implement a RAG agent, and add summarization of the conversation history.

"""
logger.info("# Compressing Context in LangGraph")


urls = [
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000,
    chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

embeddings = init_embeddings("ollama:nomic-embed-text")
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

test_result = retriever_tool.invoke({"query": "types of reward hacking"})



llm = init_chat_model("ollama:claude-sonnet-4-20250514", temperature=0)

tools = [retriever_tool]
tools_by_name = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)





class State(MessagesState):
    """Extended state that includes a summary field for context compression."""
    summary: str


rag_prompt = """You are a helpful assistant tasked with retrieving information from a series of technical blog posts by Lilian Weng.
Clarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and
proceed until you have sufficient context to answer the user's research request."""

summarization_prompt = """Summarize the full chat history and all tool feedback to
give an overview of what the user asked about and what the agent did."""


def llm_call(state: MessagesState) -> dict:
    """Execute LLM call with system prompt and message history.

    Args:
        state: Current conversation state

    Returns:
        Dictionary with new messages
    """
    messages = [SystemMessage(content=rag_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: MessagesState) -> dict:
    """Execute tool calls and return results.

    Args:
        state: Current conversation state with tool calls

    Returns:
        Dictionary with tool results
    """
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def summary_node(state: MessagesState) -> dict:
    """Generate a summary of the conversation and tool interactions.

    Args:
        state: Current conversation state

    Returns:
        Dictionary with conversation summary
    """
    messages = [SystemMessage(content=summarization_prompt)] + state["messages"]
    result = llm.invoke(messages)
    return {"summary": result.content}


def should_continue(state: MessagesState) -> Literal["Action", "summary_node"]:
    """Determine next step based on whether LLM made tool calls.

    Args:
        state: Current conversation state

    Returns:
        Next node to execute
    """
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "Action"
    return "summary_node"


agent_builder = StateGraph(State)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)
agent_builder.add_node("summary_node", summary_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        "summary_node": "summary_node",
    },
)
agent_builder.add_edge("environment", "llm_call")
agent_builder.add_edge("summary_node", END)

agent = agent_builder.compile()

display(Image(agent.get_graph(xray=True).draw_mermaid_png()))




query = "Why does RL improve LLM reasoning according to the blogs?"
result = agent.invoke({"messages": query})
format_message(result['messages'])


Markdown(result["summary"])

"""
Nice, but it uses `115k tokens`!

See trace:  

https://smith.langchain.com/public/50d70503-1a8e-46c1-bbba-a1efb8626b05/r

This is often a challenge with agents that have token-heavy tool calls!

#### Summarize Tools

Let's update the RAG agent, and add summarization the tool call output.

"""
logger.info("#### Summarize Tools")

tool_summarization_prompt = """You will be provided a doc from a RAG system.
Summarize the docs, ensuring to retain all relevant / essential information.
Your goal is simply to reduce the size of the doc (tokens) to a more manageable size."""

def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "Action"
    return END

def tool_node_with_summarization(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        summary = llm.invoke([{"role":"system",
                              "content":tool_summarization_prompt},
                              {"role":"user",
                               "content":observation}])
        result.append(ToolMessage(content=summary.content, tool_call_id=tool_call["id"]))
    return {"messages": result}

agent_builder = StateGraph(State)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment_with_summarization", tool_node_with_summarization)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment_with_summarization",
        END: END,
    },
)
agent_builder.add_edge("environment_with_summarization", "llm_call")

agent = agent_builder.compile()

display(Image(agent.get_graph(xray=True).draw_mermaid_png()))



query = "Why does RL improve LLM reasoning according to the blogs?"
result = agent.invoke({"messages": query})
format_messages(result['messages'])

"""
This uses 60k tokens. 

https://smith.langchain.com/public/994cdf93-e837-4708-9628-c83b397dd4b5/r

#### Learn More

* **Heuristic Compression and Message Trimming** - https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#trim-messages

LangGraph provides several message management strategies to handle context window limitations. The `trim_messages()` function allows you to limit token count by keeping the "last" messages and controlling maximum tokens and message boundaries. This can be implemented as pre-model hooks for agents with custom state management. Key benefits include preventing context window overflow, maintaining conversation context, optimizing memory usage, and enabling long-running conversations. The approach emphasizes flexible, programmatic management of conversational memory across different AI interaction scenarios.

* **SummarizationNode as Pre-Model Hook** - https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/

SummarizationNode helps manage conversation history by summarizing messages when token count exceeds specified limits. It can be implemented as a pre-model hook in ReAct agents, allowing you to keep original message history or overwrite it with summaries. The node uses `count_tokens_approximately()` to track message history size and supports configurable parameters including `max_tokens` (threshold), `max_summary_tokens` (summary length), and `output_messages_key` (storage location). This approach provides flexible mechanisms for managing conversation memory in AI agents while preventing context window overflow and maintaining conversation context.

* **LangMem Summarization** - https://langchain-ai.github.io/langmem/guides/summarization/

LangMem provides strategies for managing long context through message history summarization. It offers two primary approaches: direct summarization using `summarize_messages()` function with configurable token thresholds and "running summary" maintenance, and the SummarizationNode approach with dedicated nodes for automatic summary propagation. Key implementation considerations include configuring token limits, using separate state keys for full message history versus summaries, and maintaining conversation context across multiple interactions. LangMem integrates seamlessly with LangGraph state management for both simple chatbots and ReAct-style agents.

"""
logger.info("#### Learn More")


logger.info("\n\n[DONE]", bright=True)