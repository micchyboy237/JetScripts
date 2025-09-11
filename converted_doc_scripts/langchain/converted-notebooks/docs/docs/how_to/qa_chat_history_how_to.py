from IPython.display import Image, display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt import create_react_agent
from typing_extensions import List, TypedDict
import ChatModelTabs from "@theme/ChatModelTabs";
import EmbeddingTabs from "@theme/EmbeddingTabs";
import VectorStoreTabs from "@theme/VectorStoreTabs";
import bs4
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)


"""
# How to add chat history

:::note

This guide previously used the [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html) abstraction. You can access this version of the documentation in the [v0.2 docs](https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/).

As of the v0.3 release of LangChain, we recommend that LangChain users take advantage of [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) to incorporate `memory` into new LangChain applications.

If your code is already relying on `RunnableWithMessageHistory` or `BaseChatMessageHistory`, you do **not** need to make any changes. We do not plan on deprecating this functionality in the near future as it works for simple chat applications and any code that uses `RunnableWithMessageHistory` will continue to work as expected.

Please see [How to migrate to LangGraph Memory](/docs/versions/migrating_memory/) for more details.
:::


In many [Q&A applications](/docs/concepts/rag/) we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of "memory" of past questions and answers, and some logic for incorporating those into its current thinking.

In this guide we focus on **adding logic for incorporating historical messages.**

This is largely a condensed version of the [Conversational RAG tutorial](/docs/tutorials/qa_chat_history).

We will cover two approaches:
1. [Chains](/docs/how_to/qa_chat_history_how_to#chains), in which we always execute a retrieval step;
2. [Agents](/docs/how_to/qa_chat_history_how_to#agents), in which we give an LLM discretion over whether and how to execute a retrieval step (or multiple steps).

For the external knowledge source, we will use the same [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng from the [RAG tutorial](/docs/tutorials/rag).

Both approaches leverage [LangGraph](https://langchain-ai.github.io/langgraph/) as an orchestration framework. LangGraph implements a built-in [persistence layer](https://langchain-ai.github.io/langgraph/concepts/persistence/), making it ideal for chat applications that support multiple conversational turns.

## Setup

### Dependencies

We'll use Ollama embeddings and an InMemory vector store in this walkthrough, but everything shown here works with any [Embeddings](/docs/concepts/embedding_models), and [VectorStore](/docs/concepts/vectorstores) or [Retriever](/docs/concepts/retrievers). 

We'll use the following packages:
"""
logger.info("# How to add chat history")

# %%capture --no-stderr
# %pip install --upgrade --quiet langgraph langchain-community beautifulsoup4

"""
### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with [LangSmith](https://smith.langchain.com).

Note that LangSmith is not needed, but it is helpful. If you do want to use LangSmith, after you sign up at the link above, make sure to set your environment variables to start logging traces:

```python
os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("LANGSMITH_API_KEY"):
#     os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

### Components

We will need to select three components from LangChain's suite of integrations.

A [chat model](/docs/integrations/chat/):


<ChatModelTabs customVarName="llm" />
"""
logger.info("### LangSmith")


llm = ChatOllama(model="llama3.2")

"""
An [embedding model](/docs/integrations/text_embedding/):


<EmbeddingTabs/>
"""
logger.info("An [embedding model](/docs/integrations/text_embedding/):")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
And a [vector store](/docs/integrations/vectorstores/):


<VectorStoreTabs/>
"""
logger.info("And a [vector store](/docs/integrations/vectorstores/):")


vector_store = InMemoryVectorStore(embeddings)

"""
## Chains {#chains}

The [RAG Tutorial](/docs/tutorials/rag) indexes an [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng. We will repeat that here. Below we load the content of the page, split it into sub-documents, and embed the documents into our [vector store](/docs/concepts/vectorstores/):
"""
logger.info("## Chains {#chains}")


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(documents=all_splits)

"""
As detailed in [Part 2](/docs/tutorials/qa_chat_history) of the RAG tutorial, we can naturally support a conversational experience by representing the flow of the RAG application as a sequence of [messages](/docs/concepts/messages/):

1. User input as a `HumanMessage`;
2. Vector store query as an `AIMessage` with tool calls;
3. Retrieved documents as a `ToolMessage`;
4. Final response as a `AIMessage`.

We will use [tool-calling](/docs/concepts/tool_calling/) to facilitate this, which additionally allows the query to be generated by the LLM. We can build a [tool](/docs/concepts/tools) to execute the retrieval step:
"""
logger.info("As detailed in [Part 2](/docs/tutorials/qa_chat_history) of the RAG tutorial, we can naturally support a conversational experience by representing the flow of the RAG application as a sequence of [messages](/docs/concepts/messages/):")



@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

"""
We can now build our LangGraph application.

Note that we compile it with a [checkpointer](https://langchain-ai.github.io/langgraph/concepts/persistence/) to support a back-and-forth conversation. LangGraph comes with a simple [in-memory checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#memorysaver), which we use below. See its documentation for more detail, including how to use different persistence backends (e.g., SQLite or Postgres).
"""
logger.info("We can now build our LangGraph application.")



def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


display(Image(graph.get_graph().draw_mermaid_png()))

"""
Let's test our application.

Note that it responds appropriately to messages that do not require an additional retrieval step:
"""
logger.info("Let's test our application.")

config = {"configurable": {"thread_id": "abc123"}}

input_message = "Hello"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_logger.debug()

"""
And when executing a search, we can stream the steps to observe the query generation, retrieval, and answer generation:
"""
logger.info("And when executing a search, we can stream the steps to observe the query generation, retrieval, and answer generation:")

input_message = "What is Task Decomposition?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_logger.debug()

"""
Finally, because we have compiled our application with a [checkpointer](https://langchain-ai.github.io/langgraph/concepts/persistence/), historical messages are maintained in the state. This allows the model to contextualize user queries:
"""
logger.info("Finally, because we have compiled our application with a [checkpointer](https://langchain-ai.github.io/langgraph/concepts/persistence/), historical messages are maintained in the state. This allows the model to contextualize user queries:")

input_message = "Can you look up some common ways of doing it?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_logger.debug()

"""
Note that we can observe the full sequence of messages sent to the chat model-- including tool calls and retrieved context-- in the [LangSmith trace](https://smith.langchain.com/public/3c85919e-9609-4a0d-8df1-21726f8f3e5c/r).

The conversation history can also be inspected via the state of the application:
"""
logger.info("Note that we can observe the full sequence of messages sent to the chat model-- including tool calls and retrieved context-- in the [LangSmith trace](https://smith.langchain.com/public/3c85919e-9609-4a0d-8df1-21726f8f3e5c/r).")

chat_history = graph.get_state(config).values["messages"]
for message in chat_history:
    message.pretty_logger.debug()

"""
## Agents {#agents}

[Agents](/docs/concepts/agents) leverage the reasoning capabilities of LLMs to make decisions during execution. Using agents allows you to offload additional discretion over the retrieval process. Although their behavior is less predictable than the above "chain", they are able to execute multiple retrieval steps in service of a query, or iterate on a single search.

Below we assemble a minimal RAG agent. Using LangGraph's [pre-built ReAct agent constructor](https://langchain-ai.github.io/langgraph/how-tos/#langgraph.prebuilt.chat_agent_executor.create_react_agent), we can do this in one line.

:::tip

Check out LangGraph's [Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/) tutorial for more advanced formulations.

:::
"""
logger.info("## Agents {#agents}")


agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)

"""
Let's inspect the graph:
"""
logger.info("Let's inspect the graph:")

display(Image(agent_executor.get_graph().draw_mermaid_png()))

"""
The key difference from our earlier implementation is that instead of a final generation step that ends the run, here the tool invocation loops back to the original LLM call. The model can then either answer the question using the retrieved context, or generate another tool call to obtain more information.

Let's test this out. We construct a question that would typically require an iterative sequence of retrieval steps to answer:
"""
logger.info("The key difference from our earlier implementation is that instead of a final generation step that ends the run, here the tool invocation loops back to the original LLM call. The model can then either answer the question using the retrieved context, or generate another tool call to obtain more information.")

config = {"configurable": {"thread_id": "def234"}}

input_message = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    event["messages"][-1].pretty_logger.debug()

"""
Note that the agent:

1. Generates a query to search for a standard method for task decomposition;
2. Receiving the answer, generates a second query to search for common extensions of it;
3. Having received all necessary context, answers the question.

We can see the full sequence of steps, along with latency and other metadata, in the [LangSmith trace](https://smith.langchain.com/public/48cbd35e-9ac1-49ab-8c09-500d54c06b81/r).

## Next steps

We've covered the steps to build a basic conversational Q&A application:

- We used chains to build a predictable application that generates search queries for each user input;
- We used agents to build an application that "decides" when and how to generate search queries.

To explore different types of retrievers and retrieval strategies, visit the [retrievers](/docs/how_to#retrievers) section of the how-to guides.

For a detailed walkthrough of LangChain's conversation memory abstractions, visit the [How to add message history (memory)](/docs/how_to/message_history) LCEL page.

To learn more about agents, head to the [Agents Modules](/docs/tutorials/agents).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)