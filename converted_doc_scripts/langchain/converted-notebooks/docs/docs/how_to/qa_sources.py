from IPython.display import Image, display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List
from typing_extensions import Annotated, TypedDict
from typing_extensions import List, TypedDict
import ChatModelTabs from "@theme/ChatModelTabs"
import EmbeddingTabs from "@theme/EmbeddingTabs"
import VectorStoreTabs from "@theme/VectorStoreTabs"
import bs4
import json
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
# How to get your RAG application to return sources

Often in [Q&A](/docs/concepts/rag/) applications it's important to show users the sources that were used to generate the answer. The simplest way to do this is for the chain to return the Documents that were retrieved in each generation.

We'll work off of the Q&A app we built over the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng in the [RAG tutorial](/docs/tutorials/rag).

We will cover two approaches:

1. Using the basic RAG chain covered in [Part 1](/docs/tutorials/rag) of the RAG tutorial;
2. Using a conversational RAG chain as convered in [Part 2](/docs/tutorials/qa_chat_history) of the tutorial.

We will also show how to structure sources into the model response, such that a model can report what specific sources it used in generating its answer.

## Setup

### Dependencies

We'll use the following packages:
"""
logger.info("# How to get your RAG application to return sources")

# %pip install --upgrade --quiet langchain langchain-community langchainhub beautifulsoup4

"""
### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with [LangSmith](https://smith.langchain.com).

Note that LangSmith is not needed, but it is helpful. If you do want to use LangSmith, after you sign up at the link above, make sure to set your environment variables to start logging traces:

```python
os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
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


embeddings = OllamaEmbeddings(model="nomic-embed-text")

"""
And a [vector store](/docs/integrations/vectorstores/):


<VectorStoreTabs/>
"""
logger.info("And a [vector store](/docs/integrations/vectorstores/):")


vector_store = InMemoryVectorStore(embeddings)

"""
## RAG application

Let's reconstruct the Q&A app with sources we built over the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng in the [RAG tutorial](/docs/tutorials/rag).

First we index our documents:
"""
logger.info("## RAG application")


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(documents=all_splits)

"""
Next we build the application:
"""
logger.info("Next we build the application:")


prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


display(Image(graph.get_graph().draw_mermaid_png()))

"""
Because we're tracking the retrieved context in our application's state, it is accessible after invoking the application:
"""
logger.info("Because we're tracking the retrieved context in our application's state, it is accessible after invoking the application:")

result = graph.invoke({"question": "What is Task Decomposition?"})

logger.debug(f"Context: {result['context']}\n\n")
logger.debug(f"Answer: {result['answer']}")

"""
Here, `"context"` contains the sources that the LLM used in generating the response in `"answer"`.

## Structure sources in model response

Up to this point, we've simply propagated the documents returned from the retrieval step through to the final response. But this may not illustrate what subset of information the model relied on when generating its answer. Below, we show how to structure sources into the model response, allowing the model to report what specific context it relied on for its answer.

It is straightforward to extend the above LangGraph implementation. Below, we make a simple change: we use the model's tool-calling features to generate [structured output](/docs/how_to/structured_output/), consisting of an answer and list of sources. The schema for the response is represented in the `AnswerWithSources` TypedDict, below.
"""
logger.info("## Structure sources in model response")


class AnswerWithSources(TypedDict):
    """An answer to the question, with sources."""

    answer: str
    sources: Annotated[
        List[str],
        ...,
        "List of sources (author + year) used to answer the question",
    ]


class State(TypedDict):
    question: str
    context: List[Document]
    answer: AnswerWithSources


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content})
    structured_llm = llm.with_structured_output(AnswerWithSources)
    response = structured_llm.invoke(messages)
    return {"answer": response}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


result = graph.invoke({"question": "What is Chain of Thought?"})
logger.debug(json.dumps(result["answer"], indent=2))

"""
:::tip

View [LangSmith trace](https://smith.langchain.com/public/51d543f7-bdf6-4d93-9ecd-2fc09bf6d666/r).

:::

## Conversational RAG

[Part 2](/docs/tutorials/qa_chat_history) of the RAG tutorial implements a different architecture, in which steps in the RAG flow are represented via successive [message](/docs/concepts/messages/) objects. This leverages additional [tool-calling](/docs/concepts/tool_calling/) features of chat models, and more naturally accommodates a "back-and-forth" conversational user experience.

In that tutorial (and below), we propagate the retrieved documents as [artifacts](/docs/how_to/tool_artifacts/) on the tool messages. That makes it easy to pluck out the retrieved documents. Below, we add them as an additional key in the state, for convenience.

Note that we define the response format of the tool as `"content_and_artifact"`:
"""
logger.info("## Conversational RAG")


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
We can now build and compile the exact same application as in [Part 2](/docs/tutorials/qa_chat_history) of the RAG tutorial, with two changes:

1. We add a `context` key of the state to store retrieved documents;
2. In the `generate` step, we pluck out the retrieved documents and populate them in the state.

These changes are highlighted below.
"""
logger.info(
    "We can now build and compile the exact same application as in [Part 2](/docs/tutorials/qa_chat_history) of the RAG tutorial, with two changes:")


class State(MessagesState):
    context: List[Document]


def query_or_respond(state: State):
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
    context = []
    for tool_message in tool_messages:
        context.extend(tool_message.artifact)
    return {"messages": [response], "context": context}


"""
We can compile the application as before:
"""
logger.info("We can compile the application as before:")

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

graph = graph_builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

"""
Invoking our application, we see that the retrieved [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects are accessible from the application state.
"""
logger.info(
    "Invoking our application, we see that the retrieved [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects are accessible from the application state.")

input_message = "What is Task Decomposition?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

step["context"]

"""
:::tip

Check out the [LangSmith trace](https://smith.langchain.com/public/cc25515d-2e46-44fa-8bb2-b9cb0f451504/r).

:::
"""
logger.info(
    "Check out the [LangSmith trace](https://smith.langchain.com/public/cc25515d-2e46-44fa-8bb2-b9cb0f451504/r).")

logger.info("\n\n[DONE]", bright=True)
