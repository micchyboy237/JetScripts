from IPython.display import Image, display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import ChatModelTabs from "@theme/ChatModelTabs"
import EmbeddingTabs from "@theme/EmbeddingTabs"
import VectorStoreTabs from "@theme/VectorStoreTabs"
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
# How to stream results from your RAG application

This guide explains how to stream results from a [RAG](/docs/concepts/rag/) application. It covers streaming tokens from the final output as well as intermediate steps of a chain (e.g., from query re-writing).

We'll work off of the Q&A app with sources we built over the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng in the [RAG tutorial](/docs/tutorials/rag).

## Setup

### Dependencies

We'll use the following packages:
"""
logger.info("# How to stream results from your RAG application")

# %pip install --upgrade --quiet  langchain langchain-community langchainhub beautifulsoup4

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
## Streaming final outputs

LangGraph supports several [streaming modes](https://langchain-ai.github.io/langgraph/how-tos/#streaming), which can be controlled by specifying the `stream_mode` parameter. Setting `stream_mode="messages"` allows us to stream tokens from chat model invocations.

In general there can be multiple chat model invocations in an application (although here there is just one). Below, we filter to only the last step using the name of the corresponding node:
"""
logger.info("## Streaming final outputs")

input_message = "What is Task Decomposition?"

for message, metadata in graph.stream(
    {"question": "What is Task Decomposition?"},
    stream_mode="messages",
):
    if metadata["langgraph_node"] == "generate":
        logger.debug(message.content, end="|")

"""
## Streaming intermediate steps

Other streaming modes will generally stream steps from our invocation-- i.e., state updates from individual nodes. In this case, each node is just appending a new key to the state:
"""
logger.info("## Streaming intermediate steps")

for step in graph.stream(
    {"question": "What is Task Decomposition?"},
    stream_mode="updates",
):
    logger.debug(f"{step}\n\n----------------\n")

"""
For more on streaming with LangGraph, check out its [streaming documentation](https://langchain-ai.github.io/langgraph/how-tos/#streaming). For more information on streaming individual LangChain [Runnables](/docs/concepts/runnables/), refer to [this guide](/docs/how_to/streaming/).
"""
logger.info(
    "For more on streaming with LangGraph, check out its [streaming documentation](https://langchain-ai.github.io/langgraph/how-tos/#streaming). For more information on streaming individual LangChain [Runnables](/docs/concepts/runnables/), refer to [this guide](/docs/how_to/streaming/).")

logger.info("\n\n[DONE]", bright=True)
