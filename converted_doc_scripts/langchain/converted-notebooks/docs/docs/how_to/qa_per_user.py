from IPython.display import Image, display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.runnables import RunnableConfig
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import ChatModelTabs from "@theme/ChatModelTabs"
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
# How to do per-user retrieval

This guide demonstrates how to configure runtime properties of a retrieval chain. An example application is to limit the documents available to a [retriever](/docs/concepts/retrievers/) based on the user.

When building a [retrieval app](/docs/concepts/rag/), you often have to build it with multiple users in mind. This means that you may be storing data not just for one user, but for many different users, and they should not be able to see eachother's data. This means that you need to be able to configure your retrieval chain to only retrieve certain information. This generally involves two steps.

**Step 1: Make sure the retriever you are using supports multiple users**

At the moment, there is no unified flag or filter for this in LangChain. Rather, each vectorstore and retriever may have their own, and may be called different things (namespaces, multi-tenancy, etc). For vectorstores, this is generally exposed as a keyword argument that is passed in during `similarity_search`. By reading the documentation or source code, figure out whether the retriever you are using supports multiple users, and, if so, how to use it.

Note: adding documentation and/or support for multiple users for retrievers that do not support it (or document it) is a GREAT way to contribute to LangChain

**Step 2: Add that parameter as a configurable field for the chain**

This will let you easily call the chain and configure any relevant flags at runtime. See [this documentation](/docs/how_to/configure) for more information on configuration.

Now, at runtime you can call this chain with configurable field.

## Code Example

Let's see a concrete example of what this looks like in code. We will use Pinecone for this example.

To configure Pinecone, set the following environment variable:

- `PINECONE_API_KEY`: Your Pinecone API key
"""
logger.info("# How to do per-user retrieval")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = PineconeVectorStore(
    index_name="test-example", embedding=embeddings)

vectorstore.add_texts(["I worked at Kensho"], namespace="harrison")
vectorstore.add_texts(["I worked at Facebook"], namespace="ankush")

"""
The pinecone kwarg for `namespace` can be used to separate documents
"""
logger.info(
    "The pinecone kwarg for `namespace` can be used to separate documents")

vectorstore.as_retriever(search_kwargs={"namespace": "ankush"}).invoke(
    "where did i work?"
)

vectorstore.as_retriever(search_kwargs={"namespace": "harrison"}).invoke(
    "where did i work?"
)

"""
We can now create the chain that we will use to do question-answering over.

Let's first select a LLM.


<ChatModelTabs customVarName="llm" />
"""
logger.info(
    "We can now create the chain that we will use to do question-answering over.")


llm = ChatOllama(model="llama3.2")

"""
This will follow the basic implementation from the [RAG tutorial](/docs/tutorials/rag), but we will allow the retrieval step to be configurable.
"""
logger.info(
    "This will follow the basic implementation from the [RAG tutorial](/docs/tutorials/rag), but we will allow the retrieval step to be configurable.")


template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

retriever = vectorstore.as_retriever()

"""
Here we mark the retriever as having a configurable field. All vectorstore retrievers have `search_kwargs` as a field. This is just a dictionary, with vectorstore specific fields.

This will let us pass in a value for `search_kwargs` when invoking the chain.
"""
logger.info("Here we mark the retriever as having a configurable field. All vectorstore retrievers have `search_kwargs` as a field. This is just a dictionary, with vectorstore specific fields.")

configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

"""
We can now create the chain using our configurable retriever.
"""
logger.info("We can now create the chain using our configurable retriever.")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State, config: RunnableConfig):
    retrieved_docs = configurable_retriever.invoke(state["question"], config)
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
We can now invoke the chain with configurable options. `search_kwargs` is the id of the configurable field. The value is the search kwargs to use for Pinecone.
"""
logger.info("We can now invoke the chain with configurable options. `search_kwargs` is the id of the configurable field. The value is the search kwargs to use for Pinecone.")

result = graph.invoke(
    {"question": "Where did the user work?"},
    config={"configurable": {"search_kwargs": {"namespace": "harrison"}}},
)

result

result = graph.invoke(
    {"question": "Where did the user work?"},
    config={"configurable": {"search_kwargs": {"namespace": "ankush"}}},
)

result

"""
For details operating your specific vector store, see the [integration pages](/docs/integrations/vectorstores/).
"""
logger.info(
    "For details operating your specific vector store, see the [integration pages](/docs/integrations/vectorstores/).")

logger.info("\n\n[DONE]", bright=True)
