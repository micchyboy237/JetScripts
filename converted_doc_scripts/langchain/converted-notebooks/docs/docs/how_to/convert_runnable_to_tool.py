from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.prebuilt import create_react_agent
from operator import itemgetter
from pydantic import BaseModel, Field
from typing import Any, Dict
from typing import List
from typing_extensions import TypedDict
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
# How to convert Runnables to Tools

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Runnables](/docs/concepts/runnables)
- [Tools](/docs/concepts/tools)
- [Agents](/docs/tutorials/agents)

:::

Here we will demonstrate how to convert a LangChain `Runnable` into a tool that can be used by agents, chains, or chat models.

## Dependencies

**Note**: this guide requires `langchain-core` >= 0.2.13. We will also use [Ollama](/docs/integrations/providers/ollama/) for embeddings, but any LangChain embeddings should suffice. We will use a simple [LangGraph](https://langchain-ai.github.io/langgraph/) agent for demonstration purposes.
"""
logger.info("# How to convert Runnables to Tools")

# %%capture --no-stderr
# %pip install -U langchain-core langchain-ollama langgraph

"""
LangChain [tools](/docs/concepts/tools) are interfaces that an agent, chain, or chat model can use to interact with the world. See [here](/docs/how_to/#tools) for how-to guides covering tool-calling, built-in tools, custom tools, and more information.

LangChain tools-- instances of [BaseTool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.BaseTool.html)-- are [Runnables](/docs/concepts/runnables) with additional constraints that enable them to be invoked effectively by language models:

- Their inputs are constrained to be serializable, specifically strings and Python `dict` objects;
- They contain names and descriptions indicating how and when they should be used;
- They may contain a detailed [args_schema](https://python.langchain.com/docs/how_to/custom_tools/) for their arguments. That is, while a tool (as a `Runnable`) might accept a single `dict` input, the specific keys and type information needed to populate a dict should be specified in the `args_schema`.

Runnables that accept string or `dict` input can be converted to tools using the [as_tool](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.as_tool) method, which allows for the specification of names, descriptions, and additional schema information for arguments.

## Basic usage

With typed `dict` input:
"""
logger.info("## Basic usage")


class Args(TypedDict):
    a: int
    b: List[int]


def f(x: Args) -> str:
    return str(x["a"] * max(x["b"]))


runnable = RunnableLambda(f)
as_tool = runnable.as_tool(
    name="My tool",
    description="Explanation of when to use tool.",
)

logger.debug(as_tool.description)

as_tool.args_schema.model_json_schema()

as_tool.invoke({"a": 3, "b": [1, 2]})

"""
Without typing information, arg types can be specified via `arg_types`:
"""
logger.info(
    "Without typing information, arg types can be specified via `arg_types`:")


def g(x: Dict[str, Any]) -> str:
    return str(x["a"] * max(x["b"]))


runnable = RunnableLambda(g)
as_tool = runnable.as_tool(
    name="My tool",
    description="Explanation of when to use tool.",
    arg_types={"a": int, "b": List[int]},
)

"""
Alternatively, the schema can be fully specified by directly passing the desired [args_schema](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.BaseTool.html#langchain_core.tools.BaseTool.args_schema) for the tool:
"""
logger.info(
    "Alternatively, the schema can be fully specified by directly passing the desired [args_schema](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.BaseTool.html#langchain_core.tools.BaseTool.args_schema) for the tool:")


class GSchema(BaseModel):
    """Apply a function to an integer and list of integers."""

    a: int = Field(..., description="Integer")
    b: List[int] = Field(..., description="List of ints")


runnable = RunnableLambda(g)
as_tool = runnable.as_tool(GSchema)

"""
String input is also supported:
"""
logger.info("String input is also supported:")


def f(x: str) -> str:
    return x + "a"


def g(x: str) -> str:
    return x + "z"


runnable = RunnableLambda(f) | g
as_tool = runnable.as_tool()

as_tool.invoke("b")

"""
## In agents

Below we will incorporate LangChain Runnables as tools in an [agent](/docs/concepts/agents) application. We will demonstrate with:

- a document [retriever](/docs/concepts/retrievers);
- a simple [RAG](/docs/tutorials/rag/) chain, allowing an agent to delegate relevant queries to it.

We first instantiate a chat model that supports [tool calling](/docs/how_to/tool_calling/):


<ChatModelTabs customVarName="llm" />
"""
logger.info("## In agents")


llm = ChatOllama(model="llama3.2")

"""
Following the [RAG tutorial](/docs/tutorials/rag/), let's first construct a retriever:
"""
logger.info(
    "Following the [RAG tutorial](/docs/tutorials/rag/), let's first construct a retriever:")


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
    ),
]

vectorstore = InMemoryVectorStore.from_documents(
    documents, embedding=OllamaEmbeddings(model="nomic-embed-text")
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

"""
We next create use a simple pre-built [LangGraph agent](https://python.langchain.com/docs/tutorials/agents/) and provide it the tool:
"""
logger.info(
    "We next create use a simple pre-built [LangGraph agent](https://python.langchain.com/docs/tutorials/agents/) and provide it the tool:")


tools = [
    retriever.as_tool(
        name="pet_info_retriever",
        description="Get information about pets.",
    )
]
agent = create_react_agent(llm, tools)

for chunk in agent.stream({"messages": [("human", "What are dogs known for?")]}):
    logger.debug(chunk)
    logger.debug("----")

"""
See [LangSmith trace](https://smith.langchain.com/public/44e438e3-2faf-45bd-b397-5510fc145eb9/r) for the above run.

Going further, we can create a simple [RAG](/docs/tutorials/rag/) chain that takes an additional parameter-- here, the "style" of the answer.
"""
logger.info(
    "See [LangSmith trace](https://smith.langchain.com/public/44e438e3-2faf-45bd-b397-5510fc145eb9/r) for the above run.")


system_prompt = """
You are an assistant for question-answering tasks.
Use the below context to answer the question. If
you don't know the answer, say you don't know.
Use three sentences maximum and keep the answer
concise.

Answer in the style of {answer_style}.

Question: {question}

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])

rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "answer_style": itemgetter("answer_style"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

"""
Note that the input schema for our chain contains the required arguments, so it converts to a tool without further specification:
"""
logger.info("Note that the input schema for our chain contains the required arguments, so it converts to a tool without further specification:")

rag_chain.input_schema.model_json_schema()

rag_tool = rag_chain.as_tool(
    name="pet_expert",
    description="Get information about pets.",
)

"""
Below we again invoke the agent. Note that the agent populates the required parameters in its `tool_calls`:
"""
logger.info("Below we again invoke the agent. Note that the agent populates the required parameters in its `tool_calls`:")

agent = create_react_agent(llm, [rag_tool])

for chunk in agent.stream(
    {"messages": [("human", "What would a pirate say dogs are known for?")]}
):
    logger.debug(chunk)
    logger.debug("----")

"""
See [LangSmith trace](https://smith.langchain.com/public/147ae4e6-4dfb-4dd9-8ca0-5c5b954f08ac/r) for the above run.
"""
logger.info(
    "See [LangSmith trace](https://smith.langchain.com/public/147ae4e6-4dfb-4dd9-8ca0-5c5b954f08ac/r) for the above run.")

logger.info("\n\n[DONE]", bright=True)
