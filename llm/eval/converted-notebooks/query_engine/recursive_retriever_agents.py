"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/recursive_retriever_agents.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Recursive Retriever + Document Agents

This guide shows how to combine recursive retrieval and "document agents" for advanced decision making over heterogeneous documents.

There are two motivating factors that lead to solutions for better retrieval:
- Decoupling retrieval embeddings from chunk-based synthesis. Oftentimes fetching documents by their summaries will return more relevant context to queries rather than raw chunks. This is something that recursive retrieval directly allows.
- Within a document, users may need to dynamically perform tasks beyond fact-based question-answering. We introduce the concept of "document agents" - agents that have access to both vector search and summary tools for a given document.
"""

"""
### Setup and Download Data

In this section, we'll define imports and then download Wikipedia articles about different cities. Each article is stored separately.
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-llms-ollama
# %pip install llama-index-agent-openai

# !pip install llama-index


from llama_index.agent.openai import OllamaAgent
from llama_index.core import Settings
import os
import requests
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from jet.llm.ollama import Ollama
wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]


for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

"""
Define LLM + Service Context + Callback Manager
"""


# os.environ["OPENAI_API_KEY"] = "sk-..."


Settings.llm = Ollama(temperature=0, model="llama3.2",
                      request_timeout=300.0, context_window=4096)

"""
## Build Document Agent for each Document

In this section we define "document agents" for each document.

First we define both a vector index (for semantic search) and summary index (for summarization) for each document. The two query engines are then converted into tools that are passed to an Ollama function calling agent.

This document agent can dynamically choose to perform semantic search or summarization within a given document.

We create a separate document agent for each city.
"""


agents = {}

for wiki_title in wiki_titles:
    vector_index = VectorStoreIndex.from_documents(
        city_docs[wiki_title],
    )
    summary_index = SummaryIndex.from_documents(
        city_docs[wiki_title],
    )
    vector_query_engine = vector_index.as_query_engine()
    list_query_engine = summary_index.as_query_engine()

    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    f"Useful for retrieving specific context from {wiki_title}"
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=list_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    "Useful for summarization questions related to"
                    f" {wiki_title}"
                ),
            ),
        ),
    ]

    function_llm = Ollama(
        model="llama3.2", request_timeout=300.0, context_window=4096)
    agent = OllamaAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
    )

    agents[wiki_title] = agent

"""
## Build Composable Retriever over these Agents

Now we define a set of summary nodes, where each node links to the corresponding Wikipedia city article. We then define a composable retriever + query engine on top of these Nodes to route queries down to a given node, which will in turn route it to the relevant document agent.
"""

objects = []
for wiki_title in wiki_titles:
    wiki_summary = (
        f"This content contains Wikipedia articles about {wiki_title}. Use"
        " this index if you need to lookup specific facts about"
        f" {wiki_title}.\nDo not use this index if you want to analyze"
        " multiple cities."
    )
    node = IndexNode(
        text=wiki_summary, index_id=wiki_title, obj=agents[wiki_title]
    )
    objects.append(node)

vector_index = VectorStoreIndex(
    objects=objects,
)
query_engine = vector_index.as_query_engine(similarity_top_k=1, verbose=True)

"""
## Running Example Queries
"""

response = query_engine.query("Tell me about the sports teams in Boston")

print(response)

response = query_engine.query("Tell me about the sports teams in Houston")

print(response)

response = query_engine.query(
    "Give me a summary on all the positive aspects of Chicago"
)

print(response)
