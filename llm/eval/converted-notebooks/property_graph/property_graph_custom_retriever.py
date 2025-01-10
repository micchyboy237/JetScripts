from llama_index.core.query_engine import RetrieverQueryEngine
from typing import Optional, Any, Union
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.retrievers import (
    CustomPGRetriever,
    VectorContextRetriever,
    TextToCypherRetriever,
)
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader
import os
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Defining a Custom Property Graph Retriever
#
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_custom_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
#
# This guide shows you how to define a custom retriever against a property graph.
#
# It is more involved than using our out-of-the-box graph retrievers, but allows you to have granular control over the retrieval process so that it's better tailored for your application.
#
# We show you how to define an advanced retrieval flow by directly leveraging the property graph store. We'll execute both vector search and text-to-cypher retrieval, and then combine the results through a reranking module.

# %pip install llama-index
# %pip install llama-index-graph-stores-neo4j
# %pip install llama-index-postprocessor-cohere-rerank

# Setup and Build the Property Graph


nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "sk-..."

# Load Paul Graham Essay

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

# Define Default LLMs


llm = Ollama(model="llama3.2", request_timeout=300.0,
             context_window=4096, temperature=0.3)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Setup Neo4j
#
# To launch Neo4j locally, first ensure you have docker installed. Then, you can launch the database with the following docker command
#
# ```
# docker run \
#     -p 7474:7474 -p 7687:7687 \
#     -v $PWD/data:/data -v $PWD/plugins:/plugins \
#     --name neo4j-apoc \
#     -e NEO4J_apoc_export_file_enabled=true \
#     -e NEO4J_apoc_import_file_enabled=true \
#     -e NEO4J_apoc_import_file_use__neo4j__config=true \
#     -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
#     neo4j:latest
#
# ```
# From here, you can open the db at http://localhost:7474/. On this page, you will be asked to sign in. Use the default username/password of neo4j and neo4j.


graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
)

# Build the Property Graph


index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)

# Define Custom Retriever
#
# Now we define a custom retriever by subclassing `CustomPGRetriever`.
#
# 1. Initialization
# We initialize two pre-existing property graph retrievers: the `VectorContextRetriever` and the `TextToCypherRetriever`, as well as the cohere reranker.
#
# 2. Define `custom_retrieve`
#
# We then define the `custom_retrieve` function. It passes nodes through the two retrievers and gets back a final ranked list.
#
# The return type here can be a string, `TextNode`, `NodeWithScore`, or a list of one of those types.


class MyCustomRetriever(CustomPGRetriever):
    """Custom retriever with cohere reranking."""

    def init(
        self,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None,
        similarity_top_k: int = 4,
        path_depth: int = 1,
        llm: Optional[LLM] = None,
        text_to_cypher_template: Optional[Union[PromptTemplate, str]] = None,
        cohere_api_key: Optional[str] = None,
        cohere_top_n: int = 2,
        **kwargs: Any,
    ) -> None:
        """Uses any kwargs passed in from class constructor."""

        self.vector_retriever = VectorContextRetriever(
            self.graph_store,
            include_text=self.include_text,
            embed_model=embed_model,
            vector_store=vector_store,
            similarity_top_k=similarity_top_k,
            path_depth=path_depth,
        )

        self.cypher_retriever = TextToCypherRetriever(
            self.graph_store,
            llm=llm,
            text_to_cypher_template=text_to_cypher_template
        )

        self.reranker = CohereRerank(
            api_key=cohere_api_key, top_n=cohere_top_n
        )

    def custom_retrieve(self, query_str: str) -> str:
        """Define custom retriever with reranking.

        Could return `str`, `TextNode`, `NodeWithScore`, or a list of those.
        """
        nodes_1 = self.vector_retriever.retrieve(query_str)
        nodes_2 = self.cypher_retriever.retrieve(query_str)
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes_1 + nodes_2, query_str=query_str
        )

        final_text = "\n\n".join(
            [n.get_content(metadata_mode="llm") for n in reranked_nodes]
        )

        return final_text

# Test out the Custom Retriever
#
# Now let's initialize and test out the custom retriever against our data!
#
# To build a full RAG pipeline, we use the `RetrieverQueryEngine` to combine our retriever with the LLM synthesis module - this is also used under the hood for the property graph index.


custom_sub_retriever = MyCustomRetriever(
    index.property_graph_store,
    include_text=True,
    vector_store=index.vector_store,
    cohere_api_key="...",
)


query_engine = RetrieverQueryEngine.from_args(
    index.as_retriever(sub_retrievers=[custom_sub_retriever]), llm=llm
)

# Try out a 'baseline'
#
# We compare against a baseline retriever that's the vector context only.

base_retriever = VectorContextRetriever(
    index.property_graph_store, include_text=True
)
base_query_engine = index.as_query_engine(sub_retrievers=[base_retriever])

# Try out some Queries

response = query_engine.query("Did the author like programming?")
print(str(response))

response = base_query_engine.query("Did the author like programming?")
print(str(response))

logger.info("\n\n[DONE]", bright=True)
