from llama_index.core.query_engine import CustomQueryEngine
import re
from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import LLM
from llama_index.core.llms import ChatMessage
from graspologic.partition import hierarchical_leiden
import networkx as nx
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,
)
from llama_index.core.async_utils import run_jobs
from IPython.display import Markdown, display
from typing import Any, List, Callable, Optional, Union, Dict
import nest_asyncio
import asyncio
import requests
from llama_index.core.agent import FunctionCallingAgent
from pathlib import Path
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import FilterCondition
from typing import List, Optional
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.core.schema import TextNode
from llama_index.core import (
    StorageContext,
    Settings,
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
)
import vertexai
import sys
import IPython
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/agentic_graph_rag_vertex.ipynb)

# Agentic GraphRAG Implementation with LlamaIndex and Vertex AI
#
#
#

# ------------
# **Pre-requisites**
# - Set up a Google Cloud project
# - Create a Google Cloud storage bucket
# - Enable Vertex AI API
#
#
# References:
# This notebook is based on below references.
#
# 1. GraphRAG Implementation with LlamaIndex
# - https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/GraphRAG_v1.ipynb
#
# 2. Building Agentic RAG with Llamaindex Tutorial
# - https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex/lesson/2/router-query-engine

# _______________

# GraphRAG Aproach
#
# The GraphRAG involves two steps:
#
# 1. Graph Generation - Creates Graph, builds communities and its summaries over the given document.
# 2. Answer to the Query - Use summaries of the communities created from step-1 to answer the query.
#
# **Graph Generation:**
#
# 1. **Source Documents to Text Chunks:** Source documents are divided into smaller text chunks for easier processing.
#
# 2. **Text Chunks to Element Instances:** Each text chunk is analyzed to identify and extract entities and relationships, resulting in a list of tuples that represent these elements.
#
# 3. **Element Instances to Element Summaries:** The extracted entities and relationships are summarized into descriptive text blocks for each element using the LLM.
#
# 4. **Element Summaries to Graph Communities:** These entities, relationships and summaries form a graph, which is subsequently partitioned into communities using algorithms using Heirarchical Leiden to establish a hierarchical structure.
#
# 5. **Graph Communities to Community Summaries:** The LLM generates summaries for each community, providing insights into the dataset’s overall topical structure and semantics.
#
# **Answering the Query:**
#
# **Community Summaries to Global Answers:** The summaries of the communities are utilized to respond to user queries. This involves generating intermediate answers, which are then consolidated into a comprehensive global answer.

# GraphRAG Pipeline Components
#
# Here are the different components we implemented to build all of the processes mentioned above.
#
# 1. **Source Documents to Text Chunks:** Implemented using `SentenceSplitter` with a chunk size of 1024 and chunk overlap of 20 tokens.
#
# 2. **Text Chunks to Element Instances AND Element Instances to Element Summaries:** Implemented using `GraphRAGExtractor`.
#
# 3. **Element Summaries to Graph Communities AND Graph Communities to Community Summaries:** Implemented using `GraphRAGStore`.
#
# 4. **Community Summaries to Global Answers:** Implemented using `GraphQueryEngine`.
#
#
# Let's check into each of these components and build GraphRAG pipeline.

# Installation
#
# `graspologic` is used to use hierarchical_leiden for building communities.

# !pip install llama-index graspologic numpy==1.24.4 scipy==1.12.0

# !pip install --upgrade google-cloud-aiplatform llama-index-vector-stores-vertexaivectorsearch llama-index llama_index-llms-vertex


app = IPython.Application.instance()
app.kernel.do_shutdown(True)

# Authentication


if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()

# Set Google Cloud project information and initialize Vertex AI SDK
#
# To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).
#
# Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).

PROJECT_ID = "<your project>"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}


vertexai.init(project=PROJECT_ID, location=LOCATION)


embed_model = VertexTextEmbedding(
    model_name="text-embedding-004",
    project=PROJECT_ID,
    location=LOCATION,
)

vertex_gemini = Vertex(
    model="gemini-1.5-pro",
    temperature=0,
    context_window=100000,
    additional_kwargs={},
)

Settings.embed_model = embed_model
Settings.llm = vertex_gemini

llm = vertex_gemini

# Load Data
#

# Prepare documents as required by LlamaIndex

urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=LzPWWPAdY4",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=hSyW5go0v8",
    "https://openreview.net/pdf?id=9WD9KwssyT",
    "https://openreview.net/pdf?id=yV6fD7LYkF",
    "https://openreview.net/pdf?id=hnrB5YHoYu",
    "https://openreview.net/pdf?id=WbWtOYIzIK",
    "https://openreview.net/pdf?id=c5pwL0Soay",
    "https://openreview.net/pdf?id=TpD2aG1h0D",
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf",
    "swebench.pdf",
    "selfrag.pdf",
    "zipformer.pdf",
    "values.pdf",
    "finetune_fair_diffusion.pdf",
    "knowledge_card.pdf",
    "metra.pdf",
    "vr_mcl.pdf",
]


def download_file(url, file_path):
    """Downloads a file from a given URL and saves it to the specified file path.

    Args:
        url: The URL of the file to download.
        file_path: The path to save the downloaded file.
    """

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for non-200 status codes

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # Filter out keep-alive new chunks
                f.write(chunk)

    print(f"Downloaded file from {url} to {file_path}")


for url, paper in zip(urls, papers):
    download_file(url, paper)

# GraphRAGExtractor
#
# The GraphRAGExtractor class is designed to extract triples (subject-relation-object) from text and enrich them by adding descriptions for entities and relationships to their properties using an LLM.
#
# This functionality is similar to that of the `SimpleLLMPathExtractor`, but includes additional enhancements to handle entity, relationship descriptions. For guidance on implementation, you may look at similar existing [extractors](https://docs.llamaindex.ai/en/latest/examples/property_graph/Dynamic_KG_Extraction/?h=comparing).
#
# Here's a breakdown of its functionality:
#
# **Key Components:**
#
# 1. `llm:` The language model used for extraction.
# 2. `extract_prompt:` A prompt template used to guide the LLM in extracting information.
# 3. `parse_fn:` A function to parse the LLM's output into structured data.
# 4. `max_paths_per_chunk:` Limits the number of triples extracted per text chunk.
# 5. `num_workers:` For parallel processing of multiple text nodes.
#
#
# **Main Methods:**
#
# 1. `__call__:` The entry point for processing a list of text nodes.
# 2. `acall:` An asynchronous version of __call__ for improved performance.
# 3. `_aextract:` The core method that processes each individual node.
#
#
# **Extraction Process:**
#
# For each input node (chunk of text):
# 1. It sends the text to the LLM along with the extraction prompt.
# 2. The LLM's response is parsed to extract entities, relationships, descriptions for entities and relations.
# 3. Entities are converted into EntityNode objects. Entity description is stored in metadata
# 4. Relationships are converted into Relation objects. Relationship description is stored in metadata.
# 5. These are added to the node's metadata under KG_NODES_KEY and KG_RELATIONS_KEY.
#
# **NOTE:** In the current implementation, we are using only relationship descriptions. In the next implementation, we will utilize entity descriptions during the retrieval stage.


nest_asyncio.apply()


class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = self.llm.predict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            metadata[
                "entity_description"
                # Not used in the current implementation. But will be useful in future work.
            ] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=metadata
            )
            existing_nodes.append(entity_node)

        metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, rel, obj, description = triple
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

# GraphRAGStore
#
# The `GraphRAGStore` class is an extension of the `SimplePropertyGraphStore `class, designed to implement GraphRAG pipeline. Here's a breakdown of its key components and functions:
#
#
# The class uses community detection algorithms to group related nodes in the graph and then it generates summaries for each community using an LLM.
#
#
# **Key Methods:**
#
# `build_communities():`
#
# 1. Converts the internal graph representation to a NetworkX graph.
#
# 2. Applies the hierarchical Leiden algorithm for community detection.
#
# 3. Collects detailed information about each community.
#
# 4. Generates summaries for each community.
#
# `generate_community_summary(text):`
#
# 1. Uses LLM to generate a summary of the relationships in a community.
# 2. The summary includes entity names and a synthesis of relationship descriptions.
#
# `_create_nx_graph():`
#
# 1. Converts the internal graph representation to a NetworkX graph for community detection.
#
# `_collect_community_info(nx_graph, clusters):`
#
# 1. Collects detailed information about each node based on its community.
# 2. Creates a string representation of each relationship within a community.
#
# `_summarize_communities(community_info):`
#
# 1. Generates and stores summaries for each community using LLM.
#
# `get_community_summaries():`
#
# 1. Returns the community summaries by building them if not already done.


class GraphRAGStore(SimplePropertyGraphStore):
    community_summary = {}
    max_cluster_size = 5

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = llm.chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """Collect detailed information for each node based on their community."""
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
        return community_info

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

# GraphRAGQueryEngine
#
# The GraphRAGQueryEngine class is a custom query engine designed to process queries using the GraphRAG approach. It leverages the community summaries generated by the GraphRAGStore to answer user queries. Here's a breakdown of its functionality:
#
# **Main Components:**
#
# `graph_store:` An instance of GraphRAGStore, which contains the community summaries.
# `llm:` A Language Model (LLM) used for generating and aggregating answers.
#
#
# **Key Methods:**
#
# `custom_query(query_str: str)`
#
# 1. This is the main entry point for processing a query. It retrieves community summaries, generates answers from each summary, and then aggregates these answers into a final response.
#
# `generate_answer_from_summary(community_summary, query):`
#
# 1. Generates an answer for the query based on a single community summary.
# Uses the LLM to interpret the community summary in the context of the query.
#
# `aggregate_answers(community_answers):`
#
# 1. Combines individual answers from different communities into a coherent final response.
# 2. Uses the LLM to synthesize multiple perspectives into a single, concise answer.
#
#
# **Query Processing Flow:**
#
# 1. Retrieve community summaries from the graph store.
# 2. For each community summary, generate a specific answer to the query.
# 3. Aggregate all community-specific answers into a final, coherent response.
#
#
# **Example usage:**
#
# ```
# query_engine = GraphRAGQueryEngine(graph_store=graph_store, llm=llm)
#
# response = query_engine.query("query")
# ```


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for _, community_summary in community_summaries.items()
        ]

        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response

# Build End to End GraphRAG Pipeline
#
# Now that we have defined all the necessary components, let’s construct the GraphRAG pipeline:
#
# 1. Create nodes/chunks from the text.
# 2. Build a PropertyGraphIndex using `GraphRAGExtractor` and `GraphRAGStore`.
# 3. Construct communities and generate a summary for each community using the graph built above.
# 4. Create a `GraphRAGQueryEngine` and begin querying.


documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()

# Create nodes/ chunks from the text.


splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)

len(nodes)

# Build PropertyGraphIndex using `GraphRAGExtractor` and `GraphRAGStore`

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$<entity_name>$$$$<entity_type>$$$$<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_description>)

3. When finished, output.

-Real Data-
text: {text}
output:"""

entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'


def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships


kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=2,
    parse_fn=parse_fn,
)


index = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=GraphRAGStore(),
    kg_extractors=[kg_extractor],
    show_progress=True,
)

list(index.property_graph_store.graph.nodes.values())[-1]

list(index.property_graph_store.graph.relations.values())[0]

list(index.property_graph_store.graph.relations.values())[0].properties[
    "relationship_description"
]

# Build communities
#
# This will create communities and summary for each community.

index.property_graph_store.build_communities()

# Create QueryEngine

query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store, llm=llm
)

# Querying

response = query_engine.query("What are the main discussed in the document?")
display(Markdown(f"{response.response}"))

# Building an Agent Reasoning Loop


def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    index = PropertyGraphIndex(
        nodes=nodes,
        property_graph_store=GraphRAGStore(),
        kg_extractors=[kg_extractor],
        show_progress=True,
    )
    index.property_graph_store.build_communities()
    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store, llm=llm
    )

    summary_index = SummaryIndex(nodes)

    def vector_query(
        query: str, page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over the MetaGPT paper.

        Useful if you have specific questions over the MetaGPT paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.

        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.

        """
        vector_query_engine = query_engine
        response = vector_query_engine.query(query)
        return response

    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}", fn=vector_query
    )

    def summary_query(
        query: str,
    ) -> str:
        """Perform a summary of document
        query (str): the string query to be embedded.
        """
        summary_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )

        response = summary_engine.query(query)
        return response

    summary_tool = FunctionTool.from_defaults(
        fn=summary_query, name=f"summary_tool_{name}"
    )

    return vector_query_tool, summary_tool


vector_query_tool, summary_tool = get_doc_tools("metagpt.pdf", "metagpt")

agent = FunctionCallingAgent.from_tools(
    [summary_tool, vector_query_tool], llm=vertex_gemini, verbose=True
)

response = agent.query(
    "what are agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

logger.info("\n\n[DONE]", bright=True)
