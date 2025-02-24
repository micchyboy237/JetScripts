from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CustomQueryEngine
import re
from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import LLM
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.llms import ChatMessage
from collections import defaultdict
from graspologic.partition import hierarchical_leiden
import networkx as nx
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
from jet.llm.ollama.base import Ollama
import os
from llama_index.core import Document
import pandas as pd
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# GraphRAG Implementation with LlamaIndex - V2
#
# [GraphRAG (Graphs + Retrieval Augmented Generation)](https://www.microsoft.com/en-us/research/project/graphrag/) combines the strengths of Retrieval Augmented Generation (RAG) and Query-Focused Summarization (QFS) to effectively handle complex queries over large text datasets. While RAG excels in fetching precise information, it struggles with broader queries that require thematic understanding, a challenge that QFS addresses but cannot scale well. GraphRAG integrates these approaches to offer responsive and thorough querying capabilities across extensive, diverse text corpora.
#
# This notebook provides guidance on constructing the GraphRAG pipeline using the LlamaIndex PropertyGraph abstractions using Neo4J.
#
# This notebook updates the GraphRAG pipeline to v2. If you haven’t checked v1 yet, you can find it [here](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/GraphRAG_v1.ipynb). Following are the updates to the existing implementation:
#
# 1. Integrate with Neo4J Graph database.
# 2. Embedding based retrieval.
#

# Installation
#
# `graspologic` is used to use hierarchical_leiden for building communities.

# !pip install llama-index llama-index-graph-stores-neo4j graspologic numpy==1.24.4 scipy==1.12.0 future

# Load Data
#
# We will use a sample news article dataset retrieved from Diffbot, which Tomaz has conveniently made available on GitHub for easy access.
#
# The dataset contains 2,500 samples; for ease of experimentation, we will use 50 of these samples, which include the `title` and `text` of news articles.


news = pd.read_csv(
    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
)[:50]

news.head()

# Prepare documents as required by LlamaIndex

documents = [
    Document(text=f"{row['title']}: {row['text']}")
    for i, row in news.iterrows()
]

# Setup API Key and LLM


# os.environ["OPENAI_API_KEY"] = "sk-.."


llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)

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
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

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
# The `GraphRAGStore` class is an extension of the `Neo4jPropertyGraphStore`class, designed to implement GraphRAG pipeline. Here's a breakdown of its key components and functions:
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


class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    entity_info = None
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
        response = Ollama().chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{
                        node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

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
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""

        entities = self.get_entities(query_str, self.similarity_top_k)

        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ]

        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def get_entities(self, query_str, similarity_top_k):
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        enitites = set()
        pattern = (
            r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"
        )

        for node in nodes_retrieved:
            matches = re.findall(
                pattern, node.text, re.MULTILINE | re.IGNORECASE
            )

            for match in matches:
                subject = match[0]
                obj = match[2]
                enitites.add(subject)
                enitites.add(obj)

        return list(enitites)

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

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

# Create nodes/ chunks from the text.


splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)

len(nodes)

# Build ProperGraphIndex using `GraphRAGExtractor` and `GraphRAGStore`

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$"<entity_name>"$$$$"<entity_type>"$$$$"<entity_description>")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$"<source_entity>"$$$$"<target_entity>"$$$$"<relation>"$$$$"<relationship_description>")

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

# Docker Setup And Neo4J setup
#
# To launch Neo4j locally, first ensure you have docker installed. Then, you can launch the database with the following docker command.
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
# ```
# From here, you can open the db at http://localhost:7474/. On this page, you will be asked to sign in. Use the default username/password of neo4j and neo4j.
#
# Once you login for the first time, you will be asked to change the password.


graph_store = GraphRAGStore(
    username="neo4j", password="<PASSWORD>", url="bolt://localhost:7687"
)


index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    show_progress=True,
)

index.property_graph_store.get_triplets()[10]

index.property_graph_store.get_triplets()[10][0].properties

index.property_graph_store.get_triplets()[10][1].properties

# Build communities
#
# This will create communities and summary for each community.

index.property_graph_store.build_communities()

# Create QueryEngine

query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store,
    llm=llm,
    index=index,
    similarity_top_k=10,
)

# Querying

query = "What are the main news discussed in the document?"
response = query_engine.query(query)
display_jet_source_nodes(query, response)

query = "What are the main news in energy sector?"
response = query_engine.query(query)
display_jet_source_nodes(query, response)

logger.info("\n\n[DONE]", bright=True)
