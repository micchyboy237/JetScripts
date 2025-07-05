from collections import defaultdict
from jet.llm.ollama.base_langchain import ChatOllama, OllamaEmbeddings
from jet.logger import CustomLogger
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pymilvus import MilvusClient
from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/graphrag_with_milvus_vectordb.ipynb)

# Graph RAG with Milvus Vector Database

## Overview

### What You'll Learn
This notebook demonstrates an innovative approach to **Graph RAG (Retrieval-Augmented Generation)** that combines the power of knowledge graphs with vector databases to dramatically improve question-answering performance, especially for complex multi-hop queries. By the end of this tutorial, you'll understand how to build a Graph RAG system that can answer questions requiring multiple logical steps and relationship traversals.

### The Problem: Limitations of Traditional RAG
Traditional RAG systems, while powerful, struggle with:
- **Multi-hop questions**: Queries requiring multiple logical steps (e.g., "What contribution did the son of Euler's teacher make?")
- **Complex entity relationships**: Understanding how different entities connect and relate to each other
- **Context fragmentation**: Important relationships may be scattered across different text passages
- **Semantic gaps**: Simple similarity search may miss logically relevant but semantically distant information

### The Solution: Graph RAG with Vector Database
This notebook presents a **unified approach** that achieves Graph RAG capabilities using **only a vector database** (Milvus), eliminating the need for separate graph databases while maintaining superior performance. Here's what makes this approach special:

**Key Innovation**: Instead of storing explicit graph structures, we embed entities and relationships as vectors and use intelligent retrieval and expansion techniques to reconstruct graph-like reasoning paths.

### Key Benefits
1. **Simplified Architecture**: Single vector database instead of vector DB + graph DB combination
2. **Superior Multi-hop Performance**: Handles complex queries requiring multiple relationship traversals
3. **Scalable**: Leverages Milvus's distributed architecture for billion-scale deployments
4. **Cost-effective**: Reduces infrastructure complexity and operational overhead
5. **Flexible**: Works with any text corpus - just extract entities and relationships

### Methodology Overview
Our approach consists of four main stages:

1. **Offline Data Preparation**
   - Extract entities and relationships (triplets) from your text corpus
   - Create three vector collections: entities, relationships, and passages
   - Build adjacency mappings between entities and relationships

2. **Query-time Retrieval**
   - Retrieve similar entities and relationships using vector similarity search
   - Use Named Entity Recognition (NER) to identify query entities

3. **Subgraph Expansion** 
   - Expand retrieved entities/relationships to their neighborhood using adjacency matrices
   - Support multi-degree expansion (1-hop, 2-hop neighbors)
   - Merge results from both entity and relationship expansion paths

4. **LLM Reranking**
   - Use large language models to intelligently filter and rank candidate relationships
   - Apply Chain-of-Thought reasoning to select most relevant relationships
   - Return final passages for answer generation

### Architecture Diagram
The following diagram illustrates the complete workflow:

![](../images/graph_rag_with_milvus_1.png)

**Why This Works**: By representing both entities and relationships as vectors, we can leverage semantic similarity for initial retrieval, then use graph-theoretical expansion to discover indirect relationships, and finally apply LLM reasoning to filter for relevance. This creates a "best of both worlds" system that combines semantic search, graph traversal, and intelligent reasoning.

---

## Technical Implementation

In this section, we'll implement the Graph RAG system described in our methodology overview. The implementation follows our four-stage approach: data preparation, vector storage, query processing, and intelligent reranking.

## Prerequisites

To complete this demo, you need a vector database. You can get a fully-managed Milvus vector database for free by [signing up Zilliz Cloud](https://cloud.zilliz.com/signup?utm_source=github&utm_medium=referral&utm_campaign=Nir-250512). Milvus is an open-source vector database that provides high-performance vector similarity search.

Install the following dependencies:
"""
logger.info("# Graph RAG with Milvus Vector Database")

# ! pip install --upgrade --quiet pymilvus numpy scipy langchain langchain-core langchain-openai tqdm

"""
> If you are using Google Colab, to enable dependencies just installed, you may need to **restart the runtime** (click on the "Runtime" menu at the top of the screen, and select "Restart session" from the dropdown menu).

# We will use the models from Ollama. You need to prepare the [`OPENAI_API_KEY`](https://platform.openai.com/docs/quickstart) as an environment variable.
"""
# logger.info("We will use the models from Ollama. You need to prepare the [`OPENAI_API_KEY`](https://platform.openai.com/docs/quickstart) as an environment variable.")


# os.environ["OPENAI_API_KEY"] = "sk-***********"

"""
Import the necessary libraries and dependencies.
"""
logger.info("Import the necessary libraries and dependencies.")



"""
Find your Public Endpoint and Token (i.e., API Key) on the Zilliz Cloud page.

![](../images/zilliz_interface.png)
"""
logger.info("Find your Public Endpoint and Token (i.e., API Key) on the Zilliz Cloud page.")

milvus_client = MilvusClient(
    uri="YOUR_ZILLIZ_PUBLIC_ENDPOINT",
    token="YOUR_ZILLIZ_TOKEN"
)

llm = ChatOllama(
    model="llama3.1", request_timeout=300.0, context_window=4096,
    temperature=0,
)
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

"""
## Offline Data Loading

### Understanding the Data Model

Before diving into the implementation, it's crucial to understand how we structure our data to enable graph-like reasoning with vectors. Our approach transforms traditional text documents into three interconnected components:

1. **Entities**: The "nodes" of our conceptual graph - people, places, concepts, etc.
2. **Relationships**: The "edges" connecting entities - these are full triplets (subject-predicate-object)
3. **Passages**: The original text documents that provide context and detailed information

**Why This Structure Works**: By separating entities and relationships into distinct vector collections, we can perform targeted searches for different aspects of a query. When a user asks "What contribution did the son of Euler's teacher make?", we can:
- Find entities related to "Euler" 
- Find relationships that connect teacher-student and parent-child concepts
- Expand the graph to discover indirect connections
- Retrieve the most relevant passages for final answer generation

### Data Preparation

We will use a nano dataset which introduce the relationship between Bernoulli family and Euler to demonstrate as an example. The nano dataset contains 4 passages and a set of corresponding triplets, where each triplet contains a subject, a predicate, and an object.

**Triplet Structure**: Each relationship is represented as a triplet [Subject, Predicate, Object]. For example:
- `["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"]` captures a family relationship
- `["Johann Bernoulli", "was a student of", "Leonhard Euler"]` captures an educational relationship

In practice, you can use any approach to extract the triplets from your own custom corpus. Common methods include:
- **Named Entity Recognition (NER)** + **Relation Extraction** models
- **Open Information Extraction** systems like OpenIE
- **Large Language Models** with structured prompting
- **Manual annotation** for high-precision domains
"""
logger.info("## Offline Data Loading")

nano_dataset = [
    {
        "passage": "Jakob Bernoulli (1654–1705): Jakob was one of the earliest members of the Bernoulli family to gain prominence in mathematics. He made significant contributions to calculus, particularly in the development of the theory of probability. He is known for the Bernoulli numbers and the Bernoulli theorem, a precursor to the law of large numbers. He was the older brother of Johann Bernoulli, another influential mathematician, and the two had a complex relationship that involved both collaboration and rivalry.",
        "triplets": [
            ["Jakob Bernoulli", "made significant contributions to", "calculus"],
            [
                "Jakob Bernoulli",
                "made significant contributions to",
                "the theory of probability",
            ],
            ["Jakob Bernoulli", "is known for", "the Bernoulli numbers"],
            ["Jakob Bernoulli", "is known for", "the Bernoulli theorem"],
            ["The Bernoulli theorem", "is a precursor to", "the law of large numbers"],
            ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"],
        ],
    },
    {
        "passage": "Johann Bernoulli (1667–1748): Johann, Jakob’s younger brother, was also a major figure in the development of calculus. He worked on infinitesimal calculus and was instrumental in spreading the ideas of Leibniz across Europe. Johann also contributed to the calculus of variations and was known for his work on the brachistochrone problem, which is the curve of fastest descent between two points.",
        "triplets": [
            [
                "Johann Bernoulli",
                "was a major figure of",
                "the development of calculus",
            ],
            ["Johann Bernoulli", "was", "Jakob's younger brother"],
            ["Johann Bernoulli", "worked on", "infinitesimal calculus"],
            ["Johann Bernoulli", "was instrumental in spreading", "Leibniz's ideas"],
            ["Johann Bernoulli", "contributed to", "the calculus of variations"],
            ["Johann Bernoulli", "was known for", "the brachistochrone problem"],
        ],
    },
    {
        "passage": "Daniel Bernoulli (1700–1782): The son of Johann Bernoulli, Daniel made major contributions to fluid dynamics, probability, and statistics. He is most famous for Bernoulli’s principle, which describes the behavior of fluid flow and is fundamental to the understanding of aerodynamics.",
        "triplets": [
            ["Daniel Bernoulli", "was the son of", "Johann Bernoulli"],
            ["Daniel Bernoulli", "made major contributions to", "fluid dynamics"],
            ["Daniel Bernoulli", "made major contributions to", "probability"],
            ["Daniel Bernoulli", "made major contributions to", "statistics"],
            ["Daniel Bernoulli", "is most famous for", "Bernoulli’s principle"],
            [
                "Bernoulli’s principle",
                "is fundamental to",
                "the understanding of aerodynamics",
            ],
        ],
    },
    {
        "passage": "Leonhard Euler (1707–1783) was one of the greatest mathematicians of all time, and his relationship with the Bernoulli family was significant. Euler was born in Basel and was a student of Johann Bernoulli, who recognized his exceptional talent and mentored him in mathematics. Johann Bernoulli’s influence on Euler was profound, and Euler later expanded upon many of the ideas and methods he learned from the Bernoullis.",
        "triplets": [
            [
                "Leonhard Euler",
                "had a significant relationship with",
                "the Bernoulli family",
            ],
            ["leonhard Euler", "was born in", "Basel"],
            ["Leonhard Euler", "was a student of", "Johann Bernoulli"],
            ["Johann Bernoulli's influence", "was profound on", "Euler"],
        ],
    },
]

"""
We construct the entities and relations as follows:
- The entity is the subject or object in the triplet, so we directly extract them from the triplets.
- Here we construct the concept of relationship by directly concatenating the subject, predicate, and object with a space in between.

We also prepare a dict to map entity id to relation id, and another dict to map relation id to passage id for later use.

### Building the Knowledge Graph Structure

The next step transforms our triplets into a searchable vector format while maintaining the graph connectivity information. This process involves several key decisions:

**Entity Extraction Strategy**: We extract unique entities by collecting all subjects and objects from our triplets. This ensures we capture every entity mentioned in any relationship, creating comprehensive coverage of our knowledge domain.

**Relationship Representation**: Rather than storing relationships as separate subject-predicate-object components, we concatenate them into natural language sentences. For example, `["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"]` becomes `"Jakob Bernoulli was the older brother of Johann Bernoulli"`. This approach offers several advantages:
- **Semantic richness**: The full sentence provides more context for vector embeddings
- **Natural language compatibility**: LLMs can easily understand and reason about complete sentences
- **Reduced complexity**: No need to manage separate predicate vocabularies

**Adjacency Mapping Construction**: We build two critical mapping structures:
1. **`entityid_2_relationids`**: Maps each entity to all relationships it participates in (enables entity-to-relationship expansion)
2. **`relationid_2_passageids`**: Maps each relationship to the passages where it appears (enables relationship-to-passage retrieval)

These mappings are essential for the subgraph expansion process, allowing us to efficiently traverse the conceptual graph during query time.
"""
logger.info("### Building the Knowledge Graph Structure")

entityid_2_relationids = defaultdict(list)
relationid_2_passageids = defaultdict(list)

entities = []
relations = []
passages = []
for passage_id, dataset_info in enumerate(nano_dataset):
    passage, triplets = dataset_info["passage"], dataset_info["triplets"]
    passages.append(passage)
    for triplet in triplets:
        if triplet[0] not in entities:
            entities.append(triplet[0])
        if triplet[2] not in entities:
            entities.append(triplet[2])
        relation = " ".join(triplet)
        if relation not in relations:
            relations.append(relation)
            entityid_2_relationids[entities.index(triplet[0])].append(
                len(relations) - 1
            )
            entityid_2_relationids[entities.index(triplet[2])].append(
                len(relations) - 1
            )
        relationid_2_passageids[relations.index(relation)].append(passage_id)

"""
### Data Insertion

Create Milvus collections for entity, relation, and passage. We create three separate Milvus collections, each optimized for different types of retrieval:

1. **Entity Collection**: Stores vector embeddings of entity names and descriptions
   - **Purpose**: Enables entity-centric queries like "find entities similar to 'Euler'"
   - **Search pattern**: Direct semantic similarity to query entities

2. **Relationship Collection**: Stores vector embeddings of complete relationship sentences  
   - **Purpose**: Captures semantic patterns in relationships that match query intent
   - **Search pattern**: Finds relationships semantically similar to the entire query

3. **Passage Collection**: Stores vector embeddings of original text passages
   - **Purpose**: Provides comparison baseline and detailed context for final answers
   - **Search pattern**: Traditional RAG-style document retrieval

**Why Three Collections?** This separation allows for **multi-modal retrieval**:
- If a query mentions specific entities, we retrieve through the entity collection
- If a query describes relationships or actions, we retrieve through the relationship collection  
- We can combine results from both paths and compare against traditional passage retrieval

**Embedding Consistency**: All collections use the same embedding model to ensure compatibility during similarity searches and result merging.
"""
logger.info("### Data Insertion")

embedding_dim = len(embedding_model.embed_query("foo"))


def create_milvus_collection(collection_name: str):
    """
    Create a new Milvus collection with specified configuration.

    This function creates a new Milvus collection for storing vector embeddings.
    If a collection with the same name already exists, it will be dropped first
    to ensure a clean state.

    Args:
        collection_name (str): The name of the collection to create.
    """
    if milvus_client.has_collection(collection_name=collection_name):
        milvus_client.drop_collection(collection_name=collection_name)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        consistency_level="Strong",
    )


entity_col_name = "entity_collection"
relation_col_name = "relation_collection"
passage_col_name = "passage_collection"
create_milvus_collection(entity_col_name)
create_milvus_collection(relation_col_name)
create_milvus_collection(passage_col_name)

"""
Insert the data with their metadata information into Milvus collections, including entity, relation, and passage collections. The metadata information includes the passage id and the adjacency entity or relation id.
"""
logger.info("Insert the data with their metadata information into Milvus collections, including entity, relation, and passage collections. The metadata information includes the passage id and the adjacency entity or relation id.")

def milvus_insert(
    collection_name: str,
    text_list: list[str],
):
    """
    Insert text data with embeddings into a Milvus collection in batches.

    This function processes a list of text strings, generates embeddings for them,
    and inserts the data into the specified Milvus collection in batches for
    efficient processing.

    Args:
        collection_name (str): The name of the Milvus collection to insert data into.
        text_list (list[str]): A list of text strings to be embedded and inserted.
    """
    batch_size = 512
    for row_id in tqdm(range(0, len(text_list), batch_size), desc="Inserting"):
        batch_texts = text_list[row_id : row_id + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch_texts)

        batch_ids = [row_id + j for j in range(len(batch_texts))]
        batch_data = [
            {
                "id": id_,
                "text": text,
                "vector": vector,
            }
            for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
        ]
        milvus_client.insert(
            collection_name=collection_name,
            data=batch_data,
        )


milvus_insert(
    collection_name=relation_col_name,
    text_list=relations,
)

milvus_insert(
    collection_name=entity_col_name,
    text_list=entities,
)

milvus_insert(
    collection_name=passage_col_name,
    text_list=passages,
)

"""
## Online Querying

### Understanding the Query Processing Pipeline

The querying phase implements our core innovation: combining semantic vector search with graph traversal logic. This multi-stage process transforms a natural language question into relevant knowledge by following these steps:

1. **Entity Identification**: Extract entities mentioned in the query using NER
2. **Dual Retrieval**: Search both entity and relationship collections simultaneously  
3. **Graph Expansion**: Use adjacency information to discover indirect connections
4. **LLM Reranking**: Apply intelligent filtering to select the most relevant relationships
5. **Answer Generation**: Retrieve final passages and generate the response

### Similarity Retrieval

We retrieve the topK similar entities and relations based on the input query from Milvus.

When performing the entity retrieving, we should first extract the query entities from the query text using some specific method like NER (Named-entity recognition). For simplicity, we prepare the NER results here. If you want to change the query as your custom question, you have to change the corresponding query NER list.
In practice, you can use any other model or approach to extract the entities from the query.

### Dual-Path Retrieval Strategy

Our approach performs two parallel similarity searches:

**Path 1: Entity-Based Retrieval**
- **Input**: Extracted entities from the query (using NER)  
- **Process**: Find entities in our knowledge base similar to query entities
- **Why NER?**: Many complex queries reference specific entities ("Euler", "Bernoulli family"). By identifying these explicitly, we can find direct matches and their associated relationships
- **Example**: For "What contribution did the son of Euler's teacher make?", NER identifies "Euler" as a key entity

**Path 2: Relationship-Based Retrieval**  
- **Input**: The complete query text
- **Process**: Find relationships that semantically match the query's intent
- **Purpose**: Captures the relational patterns and question structure
- **Example**: The query pattern "contribution did the son of X's teacher make" matches relationship patterns about family connections and contributions

**Benefits of Dual Retrieval**:
- **Comprehensive coverage**: Entity path catches direct mentions, relationship path catches semantic patterns
- **Redundancy for robustness**: If one path misses relevant information, the other might capture it
- **Different granularities**: Entities provide specific anchors, relationships provide structural patterns
"""
logger.info("## Online Querying")

query = "What contribution did the son of Euler's teacher make?"

query_ner_list = ["Euler"]

query_ner_embeddings = [
    embedding_model.embed_query(query_ner) for query_ner in query_ner_list
]

top_k = 3

entity_search_res = milvus_client.search(
    collection_name=entity_col_name,
    data=query_ner_embeddings,
    limit=top_k,
    output_fields=["id"],
)

query_embedding = embedding_model.embed_query(query)

relation_search_res = milvus_client.search(
    collection_name=relation_col_name,
    data=[query_embedding],
    limit=top_k,
    output_fields=["id"],
)[0]

"""
### Expand Subgraph

We use the retrieved entities and relations to expand the subgraph and obtain the candidate relationships, and then merge them from the two ways. Here is a flow chart of the subgraph expansion process:
![](../images/graph_rag_with_milvus_2.png)

Here we construct an adjacency matrix and use matrix multiplication to calculate the adjacency mapping information within a few degrees. In this way, we can quickly obtain information of any degree of expansion.

### The Mathematics of Graph Expansion

The subgraph expansion step is where our approach truly shines. Instead of storing an explicit graph database, we use **adjacency matrices** and **matrix multiplication** to efficiently compute multi-hop relationships. This mathematical approach offers several advantages:

**Adjacency Matrix Construction**: We create a binary matrix where `entity_relation_adj[i][j] = 1` if entity `i` participates in relationship `j`, and 0 otherwise. This sparse representation captures the entire graph structure.

**Multi-Degree Expansion via Matrix Powers**:
- **1-degree expansion**: `entity_adj_1_degree = entity_relation_adj @ entity_relation_adj.T`
- **2-degree expansion**: `entity_adj_2_degree = entity_adj_1_degree @ entity_adj_1_degree`  
- **n-degree expansion**: Computed by raising the 1-degree matrix to the nth power

**Why This Works**: Matrix multiplication naturally implements graph traversal. When we multiply adjacency matrices, we're computing paths through the graph:
- 1-hop: Directly connected entities/relationships
- 2-hop: Entities connected through one intermediate entity  
- n-hop: Entities connected through (n-1) intermediate steps

**Computational Efficiency**: Using sparse matrices and vectorized operations, we can expand subgraphs containing thousands of entities in milliseconds, making this approach highly scalable.

**Dual Expansion Strategy**: We expand from both retrieved entities AND retrieved relationships, then merge the results. This ensures we capture relevant information regardless of whether the initial retrieval was more successful on the entity or relationship side.
"""
logger.info("### Expand Subgraph")

entity_relation_adj = np.zeros((len(entities), len(relations)))
for entity_id, entity in enumerate(entities):
    entity_relation_adj[entity_id, entityid_2_relationids[entity_id]] = 1

entity_relation_adj = csr_matrix(entity_relation_adj)

entity_adj_1_degree = entity_relation_adj @ entity_relation_adj.T
relation_adj_1_degree = entity_relation_adj.T @ entity_relation_adj

target_degree = 1

entity_adj_target_degree = entity_adj_1_degree
for _ in range(target_degree - 1):
    entity_adj_target_degree = entity_adj_target_degree @ entity_adj_1_degree.T
relation_adj_target_degree = relation_adj_1_degree
for _ in range(target_degree - 1):
    relation_adj_target_degree = relation_adj_target_degree @ relation_adj_1_degree.T

entity_relation_adj_target_degree = entity_adj_target_degree @ entity_relation_adj

"""
By taking the value from the target degree expansion matrix, we can easily expand the corresponding degree from the retrieved entity and relations to obtain all relations of the subgraph.
"""
logger.info("By taking the value from the target degree expansion matrix, we can easily expand the corresponding degree from the retrieved entity and relations to obtain all relations of the subgraph.")

expanded_relations_from_relation = set()
expanded_relations_from_entity = set()

filtered_hit_relation_ids = [
    relation_res["entity"]["id"]
    for relation_res in relation_search_res
]
for hit_relation_id in filtered_hit_relation_ids:
    expanded_relations_from_relation.update(
        relation_adj_target_degree[hit_relation_id].nonzero()[1].tolist()
    )

filtered_hit_entity_ids = [
    one_entity_res["entity"]["id"]
    for one_entity_search_res in entity_search_res
    for one_entity_res in one_entity_search_res
]

for filtered_hit_entity_id in filtered_hit_entity_ids:
    expanded_relations_from_entity.update(
        entity_relation_adj_target_degree[filtered_hit_entity_id].nonzero()[1].tolist()
    )

relation_candidate_ids = list(
    expanded_relations_from_relation | expanded_relations_from_entity
)

relation_candidate_texts = [
    relations[relation_id] for relation_id in relation_candidate_ids
]

"""
We have get the candidate relationships by expanding the subgraph, which will be reranked by LLM in the next step.

### LLM Reranking

In this stage, we deploy the powerful self-attention mechanism of LLM to further filter and refine the candidate set of relationships. The subgraph expansion step provides us with many potentially relevant relationships, but not all of them are equally useful for answering our specific query. This is where **Large Language Models** excel - they can understand the semantic meaning of both the query and the candidate relationships, then intelligently select the most relevant ones.

**Why LLM Reranking is Necessary**:
- **Semantic understanding**: LLMs can understand complex query intentions that pure similarity search might miss
- **Multi-hop reasoning**: LLMs can trace logical connections across multiple relationships
- **Context awareness**: LLMs consider how relationships work together to answer the query
- **Quality filtering**: LLMs can identify and prioritize the most informative relationships

**Chain-of-Thought Prompting Strategy**:
We use a structured approach that encourages the LLM to:
1. **Analyze the query**: Break down what information is needed to answer the question
2. **Identify key connections**: Determine which types of relationships would be most helpful  
3. **Reason about relevance**: Explain why specific relationships are chosen
4. **Rank by importance**: Order relationships by their utility for the final answer

**One-Shot Learning Pattern**: We provide a concrete example of the reasoning process to guide the LLM's behavior. This example demonstrates how to identify core entities, trace multi-hop connections, and prioritize the most direct relationships.

**JSON Output Format**: By requiring structured JSON output, we ensure reliable parsing and consistent results, making the system robust for production use.

#### Define One-Shot Learning Examples

First, we prepare the one-shot learning examples to guide the LLM's reasoning process:
"""
logger.info("### LLM Reranking")

query_prompt_one_shot_input = """I will provide you with a list of relationship descriptions. Your task is to select 3 relationships that may be useful to answer the given question. Please return a JSON object containing your thought process and a list of the selected relationships in order of their relevance.

Question:
When was the mother of the leader of the Third Crusade born?

Relationship descriptions:
[1] Eleanor was born in 1122.
[2] Eleanor married King Louis VII of France.
[3] Eleanor was the Duchess of Aquitaine.
[4] Eleanor participated in the Second Crusade.
[5] Eleanor had eight children.
[6] Eleanor was married to Henry II of England.
[7] Eleanor was the mother of Richard the Lionheart.
[8] Richard the Lionheart was the King of England.
[9] Henry II was the father of Richard the Lionheart.
[10] Henry II was the King of England.
[11] Richard the Lionheart led the Third Crusade.

"""
query_prompt_one_shot_output = """{"thought_process": "To answer the question about the birth of the mother of the leader of the Third Crusade, I first need to identify who led the Third Crusade and then determine who his mother was. After identifying his mother, I can look for the relationship that mentions her birth.", "useful_relationships": ["[11] Richard the Lionheart led the Third Crusade", "[7] Eleanor was the mother of Richard the Lionheart", "[1] Eleanor was born in 1122"]}"""

"""
#### Create Query Prompt Template

Next, we define the template for formatting new queries:
"""
logger.info("#### Create Query Prompt Template")

query_prompt_template = """Question:
{question}

Relationship descriptions:
{relation_des_str}

"""

"""
#### Implement the Reranking Function

Now we implement the core reranking function that processes candidate relationships:
"""
logger.info("#### Implement the Reranking Function")

def rerank_relations(
    query: str, relation_candidate_texts: list[str], relation_candidate_ids: list[str]
) -> list[int]:
    """
    Rerank candidate relations using LLM to select the most relevant ones for answering a query.

    This function uses a large language model with Chain-of-Thought prompting to analyze
    candidate relationships and select the most useful ones for answering the given query.
    It employs a one-shot learning approach with a predefined example to guide the LLM's
    reasoning process.

    Args:
        query (str): The input question that needs to be answered.
        relation_candidate_texts (list[str]): List of candidate relationship descriptions.
        relation_candidate_ids (list[str]): List of IDs corresponding to the candidate relations.

    Returns:
        list[int]: A list of relation IDs ranked by their relevance to the query.
    """
    relation_des_str = "\n".join(
        map(
            lambda item: f"[{item[0]}] {item[1]}",
            zip(relation_candidate_ids, relation_candidate_texts),
        )
    ).strip()
    rerank_prompts = ChatPromptTemplate.from_messages(
        [
            HumanMessage(query_prompt_one_shot_input),
            AIMessage(query_prompt_one_shot_output),
            HumanMessagePromptTemplate.from_template(query_prompt_template),
        ]
    )
    rerank_chain = (
        rerank_prompts
        | llm.bind(response_format={"type": "json_object"})
        | JsonOutputParser()
    )
    rerank_res = rerank_chain.invoke(
        {"question": query, "relation_des_str": relation_des_str}
    )
    rerank_relation_ids = []
    rerank_relation_lines = rerank_res["useful_relationships"]
    id_2_lines = {}
    for line in rerank_relation_lines:
        id_ = int(line[line.find("[") + 1 : line.find("]")])
        id_2_lines[id_] = line.strip()
        rerank_relation_ids.append(id_)
    return rerank_relation_ids

"""
#### Execute the Reranking Process

Finally, we apply the reranking function to our candidate relationships:
"""
logger.info("#### Execute the Reranking Process")

rerank_relation_ids = rerank_relations(
    query,
    relation_candidate_texts=relation_candidate_texts,
    relation_candidate_ids=relation_candidate_ids,
)

"""
### Get Final Results

We can get final retrieved passages from the reranked relationships. The final step demonstrates the power of our Graph RAG approach by comparing it directly with traditional RAG methods. This comparison reveals why graph-based reasoning is essential for complex multi-hop questions.

**Our Method - Graph RAG Process**:
1. Start with reranked relationships from LLM filtering
2. Map relationships back to their source passages using `relationid_2_passageids`
3. Collect unique passages while preserving relevance order
4. Return the top-k most relevant passages for answer generation

**Baseline - Naive RAG Process**:
1. Directly search the passage collection using query embeddings
2. Return top-k most semantically similar passages
3. No consideration of entity relationships or graph structure

**Key Differences**:
- **Graph RAG**: Reasons through entity relationships to find relevant context
- **Naive RAG**: Relies solely on surface-level semantic similarity between query and passages

**Expected Outcome**: For multi-hop questions like "What contribution did the son of Euler's teacher make?", our Graph RAG approach should:
- **Identify the reasoning chain**: Euler → Johann Bernoulli (teacher) → Daniel Bernoulli (son) → contributions
- **Retrieve relevant passages**: Find passages about Daniel Bernoulli's contributions to fluid dynamics
- **Provide accurate answers**: Generate responses based on the correct contextual information

In contrast, naive RAG might retrieve passages about Euler directly or miss the multi-hop connection entirely, leading to incomplete or incorrect answers.
"""
logger.info("### Get Final Results")

final_top_k = 2

final_passages = []
final_passage_ids = []
for relation_id in rerank_relation_ids:
    for passage_id in relationid_2_passageids[relation_id]:
        if passage_id not in final_passage_ids:
            final_passage_ids.append(passage_id)
            final_passages.append(passages[passage_id])
passages_from_our_method = final_passages[:final_top_k]

"""
We can compare the results with the naive RAG method, which retrieves the topK passages based on the query embedding directly from the passage collection.
"""
logger.info("We can compare the results with the naive RAG method, which retrieves the topK passages based on the query embedding directly from the passage collection.")

naive_passage_res = milvus_client.search(
    collection_name=passage_col_name,
    data=[query_embedding],
    limit=final_top_k,
    output_fields=["text"],
)[0]
passages_from_naive_rag = [res["entity"]["text"] for res in naive_passage_res]

logger.debug(
    f"Passages retrieved from naive RAG: \n{passages_from_naive_rag}\n\n"
    f"Passages retrieved from our method: \n{passages_from_our_method}\n\n"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Use the following pieces of retrieved context to answer the question. If there is not enough information in the retrieved context to answer the question, just say that you don't know.
Question: {question}
Context: {context}
Answer:""",
        )
    ]
)

rag_chain = prompt | llm | StrOutputParser()

answer_from_naive_rag = rag_chain.invoke(
    {"question": query, "context": "\n".join(passages_from_naive_rag)}
)
answer_from_our_method = rag_chain.invoke(
    {"question": query, "context": "\n".join(passages_from_our_method)}
)

logger.debug(
    f"Answer from naive RAG: {answer_from_naive_rag}\n\nAnswer from our method: {answer_from_our_method}"
)

"""
The results show that the retrieved passages from the vanilla RAG missed a ground-truth passage, which led to a wrong answer.

The retrieved passages from our method are correct, and it helps to get an accurate answer to the question.

### Key Insights and Learning Outcomes

The comparison results clearly demonstrate the superiority of our Graph RAG approach for multi-hop reasoning tasks. Let's analyze what we've accomplished:

**Performance Analysis**:
- **Naive RAG Limitation**: Traditional similarity search fails because the query "What contribution did the son of Euler's teacher make?" doesn't have high semantic similarity to passages about Daniel Bernoulli's fluid dynamics contributions. The surface-level keywords don't match well.
- **Graph RAG Success**: Our method successfully traces the logical chain: Query mentions "Euler" → Entity retrieval finds "Leonhard Euler" → Graph expansion discovers "Johann Bernoulli was Euler's teacher" → Further expansion finds "Daniel Bernoulli was Johann's son" → Relationship filtering identifies Daniel's contributions → Correct passages retrieved.

**Methodological Innovations Demonstrated**:
1. **Vector-only Graph RAG**: We achieved graph-level reasoning using only vector databases, eliminating architectural complexity
2. **Multi-modal retrieval**: Combining entity-based and relationship-based search paths provided redundancy and improved coverage  
3. **Mathematical graph expansion**: Sparse matrix operations enabled efficient multi-hop traversal at scale
4. **LLM-powered filtering**: Chain-of-thought reasoning provided intelligent relationship selection beyond simple similarity

**Practical Applications**:
This approach excels in domains requiring complex reasoning:
- **Knowledge bases**: Scientific literature, historical records, technical documentation
- **Enterprise search**: Finding information across interconnected business entities and processes
- **Question answering**: Academic research, legal document analysis, medical knowledge retrieval
- **Content recommendation**: Understanding user intent through relationship networks

**Scalability Considerations**:
- **Vector database scaling**: Milvus can handle billions of vectors with distributed architecture
- **Matrix computation efficiency**: Sparse matrix operations scale logarithmically with data size
- **LLM inference optimization**: Reranking step can be parallelized and cached for repeated patterns

The tutorial demonstrates that sophisticated reasoning capabilities can be achieved through thoughtful system design, even when using simpler infrastructure components. This balance of power and simplicity makes the approach highly practical for real-world deployments.

## Scale Your Graph RAG System with Fully-Managed Milvus

While the example in this tutorial works well with a small dataset, implementing Graph RAG in production with large-scale data requires robust infrastructure. Milvus is a distributed vector database that scales to billions, making it a trustable choice for managing large-scale vector data. Managing a self-hosted Milvus cluster can become challenging in production. If your priority is developing business logic for your RAG applications, Zilliz Cloud offers a fully-managed Milvus service that handles all the operational complexity for you:

![Zilliz Cloud Screenshot](../images/zilliz_screenshot.png)

- **Production-Ready**: Built-in high availability and security features essential for mission-critical AI applications
- **10x Faster Performance**: Its proprietary Cardinal vector index engine provides 10x faster performance even compared to high-performance open-source Milvus.
- **AutoIndex**: The AutoIndex feature saving the effort of index selection and parameter tuning.
- **Lower Total Cost of Ownership (TCO)**: Focus on your application while we handle scaling, updates, and monitoring, and pay only for what you use with flexible pricing tiers, which leads to a lower TCO compared to managing a self-hosted Milvus cluster

[**Try Zilliz Cloud for Free Today →**](https://cloud.zilliz.com/signup?utm_source=github&utm_medium=referral&utm_campaign=Nir-250512)
"""
logger.info("### Key Insights and Learning Outcomes")

logger.info("\n\n[DONE]", bright=True)