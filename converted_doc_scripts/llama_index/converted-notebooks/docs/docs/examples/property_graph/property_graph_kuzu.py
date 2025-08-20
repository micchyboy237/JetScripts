from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import PropertyGraphIndex
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.property_graph import LLMSynonymRetriever
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore
from pathlib import Path
from typing import List
from typing import Literal
import kuzu
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)




"""
[Kùzu](https://kuzudb.com/) is an open source, embedded graph database that's designed for query speed and scalability. It implements the Cypher query language, and utilizes a structured property graph model (a variant of the labelled property graph model) with support for ACID transactions. Because Kùzu is embedded, there's no requirement for a server to set up and use the database.

Let's begin by creating a graph from unstructured text to demonstrate how to use Kùzu as a graph and vector store to answer questions.
"""
logger.info("Let's begin by creating a graph from unstructured text to demonstrate how to use Kùzu as a graph and vector store to answer questions.")

# import nest_asyncio

# nest_asyncio.apply()

"""
## Environment Setup
"""
logger.info("## Environment Setup")


# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

"""
We will be using MLX models for this example, so we'll specify the MLX API key.
"""
logger.info("We will be using MLX models for this example, so we'll specify the MLX API key.")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Graph Construction

We first need to create an empty Kùzu database directory by calling the `kuzu.Database` constructor. This step instantiates the database and creates the necessary directories and files within a local directory that stores the graph. This `Database` object is then passed to the `KuzuPropertyGraph` constructor.
"""
logger.info("## Graph Construction")


DB_NAME = "ex.kuzu"
Path(DB_NAME).unlink(missing_ok=True)
db = kuzu.Database(DB_NAME)

"""
Because Kùzu implements the structured graph property model, it imposes some level of structure on the schema of the graph. In the above case, because we did not specify a relationship schema that we want in our graph, it uses a generic schema, where the relationship types are not constrained, allowing the extracted triples from the LLM to be stored as relationships in the graph.

### Define LLMs

Below, we'll define the models used for embedding the text and the LLMs that are used to extract triples from the text and generate the response.
In this case, we specify different temperature settings for the same model - the extraction model has a temperature of 0.
"""
logger.info("### Define LLMs")


embed_model = MLXEmbedding(model_name="mxbai-embed-large")
extract_llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.0)
generate_llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.3)

"""
## Create property graph index with structure

The recommended way to use Kùzu is to apply a structured schema to the graph. The schema is defined by specifying the relationship types (including direction) that we want in the graph. The imposition of structure helps with generating triples that are more meaningful for the types of questions we may want to answer from the graph.

By specifying the below validation schema, we can enforce that the graph only contains relationships of the specified types.
"""
logger.info("## Create property graph index with structure")


entities = Literal["PERSON", "PLACE", "ORGANIZATION"]
relations = Literal["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"]
validation_schema = [
    ("ORGANIZATION", "HAS", "PERSON"),
    ("PERSON", "WORKED_AT", "ORGANIZATION"),
    ("PERSON", "WORKED_WITH", "PERSON"),
    ("PERSON", "WORKED_ON", "ORGANIZATION"),
    ("PERSON", "PART_OF", "ORGANIZATION"),
    ("ORGANIZATION", "PART_OF", "ORGANIZATION"),
    ("PERSON", "WORKED_AT", "PLACE"),
]

"""
## Create property graph store with a vector index

To create a `KuzuPropertyGraphStore` with a vector index, we need to specify the `use_vector_index` parameter as `True`. This will create a vector index on the property graph, allowing us to perform vector-based queries on the graph.
"""
logger.info("## Create property graph store with a vector index")


graph_store = KuzuPropertyGraphStore(
    db,
    has_structured_schema=True,
    relationship_schema=validation_schema,
    use_vector_index=True,  # Enable vector index for similarity search
    embed_model=embed_model,  # Auto-detects embedding dimension from model
)

"""
To construct a property graph with the desired schema, we'll use `SchemaLLMPathExtractor` with the following parameters.
"""
logger.info("To construct a property graph with the desired schema, we'll use `SchemaLLMPathExtractor` with the following parameters.")

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=embed_model,
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=extract_llm,
            possible_entities=entities,
            possible_relations=relations,
            kg_validation_schema=validation_schema,
            strict=True,  # if false, will allow triples outside of the schema
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

"""
We can now apply the query engine on the index as before.
"""
logger.info("We can now apply the query engine on the index as before.")

Settings.llm = generate_llm

query_text = "Tell me more about Interleaf and Viaweb?"
query_engine = index.as_query_engine(include_text=False)

response = query_engine.query(query_text)
logger.debug(str(response))

retriever = index.as_retriever(include_text=False)
nodes = retriever.retrieve(query_text)
nodes[0].text

"""
## Query the vector index

As an embedded graph database, Kuzu provides a fast and performance graph-based HNSW vector index (see the [docs](https://docs.kuzudb.com/extensions/vector/)).
This allows you to also use Kuzu for similarity (vector-based) retrieval on chunk nodes.
The vector index is created after the embeddings are ingested into the chunk nodes, so you should be able to query them directly.
"""
logger.info("## Query the vector index")


query_text = "How much funding did Idelle Weber provide to Viaweb?"
query_embedding = embed_model.get_text_embedding(query_text)
vector_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=5
)

nodes, similarities = graph_store.vector_query(vector_query)

for i, (node, similarity) in enumerate(zip(nodes, similarities)):
    logger.debug(f"  {i + 1}. Similarity: {similarity:.3f}")
    logger.debug(f"     Text: {node.text}...")
    logger.debug()



class GraphVectorRetriever(BaseRetriever):
    """
    A retriever that performs vector search on a property graph store.
    """

    def __init__(self, graph_store, embed_model, similarity_top_k: int = 5):
        self.graph_store = graph_store
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self.embed_model.get_text_embedding(
            query_bundle.query_str
        )

        vector_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k,
        )
        nodes, similarities = self.graph_store.vector_query(vector_query)

        nodes_with_scores = []
        for node, similarity in zip(nodes, similarities):
            if hasattr(node, "text"):
                text_node = TextNode(
                    text=node.text,
                    id_=node.id,
                    metadata=getattr(node, "properties", {}),
                )
                nodes_with_scores.append(
                    NodeWithScore(node=text_node, score=similarity)
                )

        return nodes_with_scores


class CombinedGraphRetriever(BaseRetriever):
    """
    A retriever that performs that combines graph and vector search on a property graph store.
    """

    def __init__(
        self, graph_store, embed_model, llm, similarity_top_k: int = 5
    ):
        self.graph_store = graph_store
        self.embed_model = embed_model
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self.embed_model.get_text_embedding(
            query_bundle.query_str
        )
        vector_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k,
        )
        vector_nodes, similarities = self.graph_store.vector_query(
            vector_query
        )

        vector_results = []
        for node, similarity in zip(vector_nodes, similarities):
            if hasattr(node, "text"):
                text_node = TextNode(
                    text=node.text,
                    id_=node.id,
                    metadata=getattr(node, "properties", {}),
                )
                vector_results.append(
                    NodeWithScore(node=text_node, score=similarity)
                )

        graph_retriever = LLMSynonymRetriever(
            self.graph_store, llm=self.llm, include_text=True
        )
        graph_results = graph_retriever.retrieve(query_bundle)

        all_results = vector_results + graph_results
        seen_nodes = set()
        combined_results = []

        for node_with_score in all_results:
            node_id = node_with_score.node.node_id
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                combined_results.append(node_with_score)

        return combined_results


combined_retriever = CombinedGraphRetriever(
    graph_store=graph_store,
    llm=generate_llm,
    embed_model=embed_model,
    similarity_top_k=5,
)

query_text = "What was the role of Idelle Weber in Viaweb?"
query_bundle = QueryBundle(query_str=query_text)
results = combined_retriever.retrieve(query_bundle)
for i, node_with_score in enumerate(results):
    logger.debug(f"{i + 1}. Score: {node_with_score.score:.3f}")
    logger.debug(
        f"   Text: {node_with_score.node.text[:100]}..."
    )  # Print first 100 chars
    logger.debug(f"   Node ID: {node_with_score.node.node_id}")
    logger.debug()


query_engine = RetrieverQueryEngine.from_args(
    retriever=combined_retriever,
    llm=generate_llm,
)

response_synthesizer = get_response_synthesizer(
    llm=generate_llm, use_async=False
)

query_engine = RetrieverQueryEngine(
    retriever=combined_retriever, response_synthesizer=response_synthesizer
)

query_text = "What was the role of Idelle Weber in Viaweb?"
response = query_engine.query(query_text)
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)