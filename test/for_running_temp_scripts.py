from deeplake.core.vectorstore import VectorStore
from jet.llm.ollama.embeddings import get_ollama_embedding_function
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.transformers.formatters import format_json
from llama_index.core.schema import NodeWithScore, TextNode

vector_store_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/semantic_search/generated/deeplake/store_1"

embedding_function = get_ollama_embedding_function("mxbai-embed-large")

vector_store = VectorStore(
    path=vector_store_path,
    read_only=True
)

query = "List your primary and secondary skills"
top_k = 20


results = vector_store.search(
    embedding_data=query, k=top_k, embedding_function=embedding_function)

nodes_with_scores = [
    NodeWithScore(
        node=TextNode(text=str(text), metadata=metadata),
        score=float(score)
    )
    for text, metadata, score in zip(results["text"], results["metadata"], results["score"])
]


display_jet_source_nodes(query, nodes_with_scores)
