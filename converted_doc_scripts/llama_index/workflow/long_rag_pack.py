import asyncio
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import EmbedModelType
from jet.transformers.formatters import format_json
from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Context,
)
from typing import Iterable
from llama_index.core.retrievers import BaseRetriever
from IPython.display import display, Markdown
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import SimpleDirectoryReader
from llama_index.core.workflow import Event
from llama_index.core.llms import LLM
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.vector_stores.simple import BasePydanticVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict, Optional, Set, FrozenSet
import os
import nest_asyncio
from jet.logger import logger

nest_asyncio.apply()

# %pip install -U llama-index llama-index-embeddings-huggingface

# Helper Functions
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_MAX_GROUP_SIZE = 20
DEFAULT_SMALL_CHUNK_SIZE = 512
DEFAULT_TOP_K = 8


def split_doc(chunk_size: int, documents: List[BaseNode]) -> List[TextNode]:
    """Splits documents into smaller pieces."""
    text_parser = SentenceSplitter(chunk_size=chunk_size)
    return text_parser.get_nodes_from_documents(documents)


def group_docs(
    nodes: List[str],
    adj: Dict[str, List[str]],
    max_group_size: Optional[int] = DEFAULT_MAX_GROUP_SIZE,
) -> Set[FrozenSet[str]]:
    """Groups documents."""
    docs = sorted(nodes, key=lambda node: len(adj[node]))
    groups = set()
    for d in docs:
        related_groups = set()
        for r in adj[d]:
            for g in groups:
                if r in g:
                    related_groups = related_groups.union(frozenset([g]))
        gnew = {d}
        related_groupsl = sorted(related_groups, key=lambda el: len(el))
        for g in related_groupsl:
            if max_group_size is None or len(gnew) + len(g) <= max_group_size:
                gnew = gnew.union(g)
                if g in groups:
                    groups.remove(g)
        groups.add(frozenset(gnew))
    return groups


def get_grouped_docs(
    nodes: List[TextNode],
    max_group_size: Optional[int] = DEFAULT_MAX_GROUP_SIZE,
) -> List[TextNode]:
    """Gets list of documents that are grouped."""
    nodes_str = [node.id_ for node in nodes]
    adj: Dict[str, List[str]] = {
        node.id_: [val.node_id for val in node.relationships.values()]
        for node in nodes
    }
    nodes_dict = {node.id_: node for node in nodes}
    res = group_docs(nodes_str, adj, max_group_size)
    ret_nodes = []
    for g in res:
        cur_node = TextNode()
        for node_id in g:
            cur_node.text += nodes_dict[node_id].text + "\n\n"
            cur_node.metadata.update(nodes_dict[node_id].metadata)
        ret_nodes.append(cur_node)
    return ret_nodes

# Custom Retriever


class LongRAGRetriever(BaseRetriever):
    """Long RAG Retriever."""

    def __init__(
        self,
        grouped_nodes: List[TextNode],
        small_toks: List[TextNode],
        vector_store: BasePydanticVectorStore,
        similarity_top_k: int = DEFAULT_TOP_K,
        embed_model: Optional[HuggingFaceEmbedding] = None,
    ) -> None:
        """Constructor."""
        self._grouped_nodes = grouped_nodes
        self._grouped_nodes_dict = {node.id_: node for node in grouped_nodes}
        self._small_toks = small_toks
        self._small_toks_dict = {node.id_: node for node in self._small_toks}
        self._similarity_top_k = similarity_top_k
        self._vec_store = vector_store
        self._embed_model = embed_model or Settings.embed_model

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieves."""
        query_embedding = generate_embeddings(
            query_bundle.query_str,
            model=self._embed_model,
            return_format="numpy"
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=500
        )
        query_res = self._vec_store.query(vector_store_query)
        top_parents_set: Set[str] = set()
        top_parents: List[NodeWithScore] = []
        for id_, similarity in zip(query_res.ids, query_res.similarities):
            cur_node = self._small_toks_dict[id_]
            parent_id = cur_node.ref_doc_id
            if parent_id not in top_parents_set:
                top_parents_set.add(parent_id)
                parent_node = self._grouped_nodes_dict[parent_id]
                node_with_score = NodeWithScore(
                    node=parent_node, score=similarity
                )
                top_parents.append(node_with_score)
                if len(top_parents_set) >= self._similarity_top_k:
                    break
        assert len(top_parents) == min(
            self._similarity_top_k, len(self._grouped_nodes)
        )
        return top_parents

# Workflow Definition


class LoadNodeEvent(Event):
    """Event for loading nodes."""
    small_nodes: Iterable[TextNode]
    grouped_nodes: list[TextNode]
    index: VectorStoreIndex
    similarity_top_k: int
    llm: LLM


class LongRAGWorkflow(Workflow):
    """Long RAG Workflow."""
    @step
    async def ingest(self, ev: StartEvent) -> LoadNodeEvent | None:
        """Ingestion step."""
        data_dir: str = ev.get("data_dir")
        llm: LLM = ev.get("llm")
        chunk_size: int | None = ev.get("chunk_size")
        similarity_top_k: int = ev.get("similarity_top_k")
        small_chunk_size: int = ev.get("small_chunk_size")
        index: VectorStoreIndex | None = ev.get("index")
        index_kwargs: dict | None = ev.get("index_kwargs", {})

        if any(
            i is None
            for i in [data_dir, llm, similarity_top_k, small_chunk_size]
        ):
            return None

        # Set embedding model in Settings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="jinaai/jina-embeddings-v2-base-en")

        if not index:
            docs = SimpleDirectoryReader(data_dir).load_data()
            if chunk_size is not None:
                nodes = split_doc(chunk_size, docs)
                grouped_nodes = get_grouped_docs(nodes)
            else:
                grouped_nodes = docs
            small_nodes = split_doc(small_chunk_size, grouped_nodes)
            index_kwargs = index_kwargs or {}
            index_kwargs["embed_model"] = Settings.embed_model
            index = VectorStoreIndex(small_nodes, **index_kwargs)
        else:
            small_nodes = index.docstore.docs.values()
            grouped_nodes = get_grouped_docs(small_nodes, None)

        return LoadNodeEvent(
            small_nodes=small_nodes,
            grouped_nodes=grouped_nodes,
            index=index,
            similarity_top_k=similarity_top_k,
            llm=llm,
        )

    @step
    async def make_query_engine(
        self, ctx: Context, ev: LoadNodeEvent
    ) -> StopEvent:
        """Query engine construction step."""
        retriever = LongRAGRetriever(
            grouped_nodes=ev.grouped_nodes,
            small_toks=ev.small_nodes,
            similarity_top_k=ev.similarity_top_k,
            vector_store=ev.index.vector_store,
            embed_model=Settings.embed_model,
        )
        query_eng = RetrieverQueryEngine.from_args(retriever, ev.llm)
        return StopEvent(
            result={
                "retriever": retriever,
                "query_engine": query_eng,
                "index": ev.index,
            }
        )

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Query step."""
        query_str: str | None = ev.get("query_str")
        query_eng = ev.get("query_eng")
        if query_str is None:
            return None
        result = query_eng.query(query_str)
        return StopEvent(result=result)

# Run Workflow


async def run_workflow():
    wf = LongRAGWorkflow(timeout=60)
    llm = MLXModelRegistry.load_model(model="qwen3-1.7b-4bit")
    data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    query = "Tell me about yourself and your greatest achievements."
    result = await wf.run(
        data_dir=data_dir,
        llm=llm,
        chunk_size=DEFAULT_CHUNK_SIZE,
        similarity_top_k=DEFAULT_TOP_K,
        small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
    )
    res = await wf.run(
        query_str=query,
        query_eng=result["query_engine"],
    )
    logger.newline()
    logger.info("Query:")
    logger.debug(query)
    logger.success(str(res))

asyncio.run(run_workflow())
logger.info("\n\n[DONE]", bright=True)
