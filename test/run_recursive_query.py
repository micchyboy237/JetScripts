import asyncio
import os
import pandas as pd
from elastic_transport._node._base import BaseNode
from jet.llm.ollama.constants import OLLAMA_BASE_EMBED_URL, OLLAMA_LARGE_EMBED_MODEL
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    get_retrieval_results_df,
)
import copy
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
import nest_asyncio
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
import json
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from jet.llm.ollama.base import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.llm.ollama.base import OllamaEmbedding
from jet.cache.joblib import load_from_cache_or_compute
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings, large_embed_model
llm_settings = initialize_ollama_settings()


def main(
    base_nodes: list[BaseNode],
    top_k=10,
):
    sub_chunk_sizes = [128, 256, 512]
    sub_node_parsers = [
        SentenceSplitter(chunk_size=c, chunk_overlap=20) for c in sub_chunk_sizes
    ]

    all_nodes = []
    for base_node in base_nodes:
        for n in sub_node_parsers:
            sub_nodes = n.get_nodes_from_documents([base_node])
            sub_inodes = [
                IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)

        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)

    all_nodes_dict = {n.node_id: n for n in all_nodes}

    vector_index_chunk = VectorStoreIndex(
        all_nodes, embed_model=llm_settings.embed_model)

    def query_nodes_func(
        query: str,
        threshold: float = 0.0,
        top_k: int = 10,
    ):
        vector_retriever_chunk = vector_index_chunk.as_retriever(
            similarity_top_k=len(all_nodes))

        retriever_chunk = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever_chunk},
            node_dict=all_nodes_dict,
            verbose=False,
        )

        retrieved_nodes = retriever_chunk.retrieve(query)
        retrieved_nodes_sorted = sorted(
            retrieved_nodes, key=lambda item: item.score, reverse=True)
        display_jet_source_nodes(query, retrieved_nodes_sorted)

        query_engine_chunk = RetrieverQueryEngine.from_args(
            retriever_chunk, llm=llm_settings.llm)

        response = query_engine_chunk.query(
            query
        )
        logger.log("Query:", query, colors=["WHITE", "DEBUG"])
        logger.success(str(response))

        return response

    return query_nodes_func


if __name__ == "__main__":
    chunk_size = 512
    chunk_overlap = 50
    data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    query = "Tell me about yourself and your latest achievements."

    top_k = 10
    score_threshold = 0.0

    documents = SimpleDirectoryReader(data_dir).load_data()
    node_parser = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    base_nodes = node_parser.get_nodes_from_documents(documents)

    query_nodes = main(base_nodes)

    result = query_nodes(query, threshold=score_threshold, top_k=top_k)
