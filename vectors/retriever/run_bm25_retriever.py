import Stemmer
from jet.llm.helpers.semantic_search.utils import generate_embeddings
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.vectors.reranker.utils import create_bm25_retriever
from llama_index.core import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.schema import QueryBundle
from llama_index.retrievers.bm25.base import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter


def test_class():
    names_of_base_classes = [b.__name__ for b in BM25Retriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes


def test_scores():
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    all_documents = SimpleDirectoryReader(data_path).load_data()
    doc_text = "\n\n".join([d.get_content() for d in all_documents])
    documents = [Document(text=doc_text)]

    query = "Tell me about yourself."

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
    # result_nodes = retriever.retrieve("llamaindex llm")
    result_nodes = retriever.retrieve(query)
    assert len(result_nodes) == 2
    for node in result_nodes:
        assert node.score is not None
        # assert node.score > 0.0
    display_jet_source_nodes(query, result_nodes)


def test_query_bundle():
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    all_documents = SimpleDirectoryReader(data_path).load_data()
    doc_text = "\n\n".join([d.get_content() for d in all_documents])
    documents = [Document(text=doc_text)]

    query = "Tell me about yourself."

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    top_k = 10
    retriever = create_bm25_retriever(
        nodes, similarity_top_k=top_k
    )

    # embeddings = generate_embeddings(
    #     engine="ollama", model="llama3.1", text=query)
    query_bundle = QueryBundle(
        query_str=query,
        # embedding=embeddings,
    )
    result_nodes = retriever.retrieve(query_bundle)
    assert len(result_nodes) == 2
    for node in result_nodes:
        assert node.score is not None
        # assert node.score > 0.0
    display_jet_source_nodes(query, result_nodes)


if __name__ == '__main__':
    logger.newline()
    logger.info("test_class()...")
    test_class()

    logger.newline()
    logger.info("test_scores()...")
    test_scores()

    logger.newline()
    logger.info("test_query_bundle()...")
    test_query_bundle()
