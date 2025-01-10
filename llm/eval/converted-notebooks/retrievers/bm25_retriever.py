from pathlib import Path
import shutil
import chromadb
import nest_asyncio
import Stemmer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import SimpleDirectoryReader
from jet.logger import logger
from jet.code import MarkdownCodeExtractor
from jet.llm.ollama import initialize_ollama_settings

initialize_ollama_settings()


def main_example_1(query, input_dir, base_dir, similarity_top_k, chunk_size, chunk_overlap):
    # Setup BM25 Retriever with Disk Persistence
    documents = SimpleDirectoryReader(
        input_dir=input_dir, required_exts=".md").load_data()
    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    output_dir = Path(base_dir / "example1")
    shutil.rmtree(output_dir, ignore_errors=True)
    Path.mkdir(output_dir)

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=len(nodes),
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    bm25_retriever.persist(output_dir / "storage")

    loaded_bm25_retriever = BM25Retriever.from_persist_dir(
        output_dir / "storage")

    # Retrieve and log results
    nodes_with_scores = loaded_bm25_retriever.retrieve(query)
    logger.log("Query:", query, colors=["WHITE", "INFO"])
    logger.log("Node Scores (BM25Retriever):",
               f"({len(nodes_with_scores)})", colors=["WHITE", "SUCCESS"])
    for node in nodes_with_scores:
        logger.newline()
        logger.log("Score:", f"{node.score:.2f}", colors=["WHITE", "SUCCESS"])
        logger.log("File:", node.metadata['file_path'], colors=[
                   "WHITE", "DEBUG"])

    query_engine = RetrieverQueryEngine(loaded_bm25_retriever)
    response = query_engine.query(query)

    # Save results to files
    save_results(output_dir, query, response)


def main_example_2(query, input_dir, base_dir, similarity_top_k, chunk_size, chunk_overlap):
    # Hybrid Retriever: BM25 + Chroma
    documents = SimpleDirectoryReader(
        input_dir=input_dir, required_exts=".md").load_data()
    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    output_dir = Path(base_dir / "example2")
    shutil.rmtree(output_dir, ignore_errors=True)
    Path.mkdir(output_dir)

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    db = chromadb.PersistentClient(path=str(output_dir / "chroma_db"))
    chroma_collection = db.get_or_create_collection("dense_vectors")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

    nest_asyncio.apply()

    similarity_top_k = len(nodes)
    retriever = QueryFusionRetriever(
        [
            index.as_retriever(similarity_top_k=similarity_top_k),
            BM25Retriever.from_defaults(
                docstore=index.docstore, similarity_top_k=similarity_top_k),
        ],
        similarity_top_k=similarity_top_k,
        num_queries=1,
        use_async=True,
    )

    nodes = retriever.retrieve(query)
    logger.log("Query:", query, colors=["WHITE", "INFO"])
    logger.log("Node Scores (QueryFusionRetriever):",
               f"({len(nodes)})", colors=["WHITE", "SUCCESS"])
    for node in nodes:
        logger.newline()
        logger.log("Score:", f"{node.score:.2f}", colors=["WHITE", "SUCCESS"])
        logger.log("File:", node.metadata['file_path'], colors=[
                   "WHITE", "DEBUG"])

    query_engine = RetrieverQueryEngine(retriever)
    response = query_engine.query(query)

    # Save results to files
    save_results(output_dir, query, response)


def save_results(output_dir, query, response):
    results_dir = output_dir / "results"
    results_dir.parent.mkdir(parents=True, exist_ok=True)

    prompt_path = results_dir / "prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    response_path = results_dir / "response.md"
    response_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_path.write_text(query)
    response_path.write_text(response.response)

    extractor = MarkdownCodeExtractor()
    code_blocks = extractor.extract_code_blocks(response.response)
    for idx, code_block in enumerate(code_blocks):
        if code_block['language'] == 'python':
            generated_code_path = results_dir / f"code_block_{idx + 1}.py"
            generated_code_path.parent.mkdir(parents=True, exist_ok=True)
            generated_code_path.write_text(code_block['code'])
            logger.log("Saved to", generated_code_path.resolve(),
                       colors=["WHITE", "BRIGHT_SUCCESS"])

    logger.info("Done")


if __name__ == "__main__":
    query = "Tell me about yourself."
    # input_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    input_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/summaries/jet-resume"
    base_dir = Path("./generated/bm25_retriever")
    similarity_top_k = None
    chunk_size = 512
    chunk_overlap = 50

    # main_example_1(query, input_dir, base_dir,
    #                similarity_top_k, chunk_size, chunk_overlap)
    main_example_2(query, input_dir, base_dir,
                   similarity_top_k, chunk_size, chunk_overlap)
