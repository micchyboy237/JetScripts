import os
from typing import List, Tuple
from jet.file.utils import load_file, save_file
from jet.llm.llm_generator import LLMGenerator
from jet.logger import logger
from jet.logger.config import colorize_log
from jet.scrapers.utils import extract_texts_by_hierarchy
from jet.vectors.clusters.retrieval import RetrievalConfigDict, VectorRetriever

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def main():
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_4/top_isekai_anime_2025/pages/www_ranker_com_list_best_isekai_anime_2025_anna_lindwasser/page.html"
    html = load_file(html_file)

    config: RetrievalConfigDict = {
        "min_cluster_size": 2,
        "k_clusters": 2,
        "top_k":  None,
        "cluster_threshold": 20,
        "model_name": 'mxbai-embed-large',
        "cache_file":  None,
        "threshold": None,
    }

    header_docs = extract_texts_by_hierarchy(html, ignore_links=True)

    texts = [f"{doc.header}\n{doc.content}" for doc in header_docs]
    query = "Top isekai anime 2025."

    save_file(texts, f"{OUTPUT_DIR}/texts.json")

    retriever = VectorRetriever(config)
    retriever.load_or_compute_embeddings(texts)
    retriever.cluster_embeddings()
    retriever.build_index()
    top_chunks = retriever.retrieve_chunks(query)

    save_file(top_chunks, f"{OUTPUT_DIR}/top_chunks.json")

    logger.info("\nTop relevant chunks for RAG:")
    for i, (chunk, score) in enumerate(top_chunks, 1):
        print(f"{colorize_log(f"{i}.", "ORANGE")} | {
              colorize_log(f"{score:.4f}", "SUCCESS")} | {chunk[:50]}")

    generator = LLMGenerator()
    response = generator.generate_response(
        query, top_chunks, generation_config={"verbose": True})

    save_file(f"# LLM Generation\n\n## Prompt\n\n{response["prompt"]}\n\n## Response\n\n{response["response"]}",
              f"{OUTPUT_DIR}/llm_generation.md")


if __name__ == "__main__":
    main()
