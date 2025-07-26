import os
import shutil
from typing import List
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.vectors.semantic_search.header_vector_search import HeaderSearchResult, search_headers


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def save_results(query: str, results: List[HeaderSearchResult], split_chunks: bool, output_dir: str):
    print(f"Search results for '{query}' in these dirs:")
    for num, result in enumerate(results[:10], start=1):
        header = result["metadata"]["header"]
        parent_header = result["metadata"]["parent_header"]
        start_idx = result["metadata"]["start_idx"]
        end_idx = result["metadata"]["end_idx"]
        chunk_idx = result["metadata"]["chunk_idx"]
        num_tokens = result["metadata"]["num_tokens"]
        score = result["score"]
        print(
            f"{colorize_log(f"{num}.)", "ORANGE")} Score: {colorize_log(f'{score:.3f}', 'SUCCESS')} | Chunk: {chunk_idx} | Tokens: {num_tokens} | Start - End: {start_idx} - {end_idx}\nParent: {parent_header} | Header: {header}")

    save_file({
        "query": query,
        "count": len(results),
        "merged": not split_chunks,
        "results": results
    }, f"{OUTPUT_DIR}/results_{'split' if split_chunks else 'merged'}.json")


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace('.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


def main(query):
    """Main function to demonstrate file search."""

    output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
    shutil.rmtree(output_dir, ignore_errors=True)

    # Example usage
    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/data/complete_jet_resume.md"
    header_docs = derive_by_header_hierarchy(md_file)

    save_file(header_docs, f"{output_dir}/header_docs.json")

    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit"

    top_k = len(header_docs)
    threshold = 0.0  # Using default threshold
    chunk_size = 500
    chunk_overlap = 100
    tokenizer = get_tokenizer_fn(embed_model)

    def count_tokens(text):
        return len(tokenizer(text))

    split_chunks = True
    with_split_chunks_results = list(
        search_headers(
            header_docs,
            query,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_chunks=split_chunks,
            tokenizer=count_tokens
        )
    )
    save_results(query,  with_split_chunks_results, split_chunks, output_dir)

    split_chunks = False
    without_split_chunks_results = list(
        search_headers(
            header_docs,
            query,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_chunks=split_chunks,
            tokenizer=count_tokens
        )
    )
    save_results(query, without_split_chunks_results, split_chunks, output_dir)


if __name__ == "__main__":
    query = "Tell me about yourself."
    main(query)
    query = "Tell me about your recent achievements."
    main(query)
