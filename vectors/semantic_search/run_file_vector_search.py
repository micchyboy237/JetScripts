"""
File Vector Search with Hybrid Reranking.

Performs semantic search over local files using llama.cpp embeddings,
followed by optional BM25 and cross-encoder reranking stages. Results
are streamed progressively and saved as JSON artifacts.

Usage Examples:
    # Basic search with defaults
    python run_file_vector_search.py "AI Agents" /path/to/docs

    # Search multiple directories with custom extensions
    python run_file_vector_search.py "error handling" ./src ./tests -e .py,.md

    # Use flag-based query and custom weights
    python run_file_vector_search.py -q "async patterns" -d ./lib -w dir:0.1,name:0.3,content:0.6

    # Verbose output with caching and result limit
    python run_file_vector_search.py "deployment config" /ops -v -c --top-k 20

    # Custom chunking parameters
    python run_file_vector_search.py "API design" ./docs --chunk-size 512 --chunk-overlap 128 -b /base/path
"""

import argparse
import shutil
from pathlib import Path
from typing import get_args

from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.rerank_utils import rerank

# from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.data.utils import generate_unique_id
from jet.file.utils import save_file
from jet.logger import logger
from jet.logger.config import colorize_log

# from jet.models.tokenizer.base import get_tokenizer_fn
from jet.utils.language import DetectLangResult, detect_lang
from jet.utils.text import format_sub_dir
from jet.vectors.reranker.bm25 import rerank_bm25
from jet.vectors.semantic_search.file_vector_search import (
    DEFAULT_WEIGHTS,
    FileSearchResult,
    Weights,
    merge_results,
    search_files,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


def print_results(
    query: str,
    results: list[FileSearchResult],
    split_chunks: bool,
    detected_lang: DetectLangResult | None = None,
):
    if detected_lang:
        print(
            f"[Detected language] lang: {detected_lang.get('lang')} | score: {detected_lang.get('score')}"
        )

    for num, result in enumerate(results[:10], start=1):
        file_path = result["metadata"]["file_path"]
        start_idx = result["metadata"]["start_idx"]
        end_idx = result["metadata"]["end_idx"]
        chunk_idx = result["metadata"]["chunk_idx"]
        num_tokens = result["metadata"]["num_tokens"]
        score = result["score"]
        print(
            f"{colorize_log(f'{num}.)', 'ORANGE')} Score: {colorize_log(f'{score:.3f}', 'SUCCESS')} | Chunk: {chunk_idx} | Tokens: {num_tokens} | Start - End: {start_idx} - {end_idx}\nFile: {file_path}"
        )


def rerank_results(
    query: str, results: list[FileSearchResult]
) -> tuple[list[str], list[dict]]:
    """Rerank search results using BM25."""
    texts = [result["text"] for result in results]
    ids = [generate_unique_id() for _ in texts]
    metadatas = [result["metadata"] for result in results]
    query_candidates, reranked_results = rerank_bm25(
        query, texts, ids=ids, metadatas=metadatas
    )
    id_to_embed_score = {
        id_: result.get("score", None) for id_, result in zip(ids, results)
    }
    id_to_embed_rank = {id_: rank for rank, id_ in enumerate(ids, start=1)}
    for reranked in reranked_results:
        reranked["embed_score"] = id_to_embed_score.get(reranked["id"], None)
        reranked["embed_rank"] = id_to_embed_rank.get(reranked["id"], None)
    return query_candidates, reranked_results


def cross_encoder_rerank(
    query: str, results: list[FileSearchResult], top_n: int = 50
) -> list[FileSearchResult]:
    """Rerank search results using the llama.cpp reranker server."""
    if not results:
        logger.debug("cross_encoder_rerank: No results to rerank")
        return results

    candidates = results[:top_n]
    texts = [result["text"] for result in candidates]

    logger.info(
        f"Starting cross-encoder reranking for {len(candidates)} candidates "
        f"(query='{query[:50]}...')"
    )

    try:
        # Use the new server-based reranker
        # normalize_scores=True is default, providing 0-1 range compatible with embed scores
        reranked_outputs = rerank(
            query=query,
            documents=texts,
            top_n=len(candidates),
            normalize_scores=True,
        )

        logger.info(f"Received {len(reranked_outputs)} reranked results from server")

        reranked_results = []
        for rr in reranked_outputs:
            # Map back to original result using index
            original_idx = rr["index"]
            if original_idx >= len(candidates):
                logger.warning(
                    f"Rerank returned invalid index {original_idx} for {len(candidates)} candidates"
                )
                continue

            original_result = candidates[original_idx]
            ce_score = float(rr["score"])

            # Hybrid scoring: 60% embedding + 40% cross-encoder
            # Both scores should be in comparable 0-1 ranges
            hybrid_score = 0.6 * original_result["score"] + 0.4 * ce_score

            reranked_result = original_result.copy()
            reranked_result["score"] = hybrid_score
            reranked_result["metadata"] = original_result["metadata"].copy()
            reranked_result["metadata"]["cross_encoder_score"] = ce_score
            reranked_result["metadata"]["cross_encoder_raw_score"] = float(
                rr.get("raw_score", 0.0)
            )
            reranked_results.append(reranked_result)

        # Sort by new hybrid score
        reranked_results.sort(key=lambda x: x["score"], reverse=True)

        # Reassign ranks
        for i, result in enumerate(reranked_results, 1):
            result["rank"] = i

        logger.info(
            f"Cross-encoder reranking complete. Top score: "
            f"{reranked_results[0]['score']:.4f}"
            if reranked_results
            else "N/A"
        )

        # Append any remaining results beyond top_n that weren't reranked
        if len(results) > top_n:
            reranked_results.extend(results[top_n:])

        return reranked_results

    except Exception as e:
        logger.error(f"Cross-encoder reranking failed: {str(e)}", exc_info=True)
        print(f"Cross-encoder reranking failed: {str(e)}")
        return results


def main(
    query: str,
    directories: list[str],
    extensions: list[str] = [".py"],
    use_cache: bool = False,
    embed_model_name: str = EMBED_MODEL,
    top_k: int | None = None,
    threshold: float = 0.0,
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    batch_size: int = 128,
    weights: "Weights" = DEFAULT_WEIGHTS,
    verbose: bool = False,
):
    """Main function to demonstrate file search with hybrid reranking, using streaming progressive results."""
    output_dir = OUTPUT_DIR / format_sub_dir(query)
    shutil.rmtree(str(output_dir), ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The following were formerly hardcoded, now are parameters:
    # embed_model_name, top_k, threshold, chunk_size, chunk_overlap, batch_size, weights

    # embed_model = SentenceTransformerRegistry.load_model(
    #     embed_model_name, truncate_dim=truncate_dim, max_seq_length=max_seq_len)
    # tokenizer = SentenceTransformerRegistry.get_tokenizer(embed_model_name)
    tokenizer = get_tokenizer(embed_model_name)

    def count_tokens(text):
        return len(tokenizer.encode(text))

    def preprocess_text(text):
        return clean_markdown_links(text)

    split_chunks = True
    print(f"Progressive results for '{query}' in these dirs (streaming):")
    for d in directories:
        print(d)

    # Progressive, streaming search
    with_split_chunks_results = []
    save_every = 250
    for result in search_files(
        directories,
        query,
        extensions,
        top_k=top_k,
        threshold=threshold,
        embed_model=embed_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_chunks=split_chunks,
        tokenizer=count_tokens,
        preprocess=preprocess_text,
        includes=[],
        excludes=["**/.venv/*", "**/.pytest_cache/*", "**/node_modules/*"],
        weights=weights,
        batch_size=batch_size,
        use_cache=use_cache,
    ):
        # Print each streaming result as received
        detected_lang = detect_lang(result["text"])
        if verbose:
            print_results(
                query, [result], split_chunks=True, detected_lang=detected_lang
            )
        if detected_lang["lang"] == "en":
            with_split_chunks_results.append(result)

            # Save after every <save_every> items
            if len(with_split_chunks_results) % save_every == 0:
                # Sort by score descending
                with_split_chunks_results.sort(key=lambda x: x["score"], reverse=True)
                # Update rank
                for i, res in enumerate(with_split_chunks_results, 1):
                    res["rank"] = i
                save_file(
                    {
                        "query": query,
                        "count": len(with_split_chunks_results),
                        "merged": not split_chunks,
                        "results": with_split_chunks_results,
                    },
                    f"{output_dir}/search_results_split.json",
                )

    # After the loop, save any remaining results that weren't saved in the last group
    if len(with_split_chunks_results) > 0:
        # Sort by score descending
        with_split_chunks_results.sort(key=lambda x: x["score"], reverse=True)
        # Update rank
        for i, res in enumerate(with_split_chunks_results, 1):
            res["rank"] = i
        save_file(
            {
                "query": query,
                "count": len(with_split_chunks_results),
                "merged": not split_chunks,
                "results": with_split_chunks_results,
            },
            f"{output_dir}/search_results_split.json",
        )

    # Merge chunks
    split_chunks = False
    merged_results = merge_results(with_split_chunks_results, tokenizer=count_tokens)
    save_file(
        {
            "query": query,
            "count": len(merged_results),
            "merged": not split_chunks,
            "results": merged_results,
        },
        f"{output_dir}/search_results_merged.json",
    )

    # BM25 reranking
    top_n = 50
    query_candidates, bm25_reranked_results = rerank_results(
        query, merged_results[:top_n]
    )
    save_file(
        {
            "query": query,
            "candidates": query_candidates,
            "count": len(bm25_reranked_results),
            "results": bm25_reranked_results,
        },
        f"{output_dir}/reranked_results_bm25_split.json",
    )

    # BM25 reranking on merged results
    query_candidates, bm25_reranked_merged = rerank_results(
        query, merged_results[:top_n]
    )
    save_file(
        {
            "query": query,
            "candidates": query_candidates,
            "count": len(bm25_reranked_merged),
            "results": bm25_reranked_merged,
        },
        f"{output_dir}/reranked_results_bm25_merged.json",
    )

    # # Cross-encoder reranking
    # cross_encoder_results = cross_encoder_rerank(
    #     query, merged_results, top_n)
    # save_file({
    #     "query": query,
    #     "count": len(cross_encoder_results),
    #     "results": cross_encoder_results
    # }, f"{output_dir}/reranked_results_cross_encoder_split.json")

    # # Cross-encoder reranking on merged results
    # cross_encoder_merged = cross_encoder_rerank(query, merged_results, top_n)
    # save_file({
    #     "query": query,
    #     "count": len(cross_encoder_merged),
    #     "results": cross_encoder_merged
    # }, f"{output_dir}/reranked_results_cross_encoder_merged.json")

    # Print final cross-encoder reranked merged results
    print_results(query, merged_results, split_chunks)


def validate_directories(
    directories: list[str], base_dir: str | Path | None = None
) -> list[str]:
    """
    Validate that provided directories exist and are accessible.

    If a directory is relative and base_dir is provided,
    resolve it against base_dir before validation.
    """
    valid_dirs: list[str] = []

    base_path = Path(base_dir).resolve() if base_dir else None

    for directory in directories:
        directory = directory.strip()
        if not directory:
            continue

        path = Path(directory)

        # Resolve relative paths against base_dir if provided
        if not path.is_absolute() and base_path:
            path = base_path / path

        path = path.resolve()

        if not path.is_dir():
            print(
                f"Warning: '{directory}' resolved to '{path}' "
                f"is not a valid or accessible directory. Skipping."
            )
            continue

        valid_dirs.append(str(path))

    if not valid_dirs:
        raise ValueError("No valid directories provided.")

    return valid_dirs


ALLOWED_EMBED_MODELS = get_args(LLAMACPP_EMBED_KEYS)


def embed_model_type(value: str) -> str:
    if value not in ALLOWED_EMBED_MODELS:
        raise argparse.ArgumentTypeError(
            f"Invalid embedding model {value!r}. "
            f"Must be one of: {', '.join(ALLOWED_EMBED_MODELS)}"
        )
    return value


def parse_arguments():
    """Parse command line arguments for file vector search."""
    parser = argparse.ArgumentParser(
        description="Search files using semantic vector embeddings + hybrid reranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional (required or with smart defaults)
    parser.add_argument(
        "query", type=str, nargs="?", default="AI Agents", help="Search query"
    )
    parser.add_argument(
        "directories",
        type=str,
        nargs="+",
        default=[
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts"
        ],
        help="Search directories (space-separated)",
    )

    # Optional flags with short versions
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        dest="query_flag",
        default=None,
        help="Alternative way to specify query",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        type=str,
        default=None,
        help="Base directory used to resolve relative search directories",
    )

    parser.add_argument(
        "-d",
        "--directories",
        type=str,
        nargs="+",
        dest="directories_flag",
        default=None,
        help="Alternative way to specify directories (space-separated)",
    )

    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        default=".py",
        help="File extensions to include, comma separated (e.g. .py,.md,.txt)",
    )

    parser.add_argument(
        "-m",
        "--embed-model",
        type=embed_model_type,
        default=EMBED_MODEL,
        help=f"Embedding model to use ({', '.join(ALLOWED_EMBED_MODELS)})",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of results to return (None = all)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum similarity score threshold",
    )

    parser.add_argument(
        "--chunk-size", type=int, default=400, help="Size of text chunks"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap between consecutive chunks",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding generation",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="dir:0.0,name:0.25,content:0.75",
        help="Similarity weights (format: dir:X,name:Y,content:Z)",
    )

    parser.add_argument(
        "-c",
        "--cache",
        action="store_true",
        default=False,
        help="Use cache for embeddings and model loading",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print verbose search progress and result details",
    )

    args = parser.parse_args()

    # Resolve query (flag takes precedence over positional)
    query = args.query_flag if args.query_flag is not None else args.query

    # Resolve directories (flag takes precedence)
    directories = (
        args.directories_flag if args.directories_flag is not None else args.directories
    )
    directories = validate_directories(
        directories,
        base_dir=args.base_dir,
    )

    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]

    # Parse weights
    weights_dict = DEFAULT_WEIGHTS.copy()
    if args.weights:
        try:
            for part in args.weights.split(","):
                key, value = part.split(":")
                weights_dict[key.strip()] = float(value.strip())
        except Exception as e:
            print(
                f"Warning: Could not parse weights '{args.weights}'. Using defaults. Error: {e}"
            )

    return argparse.Namespace(
        query=query,
        directories=directories,
        extensions=extensions,
        embed_model=args.embed_model,
        top_k=args.top_k,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        weights=weights_dict,
        use_cache=args.cache,
        verbose=args.verbose,
        base_dir=args.base_dir,
    )


if __name__ == "__main__":
    args = parse_arguments()

    # Update main call with new parameters
    main(
        query=args.query,
        directories=args.directories,
        extensions=args.extensions,
        use_cache=args.use_cache,
        # Pass additional parameters (you'll need to update main() signature too)
        embed_model_name=args.embed_model,
        top_k=args.top_k,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        weights=args.weights,
        verbose=args.verbose,
    )
