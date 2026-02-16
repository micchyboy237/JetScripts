import argparse
import os
import shutil
from pathlib import Path
from typing import get_args

from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.data.utils import generate_unique_id
from jet.file.utils import save_file

# from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.llm.utils.tokenizer import get_tokenizer
from jet.logger.config import colorize_log
from jet.models.model_registry.transformers.cross_encoder_model_registry import (
    CrossEncoderRegistry,
)

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
    """Rerank search results using CrossEncoder."""
    if not results:
        return results

    # Prepare input pairs for cross-encoder
    input_pairs = [(query, result["text"]) for result in results[:top_n]]
    try:
        # Load cross-encoder model
        cross_encoder = CrossEncoderRegistry.load_model(
            model_id="cross-encoder/ms-marco-MiniLM-L6-v2",
            device="mps" if os.uname().sysname == "Darwin" else "cpu",
        )
        # Get cross-encoder scores
        scores = CrossEncoderRegistry.predict_scores(
            input_pairs, batch_size=32, show_progress=True, return_format="list"
        )

        # Combine cross-encoder scores with original scores
        reranked_results = []
        for idx, (result, ce_score) in enumerate(zip(results[:top_n], scores)):
            # Normalize cross-encoder score (sigmoid output) and combine with original
            normalized_ce_score = float(ce_score)
            hybrid_score = 0.6 * result["score"] + 0.4 * normalized_ce_score
            reranked_result = result.copy()
            reranked_result["score"] = hybrid_score
            reranked_result["metadata"] = result["metadata"].copy()
            reranked_result["metadata"]["cross_encoder_score"] = normalized_ce_score
            reranked_results.append(reranked_result)

        # Sort by hybrid score
        reranked_results.sort(key=lambda x: x["score"], reverse=True)

        # Update ranks
        for i, result in enumerate(reranked_results, 1):
            result["rank"] = i

        # Include any remaining results beyond top_n
        if len(results) > top_n:
            reranked_results.extend(results[top_n:])

        return reranked_results
    except Exception as e:
        print(f"Cross-encoder reranking failed: {str(e)}")
        return results


def main(
    query: str,
    directories: list[str],
    extensions: list[str] = [".py"],
    use_cache: bool = False,
    embed_model_name: str = "nomic-embed-text-v2-moe",
    top_k: int | None = None,
    threshold: float = 0.0,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
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
        default="nomic-embed-text-v2-moe",
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
        "--chunk-size", type=int, default=1000, help="Size of text chunks"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
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
