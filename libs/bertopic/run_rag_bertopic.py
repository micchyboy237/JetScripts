"""
BERTopic RAG Runner

Summary:
    Executes a Retrieval-Augmented Generation (RAG) pipeline using BERTopic for
    topic modeling and Llama.cpp for embeddings. It processes job descriptions,
    clusters them into topics, and performs hybrid search (vector + keyword)
    to retrieve relevant documents based on a user query.

Usage Examples:
    # Basic run with a required positional query
    python run_rag_bertopic.py "Data Engineer SQL"

    # Advanced usage with custom hybrid weighting and verbose logging
    python run_rag_bertopic.py "React Developer" --alpha 0.5 --verbose --top-k 5

    # Custom output directory and topic constraints
    python run_rag_bertopic.py "AI Agent" --output-dir ./results --min-topic-size 10
"""

import argparse
import os
import shutil

from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_jobs
from jet.libs.bertopic.rag_bertopic import TopicRAG


def get_args():
    parser = argparse.ArgumentParser(
        description="Run BERTopic RAG example with configurable parameters."
    )

    # Positional Arguments
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="AI Engineer",
        help="The search query for retrieval (default: 'AI Engineer')",
    )

    # General Settings
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (defaults to 'generated/<script_name>')",
    )

    # Model & Embedding
    parser.add_argument(
        "--model-name",
        type=str,
        default="nomic-embed:2-moe",
        help="Llama.cpp embedding model name",
    )

    # Topic Modeling
    parser.add_argument(
        "--nr-topics", type=str, default="auto", help="Number of topics (int or 'auto')"
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=2,
        help="Minimum number of documents per topic",
    )

    # Retrieval & Hybrid Search
    parser.add_argument(
        "--top-topics",
        type=int,
        default=3,
        help="Number of top topics to search within",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of documents to retrieve per query"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for vector search in hybrid search (1-alpha for keyword)",
    )
    parser.add_argument(
        "--unique-by",
        type=str,
        choices=["text", None],
        default=None,
        help="Ensure uniqueness by field",
    )

    return parser.parse_args()


def run_example_rag_bertopic(args):
    """Demonstrates RAG retrieval with varying document sets and queries."""

    # Setup Output Directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__),
            "generated",
            os.path.splitext(os.path.basename(__file__))[0],
        )

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Data
    docs = load_sample_jobs()

    # Initialize RAG
    rag = TopicRAG(model_name=args.model_name, verbose=args.verbose)

    # Fit Topics
    rag.fit_topics(docs, nr_topics=args.nr_topics, min_topic_size=args.min_topic_size)

    # Save Topic Info
    save_file(
        rag.model.get_topic_info().to_dict(orient="records"),
        f"{args.output_dir}/topic_info.json",
    )

    # Perform Retrieval
    search_results = rag.retrieve_for_query(
        query=args.query,
        top_topics=args.top_topics,
        top_k=args.top_k,
        unique_by=args.unique_by,
        alpha=args.alpha,
    )

    # Save Results
    save_file(search_results, f"{args.output_dir}/search_results.json")

    print(f"\n✅ Process complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    args = get_args()
    run_example_rag_bertopic(args)
