import re
from typing import List, Dict, Any, TypedDict

from jet.llm.main.vector_semantic_search import VectorSemanticSearch
from jet.llm.query.retrievers import query_llm
from jet.logger import logger
from jet.token.token_utils import token_counter
from jet.file import load_file


class SummaryTokens(TypedDict):
    summary: str
    tokens: int


def group_summaries(
    summaries: list[str],
    max_tokens_per_group: int,
    tokenizer_model: str,
    separator: str = "\n\n\n"
) -> list[SummaryTokens]:
    grouped_summaries = []
    current_group = []
    current_tokens = 0

    for idx, summary in enumerate(summaries):
        prefixed_summary = f"Summary {idx + 1}\n\n{summary}"
        prefixed_summary_tokens: int = token_counter(
            prefixed_summary, tokenizer_model)

        combined_summary = separator.join(
            s["summary"] for s in current_group)
        combined_tokens: int = token_counter(
            combined_summary, tokenizer_model)
        next_group_tokens = prefixed_summary_tokens + combined_tokens

        if current_group and next_group_tokens >= max_tokens_per_group:
            grouped_summaries.append({
                "summary": combined_summary,
                "tokens": combined_tokens
            })
            current_group = []
            current_tokens = 0

        current_group.append(
            {"summary": prefixed_summary, "tokens": prefixed_summary_tokens})
        current_tokens += prefixed_summary_tokens

    if current_group:
        combined_summary = separator.join(s["summary"] for s in current_group)
        combined_tokens: int = token_counter(combined_summary, tokenizer_model)
        grouped_summaries.append({
            "summary": combined_summary,
            "tokens": combined_tokens
        })

    return grouped_summaries


# Example usage
if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"

    data = load_file(data_file)

    texts = [item['details'] for item in data]

    search = VectorSemanticSearch(texts)

    query = "I am applying for the position of a Frontend Web / Mobile Developer or Full Stack Developer roles. Aside from React, React Native and Node.js. I also have extensive experience with Firebase, AWS, MongoDB and PostgreSQL."
    queries = query.splitlines()

    # Perform Fusion search
    fusion_results = search.fusion_search(queries)
    logger.info(f"\nFusion Search Results ({len(fusion_results)}):")

    for result in fusion_results:
        text_cleaned = re.sub(r"\s+", " ", result['text'])
        logger.log(f"{text_cleaned[:50]}:", f"{
            result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform FAISS search
    faiss_results = search.faiss_search(queries)
    logger.info("\nFAISS Search Results:")
    for query_line, group in faiss_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            text_cleaned = re.sub(r"\s+", " ", result['text'])
            logger.log(f"{text_cleaned[:50]}:", f"{
                result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform Graph-based search
    graph_results = search.graph_based_search(queries)
    logger.info("\nGraph-Based Search Results:")
    for query_line, group in graph_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            text_cleaned = re.sub(r"\s+", " ", result['text'])
            logger.log(f"{text_cleaned[:50]}:", f"{
                result['score']:.4f}", colors=["DEBUG", "SUCCESS"])
