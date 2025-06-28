import os
from jet.code.markdown_utils import parse_markdown
from jet.code.splitter_markdown_utils import get_md_header_docs
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType
from jet.logger import logger
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs
from jet.models.tokenizer.base import count_tokens
from jet.models.model_types import LLMModelType
from jet.vectors.document_types import HeaderDocument
from jet.vectors.document_utils import get_leaf_documents
from jet.wordnet.text_chunker import chunk_headers


def main(with_bm25: bool):
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/page.html"
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/docs.json"
    # headers_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/generated/run_header_docs/header_texts.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    embed_model: EmbedModelType = "static-retrieval-mrl-en-v1"

    docs = load_file(docs_file)
    query = docs["query"]
    source_url = docs["source_url"]

    md_contents = parse_markdown(html_file)
    # md_content = "\n".join(item["content"] for item in md_contents)
    # docs = get_md_header_docs(md_content, base_url=source_url)

    # headers = load_file(headers_file)

    # docs = HeaderDocument.from_markdown(html)

    docs = [
        {
            "text": item["content"].lstrip('#').strip(),
            "metadata": {
                "content": item["content"],
                "doc_index": doc_index,
                "chunk_index": 0,
            }
        }
        for doc_index, item in enumerate(md_contents)
    ]
    docs = HeaderDocument.from_list(docs)
    for doc in docs:
        doc.metadata.update({
            "parent_header": ""
        })

    # chunked_docs = chunk_headers(docs, max_tokens=300, model=embed_model)
    # docs = chunked_docs
    # docs_to_search = [doc for doc in docs if doc.metadata["content"].strip()]
    # logger.debug(
    #     f"Filtered to {len(docs_to_search)} documents for search (excluding header level 1)")
    results = search_docs(
        query,
        documents=docs,
        ids=[doc.id for doc in docs],
        model=embed_model,
        top_k=None,
        with_bm25=with_bm25,
        # threshold=0.7
    )

    logger.info(f"Counting tokens ({len(results)})...")
    token_counts: list[int] = count_tokens(
        llm_model, [result['text'] for result in results], prevent_total=True)

    for result, tokens in zip(results, token_counts):
        logger.success(
            f"\nRank {result['rank']} (Doc: {result['doc_index']} | Tokens: {tokens}):")
        print(f"Embedding Score: {result['embedding_score']:.4f}")
        print(f"Combined Score: {result['combined_score']:.4f}")
        print(f"Final Score: {result['score']:.4f}")
        print(f"Headers: {result['headers']}")
        print(f"Original Document:\n{result['text']}")

    result_texts = [result["text"] for result in results]
    context_tokens: list[int] = count_tokens(
        llm_model, result_texts, prevent_total=True)
    total_tokens = sum(context_tokens)

    output_path = f"{output_dir}/results_with_bm25.json" if with_bm25 else f"{output_dir}/results_no_bm25.json"
    save_file(
        {
            "query": query,
            "total_tokens": total_tokens,
            "count": len(results),
            "with_bm25": with_bm25,
            "urls_info": {
                result["metadata"]["source_url"]: len(
                    [r for r in results if r["metadata"]["source_url"] == result["metadata"]["source_url"]])
                for result in results
            },
            "contexts": [
                {
                    "rank": result["rank"],
                    "doc_index": result["doc_index"],
                    "chunk_index": result["chunk_index"],
                    "score": result["score"],
                    "tokens": tokens,
                    "source_url": result["metadata"]["source_url"],
                    "parent_header": result["metadata"]["parent_header"],
                    "header": result["metadata"]["header"],
                    "text": result["text"]
                }
                for result, tokens in zip(results, context_tokens)
            ]
        },
        output_path
    )

    output_path = f"{output_dir}/tokens_with_bm25.json" if with_bm25 else f"{output_dir}/tokens_no_bm25.json"
    save_file(token_counts, output_path)


if __name__ == "__main__":
    main(with_bm25=True)
    main(with_bm25=False)
