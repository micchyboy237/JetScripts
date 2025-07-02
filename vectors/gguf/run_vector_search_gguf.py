from jet.code.markdown_utils import parse_markdown
from jet.data.header_types import TextNode
from jet.file.utils import load_file
from jet.vectors.document_types import HeaderDocument
from jet.vectors.gguf.vector_search import VectorSearch, get_detailed_instruct


if __name__ == "__main__":
    query_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/query.md"
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/animebytes_in_15_best_upcoming_isekai_anime_in_2025/page.html"
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/generated/run_header_docs/rag/all_nodes.json"

    # Model configuration
    model_path = "/Users/jethroestrada/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/8aa0010e73a1075e99dfc213a475a60fd971bbe7/Qwen3-Embedding-0.6B-f16.gguf"

    md_contents = parse_markdown(html_file)
    nodes = load_file(docs_file)
    nodes = [TextNode(**node) for node in nodes]
    query = load_file(query_file)

    header_texts = [node.header for node in nodes]
    content_texts = [node.content for node in nodes]
    all_texts = [node.get_text() for node in nodes]

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    top_k = 10

    queries = [
        get_detailed_instruct(task, query)
    ]
    documents = all_texts
    # documents = [
    #     {"id": doc.id, "text": doc.text}
    #     for doc in docs
    # ]
    documents = [
        {"id": f"doc_{idx}", "text": doc["content"]}
        for idx, doc in enumerate(md_contents)
    ]

    # Initialize vector search
    searcher = VectorSearch(model_path, n_ctx=2048,
                            n_threads=4, n_gpu_layers=0)

    # Example 1: Single query (same as original)
    print(f"\nExample 1: Single query - {query}")
    results = searcher.search(queries, documents, top_k=top_k)
    for query_idx, query_results in enumerate(results):
        print(f"Query: {queries[query_idx].split('Query:')[-1].strip()}")
        for result in query_results:
            print(
                f"ID: {result['id']}, Rank: {result['rank']}, Score: {result['score']:.4f}")
            print(f"Text: {result['text']}")
            print()

    # # Example 2: Multiple queries
    # print("\nExample 2: Multiple queries")
    # queries_multi = [
    #     get_detailed_instruct(task, 'What is the capital of China?'),
    #     get_detailed_instruct(task, 'Explain gravity')
    # ]
    # results = searcher.search(queries_multi, documents, top_k=top_k)
    # for query_idx, query_results in enumerate(results):
    #     print(
    #         f"Query: {queries_multi[query_idx].split('Query:')[-1].strip()}")
    #     for result in query_results:
    #         print(
    #             f"ID: {result['id']}, Rank: {result['rank']}, Score: {result['score']:.4f}")
    #         print(f"Text: {result['text']}")
    #         print()

    # # Example 3: Empty query
    # print("\nExample 3: Empty query")
    # queries_empty = [get_detailed_instruct(task, '')]
    # results = searcher.search(queries_empty, documents, top_k=top_k)
    # for query_idx, query_results in enumerate(results):
    #     print(
    #         f"Query: {queries_empty[query_idx].split('Query:')[-1].strip() or '(empty)'}")
    #     for result in query_results:
    #         print(
    #             f"ID: {result['id']}, Rank: {result['rank']}, Score: {result['score']:.4f}")
    #         print(f"Text: {result['text']}")
    #         print()

    # # Example 4: Long query
    # print("\nExample 4: Long query")
    # queries_long = [get_detailed_instruct(
    #     task, 'What is the historical significance of Beijing as the capital of China, and how has it influenced Chinese culture?')]
    # results = searcher.search(queries_long, documents, top_k=top_k)
    # for query_idx, query_results in enumerate(results):
    #     print(
    #         f"Query: {queries_long[query_idx].split('Query:')[-1].strip()}")
    #     for result in query_results:
    #         print(
    #             f"ID: {result['id']}, Rank: {1}, Score: {result['score']:.4f}")
    #         print(f"Text: {result['text']}")
    #         print()
