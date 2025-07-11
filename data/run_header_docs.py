import os
import shutil
from jet.code.markdown_utils import analyze_markdown, parse_markdown
from jet.data.header_docs import HeaderDocs
from jet.data.header_types import HeaderNode
from jet.data.header_utils import split_and_merge_headers, prepare_for_rag, search_headers
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import ModelType
from jet.wordnet.analyzers.text_analysis import analyze_readability


if __name__ == "__main__":
    query_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/query.md"
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/gamerant_com_new_isekai_anime_2025/page.html"
    chunked_json_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/run_format_html/chunked_docs.json"
    # html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/page.html"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    html = load_file(html_file)
    save_file(html, f"{output_dir}/page.html")

    analysis = analyze_markdown(html_file, ignore_links=True)
    save_file(analysis, f"{output_dir}/analysis.json")

    tokens_no_merge = parse_markdown(
        html_file, ignore_links=True, merge_contents=False, merge_headers=False)
    save_file(tokens_no_merge, f"{output_dir}/markdown_tokens_no_merge.json")

    tokens = parse_markdown(html_file, ignore_links=True)
    save_file(tokens, f"{output_dir}/markdown_tokens.json")

    # header_docs = HeaderDocs.from_dicts(docs)
    header_docs = HeaderDocs.from_tokens(tokens)

    all_texts = header_docs.as_texts()
    save_file(all_texts, f"{output_dir}/all_texts.json")

    md_content = '\n\n'.join(all_texts)
    save_file(md_content, f"{output_dir}/md_content.md")

    parent_headers = [
        {"id": node.id, "chunk_index": node.chunk_index, "level": node.level, "parent_headers": "\n".join(node.get_parent_headers()).strip(
        ), "header": node.header, "content": node.content}
        for node in header_docs.as_nodes()]
    save_file(parent_headers, f"{output_dir}/parent_headers.json")

    header_nodes = [node for node in header_docs.as_nodes()
                    if node.type == "header"]
    save_file(header_nodes, f"{output_dir}/header_nodes.json")

    header_recursive_texts = [{"id": node.id, "chunk_index": node.chunk_index, "level": node.level, "parent_headers": "\n".join(node.get_parent_headers()).strip(
    ), "header": node.header, "content": node.content, "text": node.get_recursive_text()}
        for node in header_nodes]
    save_file(header_recursive_texts,
              f"{output_dir}/header_recursive_texts.json")

    all_headers = [node.header for node in header_nodes]
    save_file(all_headers, f"{output_dir}/all_headers.json")

    all_contents = [node.content for node in header_nodes]
    save_file(all_contents, f"{output_dir}/all_contents.json")

    header_tree = header_docs.as_tree()
    save_file(header_tree, f"{output_dir}/header_tree.json")

    # RAG search
    logger.info("\nStart RAG search...")

    query = load_file(query_file)
    chunk_size = None
    chunk_overlap = 40
    truncate_dim = 300
    threshold = 0.0
    top_k = None
    model: ModelType = "all-MiniLM-L6-v2"

    chunked_docs = load_file(chunked_json_file)
    context_docs = HeaderDocs.from_dicts(chunked_docs)

    rag_output_dir = f"{output_dir}/rag"

    context_docs.calculate_num_tokens(model)
    context_nodes = context_docs.as_nodes()
    save_file(context_nodes, f"{rag_output_dir}/context_nodes.json")

    SentenceTransformerRegistry.load_model(model)

    vector_store = prepare_for_rag(
        context_nodes, model=model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    save_file({
        "embeddings_shape": vector_store.get_embeddings_shape(),
        "nodes_count": len(vector_store.get_nodes())
    }, f"{rag_output_dir}/embeddings_info.json")
    save_file(vector_store.get_nodes(),
              f"{rag_output_dir}/prepared_nodes.json")
    save_file(vector_store.get_processed_texts(),
              f"{rag_output_dir}/processed_texts.json")

    search_results = search_headers(
        query, vector_store, top_k=top_k, threshold=threshold)
    search_results = sorted(
        search_results, key=lambda n: getattr(n, "score", 0), reverse=True)
    save_file({"query": query, "count": len(search_results), "results": search_results},
              f"{rag_output_dir}/search_results.json")

    # Analyze each node
    for node in search_results:
        node.metadata.update({
            "analysis": analyze_readability(node.content)
        })

    # Filter out results with mtld < 40
    search_results = [
        node for node in search_results if node.metadata["analysis"]["mtld"] >= 40]

    search_results_top_20 = [
        {
            "rank": n.rank,
            "doc_id": n.doc_id,
            "doc_index": n.doc_index,
            "chunk_index": n.chunk_index,
            "num_tokens": n.num_tokens,
            "score": n.score,
            "mtld": n.metadata["analysis"]["mtld"],
            "mtld_category": n.metadata["analysis"]["mtld_category"],
            "parent_header": n.parent_header,
            "header": n.header,
            "content": n.content
        }
        for n in search_results[:20]
    ]
    save_file({"query": query, "results": search_results_top_20},
              f"{rag_output_dir}/search_results_top_20.json")

    search_results_top_10 = search_results_top_20[:10]
    save_file({"query": query, "results": search_results_top_10},
              f"{rag_output_dir}/search_results_top_10.json")
