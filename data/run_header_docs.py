import os
from jet.code.markdown_utils import analyze_markdown, parse_markdown
from jet.data.header_docs import HeaderDocs
from jet.file.utils import load_file, save_file

if __name__ == "__main__":
    # html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/animebytes.in_15_best_upcoming_isekai_anime_in_2025/page.html"
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/page.html"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )

    html = load_file(html_file)
    save_file(html, f"{output_dir}/page.html")

    analysis = analyze_markdown(html_file)
    save_file(analysis, f"{output_dir}/analysis.json")

    tokens = parse_markdown(html_file)
    save_file(tokens, f"{output_dir}/markdown_tokens.json")

    header_docs = HeaderDocs.from_tokens(tokens)
    save_file(header_docs, f"{output_dir}/header_docs.json")

    all_texts = header_docs.as_texts()
    save_file(all_texts, f"{output_dir}/all_texts.json")

    md_content = '\n\n'.join(all_texts)
    save_file(md_content, f"{output_dir}/md_content.md")

    all_nodes = header_docs.as_nodes()
    save_file(all_nodes, f"{output_dir}/all_nodes.json")

    header_nodes = [node for node in all_nodes if node.type == "header"]
    save_file(header_nodes, f"{output_dir}/header_nodes.json")

    header_texts = [node.content for node in header_nodes]
    save_file(header_texts, f"{output_dir}/header_texts.json")

    header_tree = header_docs.as_tree()
    save_file(header_tree, f"{output_dir}/header_tree.json")
