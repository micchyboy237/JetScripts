import os
from jet.code.markdown_utils import get_summary, parse_markdown
from jet.data.header_docs import HeaderDocs
from jet.file.utils import load_file, save_file

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/animebytes.in_15_best_upcoming_isekai_anime_in_2025/page.html"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )

    summary = get_summary(html_file)
    save_file(summary, f"{output_dir}/summary.json")

    tokens = parse_markdown(html_file)
    save_file(tokens, f"{output_dir}/markdown_tokens.json")

    header_docs = HeaderDocs.from_tokens(tokens)
    save_file(header_docs, f"{output_dir}/header_docs.json")

    header_texts = header_docs.as_texts()
    save_file(header_texts, f"{output_dir}/header_texts.json")

    header_nodes = header_docs.as_nodes()
    save_file(header_nodes, f"{output_dir}/header_nodes.json")

    header_tree = header_docs.as_tree()
    save_file(header_tree, f"{output_dir}/header_tree.json")
