

# Example Usage
import os
import shutil
from typing import List, Union
from jet.code.markdown_utils._markdown_parser import parse_markdown
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.models.embeddings.utils import chunk_headers_by_hierarchy
from mlx_lm import load
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.scrapers.utils import extract_by_heading_hierarchy, extract_texts_by_hierarchy, extract_tree_with_text, extract_text_elements, print_html
from jet.search.formatters import clean_string
from jet.transformers.formatters import format_html
from jet.utils.commands import copy_to_clipboard


if __name__ == "__main__":
    from jet.scrapers.preprocessor import html_to_markdown

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/animebytes_in_15_best_upcoming_isekai_anime_in_2025/page.html"
    # html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/page.html"
    html_dir = os.path.dirname(html_file)
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    html_str: str = load_file(html_file)

    save_file(format_html(html_str), f"{output_dir}/doc.html")

    # # Text elements
    # text_elements = extract_text_elements(html_str)
    # save_file(text_elements, f"{output_dir}/text_elements.json")

    # Headings
    headings = extract_texts_by_hierarchy(html_str, ignore_links=True)
    save_file(headings, f"{output_dir}/headings.json")

    headings2 = get_md_header_contents(html_str)
    save_file(headings2, f"{output_dir}/headings2.json")

    texts = [item["text"] for item in headings]

    # Load the model and tokenizer
    model, tokenizer = load(model_path)

    # Chunk docs with chunk size
    chunk_size = 300

    def _tokenizer(text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(text, str):
            token_ids = tokenizer.encode(
                text, add_special_tokens=False)
            return tokenizer.convert_ids_to_tokens(token_ids)
        else:
            token_ids_list = tokenizer.batch_encode_plus(
                text, add_special_tokens=False)["input_ids"]
            return [tokenizer.convert_ids_to_tokens(ids) for ids in token_ids_list]

    doc_markdown_tokens = parse_markdown(html_str)
    doc_markdown = "\n\n".join([item["content"]
                               for item in doc_markdown_tokens])
    save_file(doc_markdown, f"{output_dir}/doc_markdown.md")

    chunked_docs = chunk_headers_by_hierarchy(
        doc_markdown, chunk_size, _tokenizer)
    save_file(chunked_docs, f"{output_dir}/chunked_docs.json")

    html_docs = [item["content"] for item in chunked_docs]
    chunked_markdown = "\n\n".join(html_docs)
    save_file(chunked_markdown, f"{output_dir}/chunked_markdown.md")

    # By headings
    header_elements = extract_by_heading_hierarchy(html_str)
    save_file(header_elements, f"{output_dir}/headings_elements.json")

    header_texts = []
    for idx, node in enumerate(header_elements):
        texts = [
            f"Document {idx + 1} | Tag ({node.tag}) | Depth ({node.depth})"
        ]
        if node.parent:
            texts.append(f"Parent ({node.parent})")

        child_texts = [child_node.text or " " for child_node in node.children]

        texts.extend([
            "Text:",
            node.text + "\n" + ''.join(child_texts)
        ])
        header_texts.append("\n".join(texts))
    save_file("\n\n---\n\n".join(header_texts), f"{output_dir}/headings.md")

    # Get the tree-like structure
    tree_elements = extract_tree_with_text(html_str)
    save_file(tree_elements, f"{output_dir}/tree_elements.json")

    formatted_html = format_html(html_str)
    save_file(formatted_html, f"{output_dir}/formatted_html.html")

    print_html(html_str)
