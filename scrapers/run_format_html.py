

# Example Usage
import os
import shutil
from typing import List, Union
from jet.code.splitter_markdown_utils import get_md_header_contents
from mlx_lm import load
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.scrapers.utils import extract_by_heading_hierarchy, extract_texts_by_hierarchy, extract_tree_with_text, extract_text_elements, merge_texts_by_hierarchy, print_html
from jet.search.formatters import clean_string
from jet.transformers.formatters import format_html
from jet.utils.commands import copy_to_clipboard


if __name__ == "__main__":
    from jet.scrapers.preprocessor import html_to_markdown

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/page.html"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/run_format_html"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    html_str: str = load_file(data_file)

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

    # Merge docs with max tokens
    max_tokens = 300

    def _tokenizer(text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(text, str):
            token_ids = tokenizer.encode(
                text, add_special_tokens=False)
            return tokenizer.convert_ids_to_tokens(token_ids)
        else:
            token_ids_list = tokenizer.batch_encode_plus(
                text, add_special_tokens=False)["input_ids"]
            return [tokenizer.convert_ids_to_tokens(ids) for ids in token_ids_list]

    merged_docs = merge_texts_by_hierarchy(
        html_str, _tokenizer, max_tokens)
    save_file(merged_docs, f"{output_dir}/merged_docs.json")

    html_docs = [item["text"] for item in merged_docs]
    md_text = "\n\n".join(html_docs)
    save_file(md_text, f"{output_dir}/md_text.md")

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
