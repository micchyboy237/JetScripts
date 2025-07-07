import os
import shutil
from typing import List, Union
from jet.code.markdown_utils._converters import convert_html_to_markdown, convert_markdown_to_html
from jet.code.markdown_utils._markdown_parser import parse_markdown
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
from jet.models.model_types import LLMModelType
from jet.models.utils import resolve_model_value
from mlx_lm import load
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.scrapers.utils import extract_by_heading_hierarchy, extract_texts_by_hierarchy, extract_tree_with_text, extract_text_elements, print_html
from jet.search.formatters import clean_string
from jet.transformers.formatters import format_html
from jet.utils.commands import copy_to_clipboard


if __name__ == "__main__":
    from jet.scrapers.preprocessor import html_to_markdown

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/gamerant_com_new_isekai_anime_2025/page.html"
    html_dir = os.path.dirname(html_file)
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    model_path: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"

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
    model_id = resolve_model_value(model_path)
    model, tokenizer = load(model_id)

    # Chunk docs with chunk size
    chunk_size = 150

    def _tokenizer(text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(text, str):
            token_ids = tokenizer.encode(
                text, add_special_tokens=False)
            return tokenizer.convert_ids_to_tokens(token_ids)
        else:
            token_ids_list = tokenizer.batch_encode_plus(
                text, add_special_tokens=False)["input_ids"]
            return [tokenizer.convert_ids_to_tokens(ids) for ids in token_ids_list]

    doc_markdown_tokens = parse_markdown(html_str, ignore_links=False)
    doc_markdown = "\n\n".join([item["content"]
                               for item in doc_markdown_tokens])
    # doc_html = convert_markdown_to_html(doc_markdown)
    save_file(doc_markdown, f"{output_dir}/doc_markdown.md")

    chunked_docs = chunk_headers_by_hierarchy(
        doc_markdown, chunk_size, _tokenizer)
    save_file(chunked_docs, f"{output_dir}/chunked_docs.json")

    html_docs = [item["content"] for item in chunked_docs]
    # Group by parent_header first, then by header
    parent_header_groups = {}
    used_headers = set()  # Track used parent headers for grouping

    for doc in chunked_docs:
        parent_header = doc.get("parent_header") or ""
        header = doc.get("header") or ""

        if parent_header not in parent_header_groups:
            parent_header_groups[parent_header] = {}
            used_headers.add(parent_header)

        if header not in parent_header_groups[parent_header]:
            parent_header_groups[parent_header][header] = []

        parent_header_groups[parent_header][header].append(doc["content"])

    # Create markdown with parent headers as group headers, deduplicating globally
    grouped_markdown_parts = []
    appended_headers = set()  # Track all appended headers (parent and subheaders)

    for parent_header, header_groups in parent_header_groups.items():
        # Only append parent header if it hasn't been used before
        if parent_header and parent_header not in appended_headers:
            grouped_markdown_parts.append(parent_header)
            appended_headers.add(parent_header)

        for header, contents in header_groups.items():
            # Only append header if it hasn't been used before
            if header and header not in appended_headers:
                grouped_markdown_parts.append(header)
                appended_headers.add(header)
            grouped_markdown_parts.extend(contents)

    chunked_markdown = "\n\n".join(grouped_markdown_parts)
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
