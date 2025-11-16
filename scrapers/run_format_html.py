import os
import shutil
from typing import List
from jet.code.html_utils import clean_html, convert_dl_blocks_to_md, preprocess_html
from jet.code.markdown_utils._converters import convert_html_to_markdown, convert_markdown_to_html
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy, parse_markdown
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
from jet.scrapers.header_hierarchy import HtmlHeaderDoc, extract_header_hierarchy
from jet.scrapers.text_nodes import extract_text_nodes
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_by_heading_hierarchy, extract_tree_with_text, extract_text_elements, flatten_tree_to_base_nodes, get_leaf_nodes, get_parents_with_common_class, print_html
from jet.transformers.formatters import format_html


if __name__ == "__main__":

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_rag_context_engineering_tips_2025_reddit/https_www_reddit_com_r_rag_comments_1mvzwrq_context_engineering_for_advanced_rag_curious_how/page.html"
    html_dir = os.path.dirname(html_file)
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"

    html_str: str = load_file(html_file)
    save_file(html_str, f"{output_dir}/doc.html")

    preprocessed_html_str = convert_dl_blocks_to_md(html_str)
    preprocessed_html_str = preprocess_html(preprocessed_html_str, excludes=["head", "nav", "footer", "script", "style", "button"])
    save_file(preprocessed_html_str, f"{output_dir}/doc_preprocessed.html")

    html_str = preprocessed_html_str

    doc_markdown = convert_html_to_markdown(html_str, ignore_links=False)
    save_file(doc_markdown, f"{output_dir}/doc_markdown.md")
    
    doc_markdown_no_links = convert_html_to_markdown(html_str, ignore_links=True)
    save_file(doc_markdown_no_links, f"{output_dir}/doc_markdown_no_links.md")

    # Texts
    texts = extract_text_elements(html_str)
    save_file(texts, f"{output_dir}/texts.json")

    text_nodes = extract_text_nodes(html_str)
    save_file(text_nodes, f"{output_dir}/text_nodes.json")

    # Headings
    headings: List[HtmlHeaderDoc] = extract_header_hierarchy(html_str)
    save_file(headings, f"{output_dir}/headings.json")

    headings2 = extract_by_heading_hierarchy(html_str)
    save_file(headings2, f"{output_dir}/headings2.json")

    headings3 = derive_by_header_hierarchy(doc_markdown, ignore_links=True)
    save_file(headings3, f"{output_dir}/headings3.json")

    headings4 = get_md_header_contents(html_str, ignore_links=True)
    save_file(headings4, f"{output_dir}/headings4.json")

    header_texts = []
    for idx, node in enumerate(headings):
        texts = [
            f"Document {idx + 1} | Tag ({node["element"]["tag"]}) | Depth ({node["depth"]})"
        ]
        if node.parent_id:
            texts.append(f"Parent ({node.parent_id})")

        child_texts = [
            child_node.text or " " for child_node in node.get_children()]

        texts.extend([
            "Text:",
            node.text + "\n" + ''.join(child_texts)
        ])
        header_texts.append("\n".join(texts))
    save_file("\n\n---\n\n".join(header_texts), f"{output_dir}/headings.md")

    texts = [item.text for item in headings]

    # # Load the model and tokenizer
    # model_id = resolve_model_value(llm_model)
    # model, tokenizer = load(model_id)

    # Chunk docs with chunk size
    chunk_size = 150

    # def _tokenizer(text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
    #     if isinstance(text, str):
    #         token_ids = tokenizer.encode(
    #             text, add_special_tokens=False)
    #         return tokenizer.convert_ids_to_tokens(token_ids)
    #     else:
    #         token_ids_list = tokenizer.batch_encode_plus(
    #             text, add_special_tokens=False)["input_ids"]
    #         return [tokenizer.convert_ids_to_tokens(ids) for ids in token_ids_list]

    doc_analysis = analyze_markdown(doc_markdown)
    save_file(doc_analysis, f"{output_dir}/doc_analysis.json")

    doc_markdown_tokens_no_merge = parse_markdown(
        doc_markdown, ignore_links=False, merge_headers=False,  merge_contents=False)
    save_file(doc_markdown_tokens_no_merge,
              f"{output_dir}/doc_markdown_tokens_no_merge.json")

    doc_markdown_tokens_no_merge_html_template = "<body>{body}</body>"
    doc_markdown_tokens_no_merge_html_list = [convert_markdown_to_html(
        token["content"]) for token in doc_markdown_tokens_no_merge]
    doc_markdown_tokens_no_merge_html = doc_markdown_tokens_no_merge_html_template.format(
        body="\n".join(doc_markdown_tokens_no_merge_html_list))
    doc_markdown_tokens_no_merge_html = format_html(
        doc_markdown_tokens_no_merge_html)
    save_file(doc_markdown_tokens_no_merge_html,
              f"{output_dir}/doc_markdown_tokens_no_merge.html")
    save_file(clean_html(doc_markdown_tokens_no_merge_html),
              f"{output_dir}/clean_doc_markdown_tokens_no_merge.json")

    base_doc_markdown_tokens = base_parse_markdown(
        doc_markdown, ignore_links=False)
    save_file(base_doc_markdown_tokens,
              f"{output_dir}/base_doc_markdown_tokens.json")

    doc_markdown_tokens = parse_markdown(doc_markdown, ignore_links=False)
    save_file(doc_markdown_tokens, f"{output_dir}/doc_markdown_tokens.json")

    doc_markdown_tokens_html_template = "<body>{body}</body>"
    doc_markdown_tokens_html_list = [convert_markdown_to_html(
        token["content"]) for token in doc_markdown_tokens]
    save_file(doc_markdown_tokens_html_list,
              f"{output_dir}/doc_markdown_tokens_html_list.json")
    doc_markdown_tokens_html = doc_markdown_tokens_html_template.format(
        body="\n".join(doc_markdown_tokens_html_list))
    doc_markdown_tokens_html = format_html(
        doc_markdown_tokens_html)
    save_file(doc_markdown_tokens_html,
              f"{output_dir}/doc_markdown_tokens.html")
    save_file(clean_html(doc_markdown_tokens_html),
              f"{output_dir}/clean_doc_markdown_tokens.json")

    chunked_docs = chunk_headers_by_hierarchy(doc_markdown, chunk_size)
    save_file(chunked_docs, f"{output_dir}/chunked_docs.json")

    # chunked_docs = merge_same_level_chunks(
    #     chunked_docs, chunk_size, _tokenizer)
    # save_file({"chunk_size": chunk_size, "count": len(chunked_docs),
    #           "results": chunked_docs}, f"{output_dir}/merged_chunked_docs.json")

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

    # Get the tree-like structure
    tree_elements = extract_tree_with_text(html_str)
    save_file(tree_elements, f"{output_dir}/tree_elements.json")

    flattened_nodes = flatten_tree_to_base_nodes(tree_elements)
    save_file(flattened_nodes, f"{output_dir}/flattened_nodes.json")

    leaf_nodes = get_leaf_nodes(tree_elements, with_text=True)
    save_file(leaf_nodes, f"{output_dir}/leaf_nodes.json")

    parents_with_common_class = get_parents_with_common_class(tree_elements)
    save_file(parents_with_common_class,
              f"{output_dir}/parents_with_common_class.json")

    formatted_html = format_html(html_str)
    save_file(formatted_html, f"{output_dir}/formatted_html.html")

    # significant_nodes = get_significant_nodes(tree_elements)
    # save_file(significant_nodes, f"{output_dir}/significant_nodes.json")

    # for num, significant_node in enumerate(significant_nodes, start=1):
    #     sub_output_dir = f"{output_dir}/significant_node_{num}"
    #     node_html = significant_node.get_html()

    #     if not node_html:
    #         continue

    #     save_file(node_html, f"{sub_output_dir}/node.html")

    #     node_md_content = convert_html_to_markdown(node_html)
    #     save_file(node_md_content, f"{sub_output_dir}/node.md")

    #     analysis = analyze_markdown(node_md_content)
    #     save_file(analysis, f"{sub_output_dir}/analysis.json")

    #     markdown_tokens = base_parse_markdown(node_md_content)
    #     save_file({
    #         "count": len(markdown_tokens),
    #         "tokens": markdown_tokens,
    #     }, f"{sub_output_dir}/markdown_tokens.json")

    #     header_docs = derive_by_header_hierarchy(node_md_content)
    #     save_file({
    #         "count": len(header_docs),
    #         "documents": header_docs,
    #     }, f"{sub_output_dir}/header_docs.json")

    print_html(html_str)
