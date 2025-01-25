import os
from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL
from jet.llm.ollama.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.llm.query.retrievers import setup_index
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.token.token_utils import get_ollama_tokenizer
from jet.transformers import format_json
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.schema import BaseNode, Document, TextNode


def get_header_contents(md_text: str, headers_to_split_on: list[tuple[str, str]] = []) -> list[dict]:
    header_lines = []
    header_prefixes = [f"{prefix.strip()} " for prefix,
                       _ in headers_to_split_on]
    all_lines = md_text.splitlines()
    for line_idx, line in enumerate(all_lines):
        if any(line.lstrip().startswith(prefix) for prefix in header_prefixes):
            header_lines.append({"index": line_idx, "line": line})

    header_content_indexes = [item["index"]
                              for item in header_lines] + [len(all_lines)]
    header_content_ranges = [(header_content_indexes[item_idx], header_content_indexes[item_idx + 1])
                             for item_idx, _ in enumerate(header_lines)]
    header_groups = []
    previous_added_lines = 0
    for start_idx, end_idx in header_content_ranges:
        start_idx += previous_added_lines
        end_idx += previous_added_lines
        header_line, *contents = all_lines[start_idx: end_idx]
        header_level = get_header_level(header_line)
        content = "\n".join(contents)

        start_line_idx = start_idx - previous_added_lines
        end_line_idx = end_idx - previous_added_lines

        # if not content.strip():
        #     lines_to_insert = ["", "<placeholder>", ""]
        #     previous_added_lines += len(lines_to_insert)
        #     all_lines[end_idx:end_idx] = lines_to_insert  # Inserts these lines

        details = content if content else "<placeholder>"
        block_content = f"{header_line}\n\n{details}\n\n"

        header_groups.append({
            "header": header_line,
            "details": content,
            "content": block_content,
            "metadata": {
                "start_line_idx": start_line_idx,
                "end_line_idx": end_line_idx,
                "depth": header_level,
            }
        })

    md_text = "\n".join([item["content"] for item in header_groups])

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on, strip_headers=False, return_each_line=False)
    md_header_splits = markdown_splitter.split_text(md_text)
    md_header_contents = []
    for split_idx, split in enumerate(md_header_splits):
        content = split.page_content
        # metadata = split.metadata

        # Remove placeholder text
        content = content.replace("<placeholder>", "")

        # Remove unwanted trailing line spaces
        content_lines = [line.rstrip() for line in content.splitlines()]
        content = "\n".join(content_lines)

        content = content.strip()

        header_group = header_groups[split_idx]

        md_header_contents.append({
            "heading": header_group["header"],
            "details": header_group["details"],
            "content": content,
            "length": len(content),
            "metadata": {
                **header_group["metadata"],
                "tags": list(split.metadata.values())
            }
        })
    return md_header_contents


def get_header_level(header: str) -> int:
    """Get the header level of a markdown header or HTML header tag."""
    if header.startswith("#"):
        header_level = 0
        for c in header:
            if c == "#":
                header_level += 1
            else:
                break
        return header_level
    elif header.startswith("h") and header[1].isdigit() and 1 <= int(header[1]) <= 6:
        return int(header[1])
    else:
        raise ValueError(f"Invalid header format: {header}")


def get_file_contents(base_dir: str, extensions: list[str]) -> list[dict]:
    """
    Recursively reads all file contents from a directory with the specified extensions
    and combines them into a single string separated by newlines.

    Args:
        base_dir (str): The base directory to search for files.
        extensions (list[str]): List of file extensions to include (e.g., [".md", ".mdx"]).

    Returns:
        str: Combined contents of all matching files.
    """
    contents = []

    # Normalize extensions for consistency
    normalized_extensions = [ext.lower() if ext.startswith(
        ".") else f".{ext.lower()}" for ext in extensions]

    for root, _, files in os.walk(base_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in normalized_extensions):
                file_path = os.path.join(root, file)
                file_path = os.path.abspath(file_path)
                file_name = os.path.relpath(file_path, start=base_dir)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        contents.append({"metadata": {
                            "file_name": file_name, "file_path": file_path
                        }, "content": f.read()})
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    return contents


if __name__ == "__main__":
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    extensions = [".md", ".mdx"]

    contents = get_file_contents(base_dir, extensions)

    all_docs: list[Document] = []

    for item in contents:
        file_metadata = item["metadata"]
        md_text = item["content"]

        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ]

        header_contents = get_header_contents(md_text, headers_to_split_on)
        header_contents = [{**item, "metadata": {**item["metadata"],
                                                 **file_metadata}} for item in header_contents]
        # filtered_header_contents = [
        #     item for item in header_contents if item['details'].strip()]

        documents = [Document(text=item["content"], metadata=item["metadata"])
                     for item in header_contents]
        all_docs.extend(documents)

    # Split nodes
    chunk_size = 1024
    chunk_overlap = 100
    embed_model = OLLAMA_SMALL_EMBED_MODEL

    query_nodes = setup_index(
        all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Search nodes
    mode = "fusion"
    top_k = None
    score_threshold = 0.0
    sample_query = "Tell me about yourself."

    logger.newline()
    logger.info("Running vector search...")
    result = query_nodes(
        sample_query, threshold=score_threshold, top_k=top_k)
    logger.debug(f"Retrieved nodes ({len(result["nodes"])})")
    display_jet_source_nodes(sample_query, result["nodes"])
