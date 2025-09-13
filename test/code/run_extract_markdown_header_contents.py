from jet.code.splitter_markdown_utils import extract_md_header_contents
from jet.llm.query.splitters import split_markdown_header_nodes
from jet.logger import logger
from jet._token.token_utils import get_tokenizer
from jet.transformers.formatters import format_json
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.schema import Document
import numpy as np
from jet.file.utils import load_file


def main():
    items = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/scrapy/generated/passport/_main_preprocessed.json")
    # documents = [Document(text=item['content']) for item in items]
    # all_nodes = split_markdown_header_nodes(documents)
    # texts = [item.text for item in all_nodes]
    # markdown_text = "\n\n".join(texts)

    markdown_text = "\n\n".join([item['content'] for item in items])

    tokenizer = get_tokenizer("llama3.2")
    header_contents = extract_md_header_contents(
        markdown_text, tokenizer=tokenizer.encode)
    # logger.success(format_json(all_nodes))
    logger.success(format_json(header_contents))


if __name__ == "__main__":
    main()
