from typing import List, Dict
from jet.llm.query.cleaners import group_and_merge_texts_by_file_name
from llama_index.core.schema import BaseNode, TextNode
import unittest


# Example Usage


def main():
    nodes = [
        TextNode(text="Header 1\nContent 1",
                 metadata={"file_name": "file1.md"}),
        TextNode(text="Content 1\nHeader 2\nContent 2",
                 metadata={"file_name": "file1.md"}),
        TextNode(text="Header A\nContent A",
                 metadata={"file_name": "file2.md"}),
        TextNode(text="Content A\nHeader B\nContent B",
                 metadata={"file_name": "file2.md"}),
    ]

    result = group_and_merge_texts_by_file_name(nodes)

    for file_name, content in result.items():
        print(f"File: {file_name}\n{content}\n")


if __name__ == "__main__":
    main()
