from typing import List, Optional
from jet.code.markdown_types import ContentType, MetaType
from jet.data.header_types import HeaderNode, TextNode, NodeType, Nodes
from jet.data.header_utils import split_and_merge_headers
from jet.models.model_types import ModelType
from jet.models.tokenizer.base import get_tokenizer
from tokenizers import Tokenizer


def main_deeply_nested_with_chunking(tokenizer: Tokenizer) -> Nodes:
    # Given: A deeply nested structure with a leaf node requiring chunking
    content = "This is a long sentence. " * 20
    nodes = [
        HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Level 1 Header",
            content="Level 1 content",
            level=1,
            children=[
                HeaderNode(
                    id="header2",
                    line=2,
                    type="header",
                    header="Level 2 Header",
                    content="Level 2 content",
                    level=2,
                    children=[
                        HeaderNode(
                            id="header3",
                            line=3,
                            type="header",
                            header="Level 3 Header",
                            content="Level 3 content",
                            level=3,
                            children=[
                                TextNode(
                                    id="child1",
                                    line=4,
                                    type="paragraph",
                                    header="Child Header",
                                    content=content,
                                    meta=None,
                                    parent_id="header3",
                                    parent_header="Level 3 Header"
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
    chunk_size = 50
    chunk_overlap = 10
    buffer = 5
    expected_chunk_count = 3  # Approximate, based on token count

    # When: We call split_and_merge_headers with chunking
    result_nodes = split_and_merge_headers(
        docs=nodes,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        buffer=buffer
    )

    # Then: The output should preserve hierarchy and chunk correctly
    assert len(result_nodes) >= expected_chunk_count + \
        3  # Headers + chunks
    header_nodes = [n for n in result_nodes if n.line in [1, 2, 3]]
    chunk_nodes = [n for n in result_nodes if n.line == 4]
    assert len(header_nodes) == 3
    assert len(chunk_nodes) >= expected_chunk_count
    assert header_nodes[0].header == "Level 1 Header"
    assert header_nodes[0].content == "Level 1 Header\nLevel 1 content"
    assert header_nodes[0].parent_id is None
    assert header_nodes[0].parent_header is None
    assert header_nodes[1].header == "Level 2 Header"
    assert header_nodes[1].content == "Level 2 Header\nLevel 2 content"
    assert header_nodes[1].parent_id == "header1"
    assert header_nodes[1].parent_header == "Level 1 Header"
    assert header_nodes[2].header == "Level 3 Header"
    assert header_nodes[2].content == "Level 3 Header\nLevel 3 content"
    assert header_nodes[2].parent_id == "header2"
    assert header_nodes[2].parent_header == "Level 2 Header"
    for node in chunk_nodes:
        assert node.header == "Child Header"
        assert node.content.startswith("Child Header\n")
        assert node.type == "paragraph"
        assert node.line == 4
        assert node.meta is None
        assert node.parent_id == "header3"
        assert node.parent_header == "Level 3 Header"
        tokens = tokenizer.encode(
            node.content[len("Child Header\n"):], add_special_tokens=False).ids
        assert len(tokens) <= chunk_size - buffer

    return result_nodes


if __name__ == "__main__":
    import os
    import shutil
    from jet.file.utils import save_file

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    model: ModelType = "bert-base-uncased"
    tokenizer = get_tokenizer(model)

    deeply_nested_with_chunking_results = main_deeply_nested_with_chunking(
        tokenizer)
    save_file({"tokenizer": model, "results": deeply_nested_with_chunking_results},
              f"{output_dir}/deeply_nested_with_chunking_results.json")
