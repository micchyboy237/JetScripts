import pytest
from typing import List, Optional
from jet.code.markdown_types import ContentType, MetaType
from jet.data.header_types import HeaderNode, TextNode, NodeType, Nodes
from jet.models.tokenizer.base import get_tokenizer
from tokenizers import Tokenizer
from _temp import split_and_merge_nodes


class TestSplitAndMergeNodes:
    @pytest.fixture
    def tokenizer(self) -> Tokenizer:
        return get_tokenizer("bert-base-uncased")

    def test_single_text_node_no_chunking(self, tokenizer: Tokenizer) -> None:
        # Given: A single text node with short content
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="Short content.",
            meta=None
        )
        expected_nodes = [
            TextNode(
                id=node.id,  # Will be different due to UUID, but we check content
                line=1,
                type="paragraph",
                header="Test Header",
                content="Test Header\nShort content.",
                meta=None
            )
        ]

        # When: We call split_and_merge_nodes with no chunking needed
        result_nodes = split_and_merge_nodes(
            docs=node,
            tokenizer=tokenizer,
            chunk_size=100,
            chunk_overlap=0
        )

        # Then: The output should contain one node with the same content
        assert len(result_nodes) == 1
        assert result_nodes[0].type == expected_nodes[0].type
        assert result_nodes[0].header == expected_nodes[0].header
        assert result_nodes[0].content == expected_nodes[0].content
        assert result_nodes[0].line == expected_nodes[0].line
        assert result_nodes[0].meta == expected_nodes[0].meta

    def test_single_text_node_with_chunking_and_overlap(self, tokenizer: Tokenizer) -> None:
        # Given: A text node with content requiring chunking
        content = "This is a long sentence. " * 20
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Long Content",
            content=content,
            meta=None
        )
        chunk_size = 50
        chunk_overlap = 10
        expected_chunk_count = 3  # Approximate, based on token count Ascending
        buffer = 5  # Buffer to account for tokenization edge cases

        # When: We call split_and_merge_nodes with chunking and overlap
        result_nodes = split_and_merge_nodes(
            docs=node,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            buffer=buffer
        )

        # Then: The output should have multiple chunks with correct headers and overlap
        assert len(result_nodes) >= expected_chunk_count
        for node in result_nodes:
            assert node.header == "Long Content"
            assert node.content.startswith("Long Content\n")
            assert node.type == "paragraph"
            assert node.line == 1
            assert node.meta is None
            tokens = tokenizer.encode(
                node.content[len("Long Content\n"):], add_special_tokens=False).ids
            assert len(tokens) <= chunk_size - buffer

    def test_header_node_with_children(self, tokenizer: Tokenizer) -> None:
        # Given: A header node with children
        header_node = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Main Header",
            content="Header content",
            level=1,
            children=[
                TextNode(
                    id="child1",
                    line=2,
                    type="paragraph",
                    header="Child Header",
                    content="Child content.",
                    meta=None,
                    parent_id="header1",
                    parent_header="Main Header"
                )
            ]
        )
        expected_nodes = [
            TextNode(
                id=header_node.id,  # Will be different due to UUID
                line=1,
                type="paragraph",
                header="Main Header",
                content="Main Header\nHeader content",
                meta=None,
                parent_id=None,
                parent_header=None
            ),
            TextNode(
                id=header_node.children[0].id,  # Will be different
                line=2,
                type="paragraph",
                header="Child Header",
                content="Child Header\nChild content.",
                meta=None,
                parent_id="header1",
                parent_header="Main Header"
            )
        ]

        # When: We call split_and_merge_nodes with a header node
        result_nodes = split_and_merge_nodes(
            docs=header_node,
            tokenizer=tokenizer,
            chunk_size=100,
            chunk_overlap=0
        )

        # Then: The output should preserve header, child structure, and parent information
        assert len(result_nodes) == 2
        for i, expected in enumerate(expected_nodes):
            assert result_nodes[i].type == expected.type
            assert result_nodes[i].header == expected.header
            assert result_nodes[i].content == expected.content
            assert result_nodes[i].line == expected.line
            assert result_nodes[i].meta == expected.meta
            assert result_nodes[i].parent_id == expected.parent_id
            assert result_nodes[i].parent_header == expected.parent_header

    def test_empty_content_node(self, tokenizer: Tokenizer) -> None:
        # Given: A node with empty content
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Empty Header",
            content="",
            meta=None
        )
        expected_nodes = [node]

        # When: We call split_and_merge_nodes with an empty node
        result_nodes = split_and_merge_nodes(
            docs=node,
            tokenizer=tokenizer,
            chunk_size=100,
            chunk_overlap=0
        )

        # Then: The output should return the node unchanged
        assert len(result_nodes) == 1
        assert result_nodes[0].type == expected_nodes[0].type
        assert result_nodes[0].header == expected_nodes[0].header
        assert result_nodes[0].content == expected_nodes[0].content
        assert result_nodes[0].line == expected_nodes[0].line
        assert result_nodes[0].meta == expected_nodes[0].meta

    def test_multiple_nodes(self, tokenizer: Tokenizer) -> None:
        # Given: Multiple nodes with a parent-child relationship
        nodes = [
            HeaderNode(
                id="header1",
                line=1,
                type="header",
                header="Header 1",
                content="Header content",
                level=1,
                children=[
                    TextNode(
                        id="child1",
                        line=2,
                        type="paragraph",
                        header="Child Header",
                        content="Child content.",
                        meta=None,
                        parent_id="header1",
                        parent_header="Header 1"
                    )
                ]
            ),
            TextNode(
                id="text1",
                line=3,
                type="paragraph",
                header="Text Header",
                content="Text content.",
                meta=None,
                parent_id=None,
                parent_header=None
            )
        ]
        expected_nodes = [
            TextNode(
                id=nodes[0].id,  # Will be different
                line=1,
                type="paragraph",
                header="Header 1",
                content="Header 1\nHeader content",
                meta=None,
                parent_id=None,
                parent_header=None
            ),
            TextNode(
                id=nodes[0].children[0].id,  # Will be different
                line=2,
                type="paragraph",
                header="Child Header",
                content="Child Header\nChild content.",
                meta=None,
                parent_id="header1",
                parent_header="Header 1"
            ),
            TextNode(
                id=nodes[1].id,  # Will be different
                line=3,
                type="paragraph",
                header="Text Header",
                content="Text Header\nText content.",
                meta=None,
                parent_id=None,
                parent_header=None
            )
        ]

        # When: We call split_and_merge_nodes with multiple nodes
        result_nodes = split_and_merge_nodes(
            docs=nodes,
            tokenizer=tokenizer,
            chunk_size=100,
            chunk_overlap=0
        )

        # Then: The output should process all nodes correctly and preserve parent information
        assert len(result_nodes) == 3
        for i, expected in enumerate(expected_nodes):
            assert result_nodes[i].type == expected.type
            assert result_nodes[i].header == expected.header
            assert result_nodes[i].content == expected.content
            assert result_nodes[i].line == expected.line
            assert result_nodes[i].meta == expected.meta
            assert result_nodes[i].parent_id == expected.parent_id
            assert result_nodes[i].parent_header == expected.parent_header

    def test_complex_nested_headers(self, tokenizer: Tokenizer) -> None:
        # Given: A complex hierarchy with unsorted header levels
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
                        header="Level 3 Header",
                        content="Level 3 content",
                        level=3,
                        children=[
                            TextNode(
                                id="child1",
                                line=3,
                                type="paragraph",
                                header="Child Paragraph",
                                content="Paragraph content.",
                                meta=None,
                                parent_id="header2",
                                parent_header="Level 3 Header"
                            )
                        ]
                    ),
                    HeaderNode(
                        id="header3",
                        line=4,
                        type="header",
                        header="Level 2 Header",
                        content="Level 2 content",
                        level=2,
                        children=[
                            TextNode(
                                id="child2",
                                line=5,
                                type="code",
                                header="Child Code",
                                content="print('Hello')",
                                meta={"language": "python"},
                                parent_id="header3",
                                parent_header="Level 2 Header"
                            )
                        ]
                    )
                ]
            )
        ]
        expected_nodes = [
            TextNode(
                id="header1",
                line=1,
                type="paragraph",
                header="Level 1 Header",
                content="Level 1 Header\nLevel 1 content",
                meta=None,
                parent_id=None,
                parent_header=None
            ),
            TextNode(
                id="header2",
                line=2,
                type="paragraph",
                header="Level 3 Header",
                content="Level 3 Header\nLevel 3 content",
                meta=None,
                parent_id="header1",
                parent_header="Level 1 Header"
            ),
            TextNode(
                id="child1",
                line=3,
                type="paragraph",
                header="Child Paragraph",
                content="Child Paragraph\nParagraph content.",
                meta=None,
                parent_id="header2",
                parent_header="Level 3 Header"
            ),
            TextNode(
                id="header3",
                line=4,
                type="paragraph",
                header="Level 2 Header",
                content="Level 2 Header\nLevel 2 content",
                meta=None,
                parent_id="header1",
                parent_header="Level 1 Header"
            ),
            TextNode(
                id="child2",
                line=5,
                type="code",
                header="Child Code",
                content="Child Code\nprint('Hello')",
                meta={"language": "python"},
                parent_id="header3",
                parent_header="Level 2 Header"
            )
        ]

        # When: We call split_and_merge_nodes with complex nested headers
        result_nodes = split_and_merge_nodes(
            docs=nodes,
            tokenizer=tokenizer,
            chunk_size=100,
            chunk_overlap=0
        )

        # Then: The output should preserve all headers, parent info, and hierarchy
        assert len(result_nodes) == len(expected_nodes)
        for i, expected in enumerate(expected_nodes):
            assert result_nodes[i].type == expected.type
            assert result_nodes[i].header == expected.header
            assert result_nodes[i].content == expected.content
            assert result_nodes[i].line == expected.line
            assert result_nodes[i].meta == expected.meta
            assert result_nodes[i].parent_id == expected.parent_id
            assert result_nodes[i].parent_header == expected.parent_header

    def test_multiple_children_mixed_types(self, tokenizer: Tokenizer) -> None:
        # Given: A header node with children of different content types
        nodes = [
            HeaderNode(
                id="header1",
                line=1,
                type="header",
                header="Main Header",
                content="Header content",
                level=1,
                children=[
                    TextNode(
                        id="child1",
                        line=2,
                        type="paragraph",
                        header="Paragraph Child",
                        content="Paragraph content.",
                        meta=None,
                        parent_id="header1",
                        parent_header="Main Header"
                    ),
                    TextNode(
                        id="child2",
                        line=3,
                        type="code",
                        header="Code Child",
                        content="def func(): pass",
                        meta={"language": "python"},
                        parent_id="header1",
                        parent_header="Main Header"
                    ),
                    TextNode(
                        id="child3",
                        line=4,
                        type="table",
                        header="Table Child",
                        content="| Col1 | Col2 |\n|------|------|\n| A    | B    |",
                        meta={"header": ["Col1", "Col2"],
                              "rows": [["A", "B"]]},
                        parent_id="header1",
                        parent_header="Main Header"
                    )
                ]
            )
        ]
        expected_nodes = [
            TextNode(
                id="header1",
                line=1,
                type="paragraph",
                header="Main Header",
                content="Main Header\nHeader content",
                meta=None,
                parent_id=None,
                parent_header=None
            ),
            TextNode(
                id="child1",
                line=2,
                type="paragraph",
                header="Paragraph Child",
                content="Paragraph Child\nParagraph content.",
                meta=None,
                parent_id="header1",
                parent_header="Main Header"
            ),
            TextNode(
                id="child2",
                line=3,
                type="code",
                header="Code Child",
                content="Code Child\ndef func(): pass",
                meta={"language": "python"},
                parent_id="header1",
                parent_header="Main Header"
            ),
            TextNode(
                id="child3",
                line=4,
                type="table",
                header="Table Child",
                content="Table Child\n| Col1 | Col2 |\n|------|------|\n| A    | B    |",
                meta={"header": ["Col1", "Col2"], "rows": [["A", "B"]]},
                parent_id="header1",
                parent_header="Main Header"
            )
        ]

        # When: We call split_and_merge_nodes with mixed-type children
        result_nodes = split_and_merge_nodes(
            docs=nodes,
            tokenizer=tokenizer,
            chunk_size=100,
            chunk_overlap=0
        )

        # Then: The output should preserve all types, headers, and parent info
        assert len(result_nodes) == len(expected_nodes)
        for i, expected in enumerate(expected_nodes):
            assert result_nodes[i].type == expected.type
            assert result_nodes[i].header == expected.header
            assert result_nodes[i].content == expected.content
            assert result_nodes[i].line == expected.line
            assert result_nodes[i].meta == expected.meta
            assert result_nodes[i].parent_id == expected.parent_id
            assert result_nodes[i].parent_header == expected.parent_header

    def test_deeply_nested_with_chunking(self, tokenizer: Tokenizer) -> None:
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

        # When: We call split_and_merge_nodes with chunking
        result_nodes = split_and_merge_nodes(
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

    def test_edge_case_empty_children_and_oversized_header(self, tokenizer: Tokenizer) -> None:
        # Given: A header with empty children and an oversized header content
        content = "This is a long header content. " * 20
        nodes = [
            HeaderNode(
                id="header1",
                line=1,
                type="header",
                header="Main Header",
                content=content,
                level=1,
                children=[
                    TextNode(
                        id="child1",
                        line=2,
                        type="paragraph",
                        header="Empty Child",
                        content="",
                        meta=None,
                        parent_id="header1",
                        parent_header="Main Header"
                    )
                ]
            )
        ]
        chunk_size = 50
        chunk_overlap = 10
        buffer = 5
        expected_chunk_count = 3  # Approximate, based on token count

        # When: We call split_and_merge_nodes with chunking
        result_nodes = split_and_merge_nodes(
            docs=nodes,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            buffer=buffer
        )

        # Then: The output should handle empty children and chunk the header
        assert len(result_nodes) >= expected_chunk_count + \
            1  # Chunks + empty child
        header_nodes = [n for n in result_nodes if n.line == 1]
        empty_child = [n for n in result_nodes if n.line == 2]
        assert len(header_nodes) >= expected_chunk_count
        assert len(empty_child) == 1
        assert empty_child[0].header == "Empty Child"
        assert empty_child[0].content == ""
        assert empty_child[0].parent_id == "header1"
        assert empty_child[0].parent_header == "Main Header"
        for node in header_nodes:
            assert node.header == "Main Header"
            assert node.content.startswith("Main Header\n")
            assert node.type == "paragraph"
            assert node.line == 1
            assert node.meta is None
            assert node.parent_id is None
            assert node.parent_header is None
            tokens = tokenizer.encode(
                node.content[len("Main Header\n"):], add_special_tokens=False).ids
            assert len(tokens) <= chunk_size - buffer
