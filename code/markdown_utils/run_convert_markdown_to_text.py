from typing import List
from jet.code.markdown_utils._converters import convert_markdown_to_text
from jet.logger import logger

def preprocess_for_rag(md_docs: List[str]) -> List[str]:
    return [convert_markdown_to_text(doc) for doc in md_docs]

# Test with pytest
if __name__ == "__main__":
    # Given: Sample Markdown documents
    md_docs = ["# Title\n**Bold** [link](url)", "> Quote\n- Item"]
    # When: Convert to plain text
    result = preprocess_for_rag(md_docs)
    # Then: Verify clean output for embeddings
    expected = ["Title\n\nBold link", "Quote\n\nItem"]
    assert result == expected, f"Expected {expected}, but got {result}"

    logger.success(result)