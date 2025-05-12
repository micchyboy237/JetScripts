```python
class Chunker:
    """
    A class for chunking text into smaller segments.
    """

    def chunk_text(self, text, n, overlap):
        """
        Chunk the given text into segments of n characters with overlap.

        Args:
            text (str): The text to be chunked.
            n (int): The number of characters in each chunk.
            overlap (int): The number of overlapping characters between chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        for i in range(0, len(text), n - overlap):
            chunks.append(text[i:i + n])
        return chunks

# Example usage:
chunker = Chunker()
text = "This is a sample text that needs to be chunked."
n = 10
overlap = 2
chunks = chunker.chunk_text(text, n, overlap)
print(chunks)
```