```python
class Chunker:
    """
    A class for chunking text data.

    Args:
        text_data (List[Dict]): Text data extracted from PDF
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters

    Returns:
        List[Dict]: Chunked text data
    """

    def chunk_text(self, text_data, chunk_size, overlap):
        """
        Split text data into overlapping chunks.

        Args:
            text_data (List[Dict]): Text data extracted from PDF
            chunk_size (int): Size of each chunk in characters
            overlap (int): Overlap between chunks in characters

        Returns:
            List[Dict]: Chunked text data
        """

        chunked_data = []  # Initialize an empty list to store chunked data
        for item in text_data:
            text = item["content"]  # Extract the text content
            metadata = item["metadata"]  # Extract the metadata

            # Skip if text is too short
            if len(text) < chunk_size / 2:
                chunked_data.append({
                    "content": text,
                    "metadata": metadata
                })
                continue

            # Create chunks with overlap
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]  # Extract a chunk of the specified size
                if chunk:  # Ensure we don't add empty chunks
                    chunks.append(chunk)

            # Add each chunk with updated metadata
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()  # Copy the original metadata
                chunk_metadata["chunk_index"] = i  # Add chunk index to metadata
                chunk_metadata["chunk_count"] = len(chunks)  # Add total chunk count to metadata
                chunked_data.append({
                    "content": chunk,  # The chunk text
                    "metadata": chunk_metadata  # The updated metadata
                })
                print(f"Created {len(chunked_data)} text chunks")  # Print the number of created chunks
        return chunked_data
```