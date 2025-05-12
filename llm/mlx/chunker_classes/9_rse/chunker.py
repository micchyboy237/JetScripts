```python
class Chunker:
    def reconstruct_segments(self, chunks, best_segments):
        """
        Reconstruct text segments based on chunk indices.

        Args:
            chunks (List[str]): List of all document chunks
            best_segments (List[Tuple[int, int]]): List of (start, end) indices for segments

        Returns:
            List[str]: List of reconstructed text segments
        """
        reconstructed_segments = []  # Initialize an empty list to store the reconstructed segments
        for start, end in best_segments:
            # Join the chunks in this segment to form the complete segment text
            segment_text = " ".join(chunks[start:end])
            # Append the segment text and its range to the reconstructed_segments list
            reconstructed_segments.append({
                "text": segment_text,
                "segment_range": (start, end),
            })
        return reconstructed_segments  # Return the list of reconstructed text segments

    def chunk_text(self, text, chunk_size=800, overlap=0):
        """
        Split text into non-overlapping chunks.

        Args:
            text (str): Input text to chunk
            chunk_size (int): Size of each chunk in characters
            overlap (int): Overlap between chunks in characters

        Returns:
            Dict: Result with query, segments, and response
        """
        print("n=== STARTING RAG WITH RELEVANT SEGMENT EXTRACTION ===")
        print(f"Query: {text}")

        # Process the document to extract text, chunk it, and create embeddings
        chunks, vector_store, doc_info = self.process_document(text, chunk_size)
        # Calculate relevance scores and chunk values based on the query
        print("nCalculating relevance scores and chunk values...")
        chunk_values = self.calculate_chunk_values(text, chunks, vector_store)
        # Find the best segments of text based on chunk values
        best_segments, scores = self.find_best_segments(
            chunk_values,
            max_segment_length=20,
            total_max_length=30,
            min_segment_value=0.2
        )
        # Reconstruct text segments from the best chunks
        print("nReconstructing text segments from chunks...")
        segments = self.reconstruct_segments(chunks, best_segments)
        # Format the segments into a context string for the language model
        context = self.format_segments_for_context(segments)
        # Generate a response from the language model using the context
        response = self.generate_response(text, context)
        # Compile the result into a dictionary
        result = {
            "query": text,
            "segments": segments,
            "response": response
        }
        print("n=== FINAL RESPONSE ===")
        print(response)
        return result

    def process_document(self, text, chunk_size=800):
        """
        Process a document for use with RSE.

        Args:
            text (str): Input text to process
            chunk_size (int): Size of each chunk in characters

        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        # Simple character-based chunking
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:  # Ensure we don't add empty chunks
                chunks.append(chunk)
        return chunks

    def calculate_chunk_values(self, query, chunks, vector_store):
        """
        Calculate chunk values by combining relevance and position.

        Args:
            query (str): User query
            chunks (List[str]): List of document chunks
            vector_store (SimpleVectorStore): Vector