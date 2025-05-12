```python
class Chunker:
    def chunk_text(self, text, chunk_size=800, overlap=100):
        """
        Split text into overlapping chunks.

        Args:
            text (str): Input text to chunk
            chunk_size (int): Size of each chunk in characters
            overlap (int): Overlap between chunks in characters
        Returns:
            List[Dict]: List of chunk dictionaries with text and metadata
        """
        chunks = []  # Initialize an empty list to store the chunks
        # Iterate over the text with the specified chunk size and overlap
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]  # Extract a chunk of the specified size
            if chunk:  # Ensure we don't add empty chunks
                chunks.append({
                    "text": chunk,  # The chunk text
                    "chunk_id": len(chunks) + 1,  # Unique ID for the chunk
                    "start_char": i,  # Starting character index of the chunk
                    "end_char": i + len(chunk)  # Ending character index of the chunk
                })
        print(f"Created {len(chunks)} text chunks")  # Print the number of created chunks
        return chunks

    def process_document_into_propositions(self, pdf_path, chunk_size=800, chunk_overlap=100, quality_thresholds=None):
        """
        Process a document into quality-checked propositions.

        Args:
            pdf_path (str): Path to the PDF file
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Overlap between chunks in characters
            quality_thresholds (Dict): Threshold scores for proposition quality
        Returns:
            List[Dict]: List of proposition dictionaries with text and metadata
        """
        # Extract text from the PDF file
        text = self.extract_text_from_pdf(pdf_path)
        # Create chunks from the extracted text
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        # Initialize a list to store all propositions
        all_propositions = []
        print("Generating propositions from chunks...")
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            # Generate propositions for the current chunk
            chunk_propositions = self.generate_propositions(chunk)
            print(f"Generated {len(chunk_propositions)} propositions")
            # Process each generated proposition
            for prop in chunk_propositions:
                proposition_data = {
                    "text": prop,
                    "source_chunk_id": chunk["chunk_id"],
                    "source_text": chunk["text"]
                }
                all_propositions.append(proposition_data)
        # Evaluate the quality of the generated propositions
        print("Evaluating proposition quality...")
        quality_propositions = []
        for i, prop in enumerate(all_propositions):
            if i % 10 == 0:  # Status update every 10 propositions
                print(f"Evaluating proposition {i+1}/{len(all_propositions)}...")
            # Evaluate the quality of the current proposition
            scores = self.evaluate_proposition(prop["text"], prop["source_text"])
            prop["quality_scores"] = scores
            # Check if the proposition passes the quality thresholds
            passes_quality = True
            if quality_thresholds is not None