```python
class Chunker:
    def chunk_text(self, text, n, overlap):
        """
        Chunks the given text into segments of n characters with overlap.
        
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

    def process_document(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        """
        Process a document for RAG.
        
        Args:
            pdf_path (str): Path to the PDF file.
            chunk_size (int): Size of each chunk in characters. Defaults to 1000.
            chunk_overlap (int): Overlap between chunks in characters. Defaults to 200.
        
        Returns:
            SimpleVectorStore: A vector store containing document chunks and their embeddings.
        """
        extracted_text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(extracted_text, chunk_size, chunk_overlap)
        chunk_embeddings = self.create_embeddings(chunks)
        store = SimpleVectorStore()
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            store.add_item(
                text=chunk,
                embedding=embedding,
                metadata={"index": i, "source": pdf_path}
            )
        return store

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file and prints the first `num_chars` characters.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            str: Extracted text from the PDF.
        """
        mypdf = fitz.open(pdf_path)
        all_text = ""
        for page_num in range(mypdf.page_count):
            page = mypdf[page_num]
            text = page.get_text("text")
            all_text += text
        return all_text

    def create_embeddings(self, text, model="BAAI/bge-en-icl"):
        """
        Creates embeddings for the given text using the specified OpenAI model.
        
        Args:
            text (str): The input text for which embeddings are to be created.
            model (str): The model to be used for creating embeddings. Defaults to "BAAI/bge-en-icl".
        
        Returns:
            List[float]: The embedding vector.
        """
        # Handle both string and list inputs by converting string input to a list
        input_text = text if isinstance(text, list) else [text]
        # Create embeddings for the input text using the specified model
        response = client.embeddings.create(
            model=model,
            input=input_text
        )
        # If input was a string, return just the first embedding
        if isinstance(text, str):
            return response.data[0].embedding
        # Otherwise, return all embeddings as a list of vectors
        return [item.embedding for item in response.data]

    def similarity_search(self, query_embedding, k=5):
        """
        Find the most similar items to a query embedding.
        
        Args:
            query_embedding (List[float]): Query embedding vector.
            k (int): Number of results to return.
        
        Returns:
           