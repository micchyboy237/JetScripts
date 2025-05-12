```python
class Chunker:
    def process_document(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        """
        Process a document into a vector store.

        Args:
            pdf_path (str): Path to the PDF file
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Overlap between chunks in characters

        Returns:
            SimpleVectorStore: Vector store containing document chunks
        """
        # Extract text from the PDF file
        text = self.extract_text_from_pdf(pdf_path)
        # Split the extracted text into chunks with specified size and overlap
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        # Create embeddings for each chunk of text
        print("Creating embeddings for chunks...")
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = self.create_embeddings(chunk_texts)
        # Initialize a new vector store
        vector_store = SimpleVectorStore()
        # Add the chunks and their embeddings to the vector store
        vector_store.add_items(chunks, chunk_embeddings)
        print(f"Vector store created with {len(chunks)} chunks")
        return vector_store

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text content from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text content
        """
        print(f"Extracting text from {pdf_path}...")
        # Open the PDF file
        pdf = fitz.open(pdf_path)
        text = ""
        # Iterate through each page in the PDF
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            # Extract text from the current page and append it to the text variable
            text += page.get_text()
        return text

    def chunk_text(self, text, chunk_size, chunk_overlap):
        """
        Split the extracted text into chunks with specified size and overlap.

        Args:
            text (str): Extracted text content
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Overlap between chunks in characters

        Returns:
            List[Dict]: Chunks of text with their corresponding metadata
        """
        # Split the text into chunks
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            # Calculate the overlap between chunks
            overlap = min(chunk_overlap, len(chunk))
            chunk = chunk[:overlap]
            chunks.append({"text": chunk, "metadata": {}})
        return chunks

    def create_embeddings(self, texts, model="text-embedding-3-small"):
        """
        Create vector embeddings for text inputs using OpenAI's embedding models.

        Args:
            texts (str or List[str]): Input text(s) to be embedded. Can be a single string
                or a list of strings
            model (str): The embedding model name to use. Defaults to