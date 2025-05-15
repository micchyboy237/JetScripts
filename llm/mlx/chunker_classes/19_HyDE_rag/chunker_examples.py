from chunker import Chunker
import fitz
import requests
import pandas as pd

class Chunker:
    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None

    def process_document(self):
        # Step 1: Extract text from the PDF file
        pages = self.extract_text_from_pdf(self.pdf_path)
        # Step 2: Create chunks
        all_chunks = self.chunk_text(pages, self.chunk_size, self.chunk_overlap)
        # Step 3: Create embeddings for the text chunks
        chunk_embeddings = self.create_embeddings(all_chunks)
        # Step 4: Create a vector store to hold the chunks and their embeddings
        self.vector_store = SimpleVectorStore()
        for i, chunk in enumerate(all_chunks):
            self.vector_store.add_item(
                text=chunk["text"],
                embedding=chunk_embeddings[i],
                metadata=chunk["metadata"]
            )
        return self.vector_store

    def extract_text_from_pdf(self, pdf_path):
        # Open the PDF file using PyMuPDF
        pdf = fitz.open(pdf_path)
        pages = []
        # Iterate over each page in the PDF
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            # Skip pages with very little text (less than 50 characters)
            if len(text.strip()) > 50:
                pages.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_path,
                        "page": page_num + 1
                    }
                })
        return pages

    def chunk_text(self, text, chunk_size, chunk_overlap):
        # Process in chunks if needed (OpenAI API limits)
        batch_size = 100
        all_chunks = []
        # Iterate over the input text in batches
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            # Create chunks for the current batch
            response = requests.get(f'https://api.openai.com/interactions?model=BAAI/bge-en-icl&input={batch}&maxOutput=100')
            batch_embeddings = [item.embedding for item in response.json()['interactions'][0]['output']['text']]
            all_chunks.extend([{"text": chunk, "metadata": {"source": pdf_path, "page": page_num + 1}} for page_num, chunk in enumerate(batch) for chunk in batch_embeddings])
        return all_chunks

    def create_embeddings(self, texts, model="BAAI/bge-en-icl"):
        # Handle empty input
        if not texts:
            return []
        # Process in batches if needed (OpenAI API limits)
        batch_size = 100
        all_embeddings = []
        # Iterate over the input texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Create embeddings for the current batch
            response = requests.get(f'https://api.openai.com/interactions?model={model}&input={batch}&maxOutput=100')
            # Extract embeddings from the response
            batch_embeddings = [item.embedding for item in response.json()['interactions'][0]['output']['text']]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def vectorize(self, texts, model="BAAI/bge-en-icl"):
        # Create a vector store to hold the chunks and their embeddings
        self.vector_store = SimpleVectorStore()
        for i, chunk in enumerate(self.vector_store.items):
            self.vector_store.add_item(
                text=chunk["text"],
                embedding=chunk["embedding"],
                metadata=chunk["metadata"]
            )
        return self.vector_store

    def main_function(self):
        chunker = Chunker('example.pdf')
        vector_store = chunker.process_document()
        chunker.vectorize([vector_store], model="BAAI/bge-en-icl")
        return vector_store

if __name__ == "__main__":
    vector_store = chunker.main_function()
    print(vector_store)