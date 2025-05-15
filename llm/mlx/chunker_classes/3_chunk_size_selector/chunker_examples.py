from chunker import Chunker
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import json
import os


# Example usage in the RAG pipeline
if __name__ == "__main__":
    # Initialize chunker
    chunker = Chunker(chunk_size=256, overlap=50, strategy="character")

    # Extract text from PDF
    pdf_path = "data/AI_Information.pdf"
    extracted_text = chunker.extract_text_from_pdf(pdf_path)
    print(f"Extracted text (first 500 chars): {extracted_text[:500]}")

    # Chunk text with single chunk size
    chunks = chunker.chunk_text(extracted_text)
    print(f"Number of chunks (size 256): {len(chunks)}")

    # Chunk text with multiple chunk sizes
    chunk_sizes = [128, 256, 512]
    text_chunks_dict = chunker.chunk_multiple_sizes(
        extracted_text, chunk_sizes)
    for size, chunks in text_chunks_dict.items():
        print(f"Chunk Size: {size}, Number of Chunks: {len(chunks)}")
