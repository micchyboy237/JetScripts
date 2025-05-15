from chunker import Chunker

# Usage examples
# Chunking single text
chunker = Chunker()
text = "This is a sample text for demonstration purposes."
chunks = chunker.chunk_text(text, n=10, overlap=10)
print(chunks)

# Chunking multiple documents
pdf_paths = ["document1.pdf", "document2.pdf"]
chunker = Chunker()
for pdf_path in pdf_paths:
    chunks, store = chunker.process_document(pdf_path)
    print(f"Document {pdf_path} chunks: {chunks}, Vector Store: {store}")

# Custom chunk size
chunker = Chunker()
text = "This is a sample text for demonstration purposes."
chunk_size = 20
chunker = Chunker()
chunks = chunker.chunk_text(text, chunk_size, overlap=10)
print(chunks)

# Processing data from JSON files
import json
import pandas as pd

data = {
    "Text": ["This is a sample text for demonstration purposes."],
    "Metadata": [{"index": 0, "source": "document1.pdf"}]
}
df = pd.DataFrame(data)
chunker = Chunker()
chunks, store = chunker.process_document(df)
print(f"Document chunks: {chunks}, Vector Store: {store}")

# Chunking text from a PDF file
chunker = Chunker()
pdf_path = "document1.pdf"
text = chunker.extract_text_from_pdf(pdf_path)
chunks = chunker.chunk_text(text, n=10, overlap=10)
print(chunks)