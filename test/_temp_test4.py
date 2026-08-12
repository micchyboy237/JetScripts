# HtmlRAG is NOT a crawler. Demonstration of its actual role:
import requests
from htmlrag import HtmlRAG

html = requests.get("https://docs.unstructured.io/core/overview/chunking").text

rag = HtmlRAG()
pruned = rag.prune(html, max_tokens=3000)
chunks = rag.chunk(pruned, strategy="semantic")

print(f"Produced {len(chunks)} chunks from ALREADY-FETCHED page.")
print(chunks[0][:500])
# ⚠️ Zero link discovery. Zero query guidance. Zero crawling.
