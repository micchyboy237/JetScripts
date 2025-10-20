from jet.libs.bertopic.examples.mock import load_sample_data
from jet.libs.bertopic.rag_bertopic import TopicRAG
from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def run_example_rag_bertopic():
    """Demonstrates RAG retrieval with varying document sets and queries."""

    docs = load_sample_data(model="embeddinggemma", chunk_size=512, truncate=True, convert_plain_text=True)
    rag = TopicRAG(verbose=True)
    rag.fit_topics(docs)
    search_results = rag.retrieve_for_query("Top isekai anime 2025")
    save_file(search_results, f"{OUTPUT_DIR}/search_results.json")

if __name__ == "__main__":
    run_example_rag_bertopic()
