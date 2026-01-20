import os
import shutil
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    
def main():
    """Example usage of EmbeddingClient."""
    model = "embeddinggemma"
    
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    ]
    
    embedder = LlamacppEmbedding(model=model)
    
    query_embeddings = embedder(query, return_format="list", use_cache=True)
    save_file(query_embeddings, f"{OUTPUT_DIR}/query_embeddings.json")

    doc_embeddings = embedder(documents, return_format="list", use_cache=True)
    save_file(doc_embeddings, f"{OUTPUT_DIR}/doc_embeddings.json")

if __name__ == "__main__":
    main()