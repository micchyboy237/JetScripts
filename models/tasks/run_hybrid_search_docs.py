

from jet.file.utils import load_file
from jet.logger import logger
from jet.models.tasks.hybrid_search_docs import search_docs
from jet.vectors.document_types import HeaderDocument


if __name__ == "__main__":
    # Example usage
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    query = "List all ongoing and upcoming isekai anime 2025."
    documents = [HeaderDocument(**doc) for doc in load_file(docs_file)]
    results = search_docs(
        query=query,
        documents=documents,
        task_description="Retrieve relevant anime documents",
        model="static-retrieval-mrl-en-v1",
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=512,
        ids=[],
        threshold=0.0
    )
    for result in results:
        print(f"Rank {result['rank']} (ID {result['id']}):")
        print(f"Score: {result['score']:.4f}")
        print(f"Text Preview: {result['text'][:200]}...")
        print(f"Tokens: {result['tokens']}\n")
