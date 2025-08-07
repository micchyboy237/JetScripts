import os
import shutil
from typing import List
from jet.code.markdown_types.markdown_parsed_types import HeaderSearchResult
from jet.file.utils import load_file, save_file
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.vectors.filters.mmr import select_mmr_texts
from jet.wordnet.n_grams import count_ngrams

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Real-world example: Document snippets about machine learning
texts = [
    "Machine learning is a method of data analysis that automates model building.",
    "Deep learning, a subset of machine learning, uses neural networks for complex tasks.",
    "Supervised learning involves training models on labeled datasets.",
    "Unsupervised learning finds patterns in unlabeled data using clustering.",
    "Python is a popular programming language for machine learning development.",
    "Reinforcement learning optimizes decisions through trial and error.",
    "Data preprocessing is critical for effective machine learning models."
]

if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_5/top_isekai_anime_2025/search_results.json"

    docs = load_file(docs_file)
    documents: List[HeaderSearchResult] = docs["results"][:20]
    texts = [f"{doc["header"]}\n{doc["content"]}" for doc in documents]

    # Query for embedding
    query = "Top isekai anime 2025."

    # Load SentenceTransformer model
    model = SentenceTransformerRegistry.load_model('all-MiniLM-L6-v2')

    # Generate embeddings for texts and query
    embeddings = model.encode(texts, convert_to_numpy=True)
    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    # IDs for tracking (optional, will be generated if not provided)
    ids = [f"doc_{i}" for i in range(len(texts))]

    # Run MMR with a balance of relevance and diversity (lambda=0.5)
    results = select_mmr_texts(
        embeddings=embeddings,
        texts=texts,
        query_embedding=query_embedding,
        lambda_param=0.5,
        max_texts=10,
        ids=ids
    )

    # Print results
    print("Selected diverse texts:")
    for result in results:
        print(
            f"ID: {result['id']}, Index: {result['index']}, Score: {result['score']:.4f}")
        print(f"Text: {result['text']}\n")

    all_ngrams = count_ngrams(texts, min_words=1)
    save_file(all_ngrams, f"{OUTPUT_DIR}/all_ngrams.json")

    result_ngrams = count_ngrams(
        [result["text"] for result in results], min_words=1)
    save_file(result_ngrams, f"{OUTPUT_DIR}/result_ngrams.json")

    save_file(results, f"{OUTPUT_DIR}/results.json")
