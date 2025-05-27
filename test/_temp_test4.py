import os
import shutil
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from jet.features.nltk_utils import get_word_counts_lemmatized
from jet.file.utils import load_file, save_file
from jet.llm.utils.transformer_embeddings import search_docs

# Example usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    docs: List[dict] = load_file(docs_file)
    print(f"Loaded JSON data {len(docs)} from: {docs_file}")

    texts = [doc["text"] for doc in docs]
    texts_str = "\n\n".join(texts)

    embed_model = "all-MiniLM-L12-v2"
    keyword_score_threshold = 10.0

    word_counts_lemmatized_text_results = get_word_counts_lemmatized(
        texts_str, pos=["noun"], min_count=2, with_score=True
    )
    filtered_keywords = {
        k: v for k, v in word_counts_lemmatized_text_results.items() if v >= keyword_score_threshold
    }

    output_path = f"{output_dir}/word_counts_lemmatized_text.json"
    save_file(filtered_keywords, output_path)

    keywords = " ".join(filtered_keywords.keys())
    query = keywords
    top_k = None

    results = search_docs(
        query=query,
        documents=texts,
        model=embed_model,
        top_k=top_k,
    )

    for result in results:
        print(
            f"Document {result['doc_index']}: {texts[result['doc_index']]} (Score: {result['score']:.4f})"
        )

    output_path = f"{output_dir}/keyword_search_results.json"
    save_file({
        "query": query,
        "results": results
    }, output_path)
