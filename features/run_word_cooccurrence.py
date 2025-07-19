import os
import shutil
from typing import List

from jet.file.utils import load_file, save_file
from jet.models.embeddings.chunking import DocChunkResult
from jet.wordnet.word_cooccurrence import find_cooccurring_words


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/vectors/semantic_search/generated/run_semantic_search/react_native/search_results.json"
    chunks: List[DocChunkResult] = load_file(docs_file)["results"][:50]

    texts = [f"{doc['header']}\n{doc['content']}" for doc in chunks]

    results = find_cooccurring_words(texts, min_docs=2, ngram_range=(1, 2))
    save_file(results, f"{output_dir}/results.json")
