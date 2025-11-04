import shutil
from jet.libs.bertopic.examples.mock import load_sample_data
from jet.wordnet.analyzers.analyze_ngrams import analyze_ngrams
from jet.file.utils import save_file
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    # Example dataset
    texts = load_sample_data()

    # Example 1 — default 2–3 n-grams, TF-IDF threshold 0.05
    print("\n=== Example 1: Default Parameters ===")
    result = analyze_ngrams(texts, top_k_texts=10)
    for r in result:
        print(f"- {r['text']} | max_tfidf={r['max_tfidf']:.4f}")
    save_file(result, f"{OUTPUT_DIR}/example1_ngrams.json")

    # Example 2 — 1–2 n-grams with stricter filtering
    print("\n=== Example 2: Custom n-gram range (1–2) & min_tfidf=0.1 ===")
    result = analyze_ngrams(
        texts,
        min_tfidf=0.1,
        ngram_ranges=[(1, 2)],
        top_k_texts=3
    )
    for r in result:
        print(f"- {r['text']} | matched_ngrams={len(r['matched_ngrams'])}")
    save_file(result, f"{OUTPUT_DIR}/example2_ngrams.json")
