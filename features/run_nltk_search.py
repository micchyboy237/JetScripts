# jet_python_modules/jet/run_nltk_search.py
import os
from typing import Dict
from jet.features.nltk_search import search_by_pos, get_pos_tag
from jet.file.utils import load_file, save_file

# Example usage
if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers: list[dict] = load_file(docs_file)
    docs = [header["text"] for header in headers]

    # Sample query (valid sentence)
    query = "List trending isekai reincarnation anime this year."

    # Get and print query POS tags with document counts
    query_pos = get_pos_tag(query)
    results = search_by_pos(query, docs)

    # Calculate document counts for each query lemma
    lemma_doc_counts: Dict[str, int] = {
        pos_tag['lemma']: 0 for pos_tag in query_pos}
    for result in results:
        matched_lemmas = {pos_tag['lemma']
                          for pos_tag in result['matching_words_with_pos_and_lemma']}
        for lemma in lemma_doc_counts:
            if lemma in matched_lemmas:
                lemma_doc_counts[lemma] += 1

    # Print query POS tags with document counts
    print("Query POS Tags:")
    print(f"  Query: {query}")
    print("  Tags with document counts:")
    for pos_tag in query_pos:
        print(
            f"    Word: {pos_tag['word']}, POS: {pos_tag['pos']}, Lemma: {pos_tag['lemma']}, Documents: {lemma_doc_counts[pos_tag['lemma']]}")
    print()

    # Print results
    for result in results:
        print(f"Document {result['document_index']}:")
        print(f"  Text: {result['text']}")
        print(
            f"  Matching words (word, POS, lemma): {[(pos_tag['word'], pos_tag['pos'], pos_tag['lemma']) for pos_tag in result['matching_words_with_pos_and_lemma']]}")
        print(f"  Match count: {result['matching_words_count']}\n")

    # Save results with query_pos and lemma document counts
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file({
        "query_pos": [
            {
                "word": pos_tag['word'],
                "pos": pos_tag['pos'],
                "lemma": pos_tag['lemma'],
                "document_count": lemma_doc_counts[pos_tag['lemma']]
            } for pos_tag in query_pos
        ],
        "documents_pos": results,
    }, f"{output_dir}/search_by_pos_results.json")
