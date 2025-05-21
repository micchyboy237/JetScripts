import os
from typing import Dict
from jet.features.nltk_search import search_by_pos, get_pos_tag
from jet.file.utils import load_file, save_file

if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers: list[dict] = load_file(docs_file)
    docs = [header["text"]
            for header in headers if header["header_level"] != 1]
    query = "List trending isekai reincarnation anime this year."
    results = search_by_pos(query, docs)
    query_pos = get_pos_tag(query)
    lemma_doc_counts: Dict[str, int] = {
        pos_tag['lemma']: 0 for pos_tag in query_pos}
    for result in results:
        matched_lemmas = {pos_tag['lemma']
                          for pos_tag in result['matching_words_with_pos_and_lemma']}
        for lemma in lemma_doc_counts:
            if lemma in matched_lemmas:
                lemma_doc_counts[lemma] += 1
    print("Query POS Tags:")
    print(f"  Query: {query}")
    print("  Tags with document counts:")
    for pos_tag in query_pos:
        print(
            f"    Word: {pos_tag['word']}, POS: {pos_tag['pos']}, Lemma: {pos_tag['lemma']}, Documents: {lemma_doc_counts[pos_tag['lemma']]}")
    print()
    for result in results:
        print(f"Document {result['doc_index']}:")
        print(f"  Text: {result['text']}")
        print(
            f"  Matching words (word, POS, lemma): {[(pos_tag['word'], pos_tag['pos'], pos_tag['lemma']) for pos_tag in result['matching_words_with_pos_and_lemma']]}")
        print(f"  Match count: {result['matching_words_count']}\n")
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    total_docs = len(docs)
    save_file({
        "query_pos": [
            {
                **pos_tag,
                "document_count": lemma_doc_counts[pos_tag['lemma']],
                "document_percentage": round((lemma_doc_counts[pos_tag['lemma']] / total_docs * 100), 2) if total_docs > 0 else 0.0
            } for pos_tag in query_pos
        ],
        "documents_pos": [
            {
                "doc_index": result["doc_index"],
                "matching_words_count": result["matching_words_count"],
                "matching_words": ", ".join(list(set(item["lemma"] for item in result["matching_words_with_pos_and_lemma"]))),
                "text": result["text"],
            } for result in results
        ],
    }, f"{output_dir}/search_by_pos_results.json")
