from typing import Any, List, TypedDict
from jet.data.utils import generate_key, generate_unique_hash
from jet.scrapers.utils import clean_newlines, clean_spaces
from jet.utils.text import extract_substrings, find_sentence_indexes
from jet.wordnet.lemmatizer import lemmatize_text
from llama_index.core import Document
from jet.file.utils import load_file
from jet.search.transformers import clean_string
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.vectors.reranker.bm25 import rerank_bm25
from jet.logger import logger
from jet.wordnet.sentence import group_sentences, merge_sentences, split_sentences
from jet.wordnet.words import count_words
from shared.data_types.job import JobData


class RerankResult(TypedDict):
    id: str
    text: str
    score: float
    matched: dict[str, int]
    metadata: dict[str, Any]


class QueryResult(TypedDict):
    query: str
    count: int
    matched: dict[str, int]
    data: List[RerankResult]


def preprocess_texts(texts: list[str]) -> list[str]:
    preprocessed_texts: list[str] = texts.copy()

    for idx, text in enumerate(preprocessed_texts):
        text = clean_newlines(text, max_newlines=1)
        text = clean_spaces(text)
        text = clean_string(text)
        text = " ".join(lemmatize_text(text))

        preprocessed_texts[idx] = text

    return preprocessed_texts


def split_text_by_docs(texts: list[str], max_tokens: int) -> list[Document]:
    docs: list[Document] = []

    for idx, text in enumerate(texts):
        token_count = count_words(text)
        if token_count > max_tokens:
            grouped_sentences = group_sentences(text, max_tokens)

            for sentence in grouped_sentences:
                start_idx = text.replace('\n', ' ').index(
                    sentence.replace('\n', ' '))
                end_idx = start_idx + len(sentence)

                node_id = generate_unique_hash()

                docs.append(Document(
                    node_id=node_id,
                    text=sentence,
                    metadata={
                        "data_id": idx,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                    }
                ))
        else:
            node_id = generate_unique_hash()

            docs.append(Document(
                node_id=node_id,
                text=text,
                metadata={
                    "data_id": idx,
                    "start_idx": 0,
                    "end_idx": len(text),
                }
            ))

    return docs


def search_and_rerank(texts: List[str], queries: List[str], max_tokens: int = 200) -> QueryResult:

    data = preprocess_texts(texts)

    docs = split_text_by_docs(data, max_tokens)

    doc_texts = [doc.text for doc in docs]

    queries = [
        "Season",
        "episode",
        "synopsis",
    ]

    queries = preprocess_texts(queries)

    # ids: List[str] = [doc.node_id for doc in docs]
    ids: List[str] = [str(idx) for idx, doc in enumerate(docs)]

    reranked_results = rerank_bm25(queries, doc_texts, ids)

    results = []
    for result in reranked_results["data"]:
        idx = int(result["id"])
        doc = docs[idx]
        orig_data: str = data[doc.metadata["data_id"]]

        matched = result["matched"]
        matched_sentences: dict[str, list[str]] = {
            key.lower(): [] for key in matched.keys()
        }
        for ngram, count in matched.items():
            lowered_ngram = ngram.lower()
            sentence_indexes = find_sentence_indexes(
                orig_data.lower(), lowered_ngram)
            word_sentences = extract_substrings(orig_data, sentence_indexes)
            matched_sentences[lowered_ngram] = [
                word_sentence for word_sentence in word_sentences
                if word_sentence.lower() in result["text"].lower()
            ]

        results.append({
            **result,
            "metadata": doc.metadata,
            "_matched_sentences": matched_sentences,
            "_data": orig_data,
        })

    copy_to_clipboard({
        "query": " ".join(queries),
        "count": reranked_results["count"],
        "matched": reranked_results["matched"],
        "data": results
    })

    response = QueryResult(
        query=" ".join(queries),
        count=reranked_results["count"],
        matched=reranked_results["matched"],
        data=results
    )

    return response


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    texts: list[str] = load_file(data_file)
    queries = ["Season", "episode", "synopsis"]

    search_results = search_and_rerank(texts, queries)

    copy_to_clipboard(search_results)

    for idx, result in enumerate(search_results["data"][:10]):
        logger.log(f"{idx + 1}:", result["text"]
                   [:30], colors=["WHITE", "DEBUG"])
        logger.success(f"{result['score']:.2f}")
