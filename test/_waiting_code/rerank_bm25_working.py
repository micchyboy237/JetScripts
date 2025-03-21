from typing import List
from jet.data.utils import generate_key, generate_unique_hash
from jet.scrapers.utils import clean_newlines, clean_spaces
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


if __name__ == "__main__":

    # data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    # data = load_file(data_file)
    # docs: List[str] = [
    #     clean_string("\n".join([
    #         item["title"],
    #         item["details"],
    #         "\n".join([f"Tech: {tech}" for tech in sorted(
    #             item["entities"]["technology_stack"], key=str.lower)]),
    #         "\n".join([f"Tag: {tech}" for tech in sorted(
    #             item["tags"], key=str.lower)]),
    #     ]).lower())
    #     for item in data
    # ]
    # queries = [
    #     "React.js",
    #     "Web",
    #     "Node.js",
    #     "Firebase",
    #     "AWS"
    # ]

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    data: list[str] = load_file(data_file)

    # Clean data
    for idx, text in enumerate(data):
        text = clean_newlines(text, max_newlines=1)
        text = clean_spaces(text)
        text = clean_string(text)

        data[idx] = text

    # Split texts by max tokens
    max_tokens = 200
    docs: list[Document] = []

    for idx, text in enumerate(data):
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

    doc_texts = [doc.text for doc in docs]

    queries = [
        "Season",
        "episode",
        "synopsis",
    ]

    # ids: List[str] = [doc.node_id for doc in docs]
    ids: List[str] = [str(idx) for idx, doc in enumerate(docs)]

    reranked_results = rerank_bm25(queries, doc_texts, ids)

    results = []
    for result in reranked_results["data"]:
        idx = int(result["id"])
        doc = docs[idx]
        results.append({
            **result,
            "metadata": doc.metadata,
            "_data": data[doc.metadata["data_id"]],
        })

    copy_to_clipboard({
        "query": " ".join(queries),
        "results": results
    })

    for idx, result in enumerate(results[:10]):
        logger.log(f"{idx + 1}:", result["text"]
                   [:30], colors=["WHITE", "DEBUG"])
        logger.success(f"{result['score']:.2f}")
