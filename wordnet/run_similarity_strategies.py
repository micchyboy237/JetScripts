from jet.search.transformers import clean_string
import numpy as np
from gensim.similarities.annoy import AnnoyIndexer
from gensim.models import TfidfModel
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.models import Word2Vec
from typing import TypedDict
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.gensim_scripts.phrase_detector import PhraseDetector
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.wordnet.words import get_words
from jet.search.similarity import get_bm25_similarities, get_cosine_similarities, get_annoy_similarities
from shared.data_types.job import JobData


if __name__ == '__main__':
    model_path = 'generated/gensim_jet_phrase_model.pkl'
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/wordnet/generated/run_similarity_strategies"

    data: list[JobData] = load_file(data_file)

    sentences_dict = {}

    for item in data:
        key = "\n".join([
            item["title"],
            item["details"],
            "\n".join([
                f"Tech: {tech}"
                for tech in sorted(
                    item["entities"]["technology_stack"],
                    key=str.lower
                )
            ]),
            "\n".join([
                f"Tag: {tech}"
                for tech in sorted(
                    item["tags"],
                    key=str.lower
                )
            ]),
        ])

        cleaned_key = clean_string(key.lower())
        cleaned_key = " ".join(get_words(key))
        sentences_dict[cleaned_key] = item

    sentences = list(sentences_dict.keys())
    print(f"Number of sentences: {len(sentences)}")

    model_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/wordnet/generated/gensim_jet_phrase_model.pkl"

    # Phrase search

    detector = PhraseDetector(model_path, sentences)
    phrase_grams = detector.get_phrase_grams()

    queries = [
        "React Native",
        "Mobile development",

        # "Web development",
        # "React.js",
        # "Node.js",
    ]

    results = detector.detect_phrases(queries)
    results = detector.query(queries)

    save_file({"queries": queries, "results": results},
              f"{output_dir}/query-phrases.json")

    # Similarity search strategies

    # from gensim.test.utils import common_texts as corpus
    corpus = [
        [
            word.lower()
            for word in get_words(
                "\n".join([
                    item["title"],
                    item["details"],
                    "\n".join(item["entities"]["technology_stack"]),
                    "\n".join(item["tags"]),
                ])
            )
        ]
        for item in data
    ]

    similarities = get_bm25_similarities(queries, sentences)
    results = [
        {"score": result["score"], **sentences_dict[result["text"]]}
        for result in similarities
    ]
    save_file({"queries": queries, "results": results},
              f"{output_dir}/bm25-similarities.json")

    similarities = get_cosine_similarities(queries, sentences)
    results = [
        {"score": result["score"], **sentences_dict[result["text"]]}
        for result in similarities
    ]
    save_file({"queries": queries, "results": results},
              f"{output_dir}/cosine-similarities.json")

    similarities = get_annoy_similarities(queries, sentences)
    results = [
        {"score": result["score"], **sentences_dict[result["text"]]}
        for result in similarities
    ]
    save_file({"queries": queries, "results": results},
              f"{output_dir}/annoy-similarities.json")
