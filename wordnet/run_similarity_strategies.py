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

    sentences = [
        "\n".join([
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
        for item in data
    ]
    print(f"Number of sentences: {len(sentences)}")

    model_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/wordnet/generated/gensim_jet_phrase_model.pkl"

    # Phrase search

    detector = PhraseDetector(model_path, sentences)
    phrase_grams = detector.get_phrase_grams()

    queries = [
        "Web development",
        "React.js",
        # "Mobile development",
        # "React Native",
        "Node.js",

        # "react_native",
        # "react_developer",
        # "react.js",
        # "react",
        # "mobile",
        # "node",
        # "web",
    ]

    results = detector.query(queries)

    save_file(results, f"{output_dir}/query-phrases.json")

    # Similarity search strategies

    # from gensim.test.utils import common_texts as corpus
    corpus = [
        [
            word
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

    similarities = get_bm25_similarities(queries, corpus)
    save_file(similarities, f"{output_dir}/bm25-similarities.json")

    similarities = get_cosine_similarities(queries, corpus)
    save_file(similarities, f"{output_dir}/cosine-similarities.json")

    similarities = get_annoy_similarities(queries, corpus)
    save_file(similarities, f"{output_dir}/annoy-similarities.json")
