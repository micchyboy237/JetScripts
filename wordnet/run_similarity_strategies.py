from jet.scrapers.utils import clean_newlines
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
    model_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/wordnet/generated/gensim_jet_phrase_model.pkl"
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/wordnet/generated/run_similarity_strategies"
    reset_cache = False

    data: list[JobData] = load_file(data_file)

    sentences_dict = {}
    sentences = []
    sentences_no_newline = []

    for item in data:
        sentence = "\n".join([
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

        cleaned_sentence = clean_string(sentence.lower())
        cleaned_sentence_no_newlines = clean_newlines(
            cleaned_sentence, max_newlines=0)

        sentences.append(cleaned_sentence)
        sentences_no_newline.append(" ".join(get_words(cleaned_sentence)))

    print(f"Number of sentences: {len(sentences)}")

    queries = [
        "React Native",
        "Mobile development",

        # "Web development",
        # "React.js",
        # "Node.js",
    ]

    # Phrase search

    detector = PhraseDetector(model_path, sentences, reset_cache=reset_cache)

    results_generator = detector.detect_phrases(sentences)
    for result in results_generator:
        multi_gram_phrases = " ".join(result["phrases"])
        orig_sentence = sentences_no_newline[result["index"]]
        updated_sentence = orig_sentence + " " + multi_gram_phrases

        orig_data = data[result["index"]]
        sentences_dict[updated_sentence] = orig_data
        sentences_no_newline[result["index"]] = updated_sentence

    results = detector.query(queries)
    save_file({"queries": queries, "results": results},
              f"{output_dir}/query-phrases.json")

    # Similarity search strategies

    queries = detector.extract_phrases(queries)
    # queries = detector.transform_queries(queries)

    # from gensim.test.utils import common_texts as corpus
    similarities = get_bm25_similarities(queries, sentences_no_newline)
    results = [
        {**result, "data": sentences_dict[result["text"]]}
        for result in similarities
    ]
    save_file({"queries": queries, "results": results},
              f"{output_dir}/bm25-similarities.json")

    similarities = get_cosine_similarities(queries, sentences_no_newline)
    results = [
        {**result, "data": sentences_dict[result["text"]]}
        for result in similarities
    ]
    save_file({"queries": queries, "results": results},
              f"{output_dir}/cosine-similarities.json")

    similarities = get_annoy_similarities(queries, sentences_no_newline)
    results = [
        {**result, "data": sentences_dict[result["text"]]}
        for result in similarities
    ]
    save_file({"queries": queries, "results": results},
              f"{output_dir}/annoy-similarities.json")
