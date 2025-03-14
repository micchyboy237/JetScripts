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
from jet.file.utils import load_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.wordnet.words import get_words
from shared.data_types.job import JobData


class SimilarityResult(TypedDict):
    text: str
    score: float


def get_bm25_similarities(queries: list[str], corpus: list[list[str]]) -> list[SimilarityResult]:
    dictionary = Dictionary(corpus)
    query_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
    document_model = OkapiBM25Model(dictionary=dictionary)

    bow_corpus = [dictionary.doc2bow(line) for line in corpus]
    bm25_corpus = document_model[bow_corpus]
    index = SparseMatrixSimilarity(
        bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
        normalize_queries=False, normalize_documents=False
    )

    bow_query = dictionary.doc2bow(queries)
    bm25_query = query_model[bow_query]
    similarities = index[bm25_query]

    results: list[SimilarityResult] = sorted(
        [{"text": " ".join(corpus[i]), "score": float(score)}
         for i, score in enumerate(similarities)],
        key=lambda x: x["score"], reverse=True
    )

    return results


def get_annoy_similarities(queries: list[str], corpus: list[list[str]]) -> list[SimilarityResult]:
    # Train Word2Vec model on the corpus
    model = Word2Vec(sentences=corpus, vector_size=100,
                     window=5, min_count=1, workers=4)

    dictionary = Dictionary(corpus)
    tfidf = TfidfModel(dictionary=dictionary)

    # # Extract word vectors for Annoy indexing
    # word_vectors = {word: model.wv[word]
    #                 for word in dictionary.token2id if word in model.wv}

    # Use Annoy for fast similarity lookups
    indexer = AnnoyIndexer(model.wv, num_trees=2)
    termsim_index = WordEmbeddingSimilarityIndex(
        model.wv, kwargs={'indexer': indexer})
    similarity_matrix = SparseTermSimilarityMatrix(
        termsim_index, dictionary, tfidf)

    tfidf_corpus = [tfidf[dictionary.doc2bow(doc)] for doc in corpus]
    docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix)

    results = []
    for query in queries:
        bow_query = dictionary.doc2bow(query.split())
        sims = docsim_index[bow_query]

        ranked_results = sorted(
            [{"text": " ".join(corpus[i]), "score": float(score)}
             for i, score in enumerate(sims)],
            key=lambda x: x["score"], reverse=True
        )
        results.extend(ranked_results)

    return results


def get_cosine_similarities(queries: list[str], corpus: list[list[str]]) -> list[SimilarityResult]:
    dictionary = Dictionary(corpus)
    bow_corpus = [dictionary.doc2bow(line) for line in corpus]
    index = SparseMatrixSimilarity(
        bow_corpus, num_docs=len(corpus), num_terms=len(dictionary)
    )

    bow_query = dictionary.doc2bow(queries)
    similarities = index[bow_query]

    results: list[SimilarityResult] = sorted(
        [{"text": " ".join(corpus[i]), "score": float(score)}
         for i, score in enumerate(similarities)],
        key=lambda x: x["score"], reverse=True
    )

    return results


if __name__ == '__main__':
    model_path = 'generated/gensim_jet_phrase_model.pkl'
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
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

    detector = PhraseDetector(model_path)
    phrase_grams = detector.get_phrase_grams()

    queries = [
        # "Mobile development",
        # "Web development",
        # "React Native",
        # "React.js",
        # "Node.js",
        "react_native",
        "react_developer",
        "react.js",
        "react",
        "mobile",
        "node",
        "web",
    ]

    results_dict = {query: [] for query in queries}
    for phrase, score in phrase_grams.items():
        for query in queries:
            if query in phrase:
                results_dict[query].append({
                    "phrase": phrase,
                    "score": score,
                })

    copy_to_clipboard(results_dict)
    logger.newline()
    logger.success(format_json(results_dict))
    logger.debug(f"Phrase grams: {len(phrase_grams)}")

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
    logger.newline()
    logger.debug("BM25 Similarities:")
    logger.success(format_json(similarities))

    similarities = get_cosine_similarities(queries, corpus)
    logger.newline()
    logger.debug("Cosine Similarities:")
    logger.success(format_json(similarities))

    similarities = get_annoy_similarities(queries, corpus)
    logger.newline()
    logger.debug("Annoy Similarities:")
    logger.success(format_json(similarities))
