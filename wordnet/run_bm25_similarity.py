from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
from jet.file.utils import load_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from shared.data_types.job import JobData


def get_bm25_similarities(queries: list[str], corpus: list[list[str]]):
    dictionary = Dictionary(corpus)  # fit dictionary
    # enforce binary weights
    query_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
    document_model = OkapiBM25Model(dictionary=dictionary)  # fit bm25 model

    # convert corpus to BoW format
    bow_corpus = [dictionary.doc2bow(line) for line in corpus]
    bm25_corpus = document_model[bow_corpus]
    index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                   normalize_queries=False, normalize_documents=False)

    # query = 'graph trees computer'.split()  # make a query
    bow_query = dictionary.doc2bow(queries)
    bm25_query = query_model[bow_query]
    # calculate similarity of query to each doc from bow_corpus
    similarities = index[bm25_query]
    return similarities


def get_cosine_similarities(queries: list[str], corpus: list[list[str]]):
    dictionary = Dictionary(corpus)  # fit dictionary
    # convert corpus to BoW format
    bow_corpus = [dictionary.doc2bow(line) for line in corpus]
    index = SparseMatrixSimilarity(
        bow_corpus, num_docs=len(corpus), num_terms=len(dictionary))

    # query = 'graph trees computer'.split()  # make a query
    bow_query = dictionary.doc2bow(queries)
    # calculate similarity of query to each doc from bow_corpus
    similarities = index[bow_query]
    return similarities


if __name__ == '__main__':
    from gensim.test.utils import common_texts as corpus

    model_path = 'generated/gensim_jet_phrase_model.pkl'
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)

    corpus = [
        f"{item["title"]}\n{item["details"]}"
        for item in data
    ]

    queries = [
        "Web development",
        "Mobile development",
        "React.js",
        "React Native",
        "Node.js",
    ]

    similarities = get_bm25_similarities(corpus)
    logger.newline()
    logger.debug("BM25 Similarities:")
    logger.success(format_json(similarities))

    similarities = get_cosine_similarities(corpus)
    logger.newline()
    logger.debug("Cosine Similarities:")
    logger.success(format_json(similarities))
