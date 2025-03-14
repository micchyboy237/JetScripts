from jet.file.utils import load_file
import numpy as np
import os
import json
from gensim.corpora import Dictionary
from gensim.models import OkapiBM25Model, TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from nltk.tokenize import word_tokenize
from jet.wordnet.stopwords import StopWords
from jet.file.utils import save_data, load_data
from concurrent.futures import ThreadPoolExecutor, as_completed
from shared.data_types.job import JobData
from tqdm import tqdm
from typing import List, Tuple, Dict
from jet.logger import logger, time_it


class BM25TopicClassifier:
    def __init__(self, language='english', output_dir=None):
        self.language = language
        self.documents = []
        self.corpus = []
        self.bow_corpus = {}
        self.doc_ids = []
        self.bm25_model = None
        self.dictionary = None
        self.output_dir = output_dir
        self.document_lengths = []
        self.avgdl = 0
        self.scores_cache = {}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        stopwords = StopWords()
        if language.lower() == 'tagalog':
            self.stopwords = stopwords.tagalog_stop_words
        else:
            self.stopwords = stopwords.english_stop_words

    def preprocess(self, text: str) -> list:
        tokens = word_tokenize(text)
        return [word.lower() for word in tokens if word.lower() not in self.stopwords and word.isalpha()]

    @time_it
    def _setup_documents(self, texts: list[str] = None):
        filepath = f"{self.output_dir}/bm25_documents"
        documents_path = filepath if self.output_dir else None
        if documents_path and os.path.exists(documents_path):
            print("Loading existing texts...")
            self.documents = load_data(documents_path)
        else:
            print("Creating a new document corpus...")
            self.documents = [{"id": str(i), "content": text}
                              for i, text in enumerate(texts)]
            if self.output_dir:
                save_data(documents_path, self.documents, is_binary=True)

    @time_it
    def _setup_corpus(self):
        filepath = f"{self.output_dir}/bm25_corpus"
        corpus_path = filepath if self.output_dir else None
        if corpus_path and os.path.exists(corpus_path):
            print("Loading existing corpus...")
            self.corpus = load_data(corpus_path)
        else:
            self.corpus = [self.preprocess(doc['content'])
                           for doc in self.documents]
            if self.output_dir:
                save_data(corpus_path, self.corpus, is_binary=True)
        self.doc_ids = [str(i) for i in range(len(self.corpus))]
        self.avgdl = sum(len(doc) for doc in self.corpus) / len(self.corpus)
        self.document_lengths = [len(doc) for doc in self.corpus]

    @time_it
    def _setup_dictionary(self):
        filepath = f"{self.output_dir}/bm25_dictionary"
        dictionary_path = filepath if self.output_dir else None
        if dictionary_path and os.path.exists(dictionary_path):
            print("Loading existing dictionary...")
            self.dictionary = Dictionary.load(dictionary_path)
        else:
            print("Creating a new dictionary...")
            self.dictionary = Dictionary(self.corpus)
            if self.output_dir:
                self.dictionary.save(dictionary_path)

    @time_it
    def _setup_bow_corpus(self):
        filepath = f"{self.output_dir}/bow_corpus"
        corpus_path = filepath if self.output_dir else None
        if corpus_path and os.path.exists(corpus_path):
            print("Loading existing bm25 corpus...")
            self.bow_corpus = load_data(corpus_path)
        else:
            print("Creating a new bm25 corpus...")
            # self.bow_corpus = [self.dictionary.doc2bow(
            #     doc) for doc in self.corpus]
            self.bow_corpus = {}
            for doc in self.corpus:
                bow = self.dictionary.doc2bow(doc)
                bow_dict = dict(bow)
                self.bow_corpus.update(bow_dict)
            if self.output_dir:
                save_data(corpus_path, self.bow_corpus, is_binary=True)

    @time_it
    def _setup_bm25_model(self):
        filepath = f"{self.output_dir}/bm25_model"
        model_path = filepath if self.output_dir else None
        if model_path and os.path.exists(model_path):
            print("Loading existing bm25 model...")
            self.bm25_model = OkapiBM25Model.load(model_path)
        else:
            print("Creating a new bm25 model...")
            self.bm25_model = OkapiBM25Model(
                corpus=self.bow_corpus, dictionary=self.dictionary)
            if self.output_dir:
                self.bm25_model.save(model_path)

    def _precompute_scores(self):
        filepath = f"{self.output_dir}/bm25_scores"
        scores_path = filepath if self.output_dir else None
        if scores_path and os.path.exists(scores_path):
            print("Loading existing scores...")
            self.scores_cache = load_data(scores_path) if os.path.exists(
                scores_path) else {}
        else:
            print("Precomputing scores...")
            futures = []
            with ThreadPoolExecutor() as executor:
                for idx, doc in enumerate(tqdm(self.corpus, desc="Computing scores", unit="doc")):
                    doc_bow = self.dictionary.doc2bow(doc)
                    future = executor.submit(
                        self._calculate_scores_for_document, idx, doc_bow)
                    futures.append(future)

                for future in tqdm(as_completed(futures), total=len(futures), desc="Finalizing scores"):
                    doc_id, scores = future.result()
                    self.scores_cache[doc_id] = scores

    def _calculate_document_score(self, query_bow, doc_bow, doc_length) -> float:
        score = 0.0
        doc_dict = dict(doc_bow)
        for word_id, freq in query_bow:
            if word_id in doc_dict:
                df = sum(word_id in dict(self.dictionary.doc2bow(doc))
                         for doc in self.corpus)
                idf = np.log(1 + (len(self.corpus) - df + 0.5) / (df + 0.5))
                tf = doc_dict[word_id]
                denom = tf + self.bm25_model.k1 * \
                    (1 - self.bm25_model.b + self.bm25_model.b *
                     (doc_length / self.avgdl))
                score += freq * (idf * tf * (self.bm25_model.k1 + 1)) / denom
        return score

    def _calculate_scores_for_document(self, doc_index: int, document_bow: List[Tuple[int, int]]) -> Tuple[str, List[Dict[str, float]]]:
        scores = []
        # Pre-compute some BM25-specific constants for each document to avoid recalculating them
        idf_cache = {}
        for word_id, _ in document_bow:
            df = sum(word_id in dict(self.dictionary.doc2bow(doc))
                     for doc in self.corpus)
            idf = np.log(1 + (len(self.corpus) - df + 0.5) / (df + 0.5))
            idf_cache[word_id] = idf

        # Convert corpus to bow outside the loop to avoid repeated computation
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in self.corpus]

        for idx, doc_bow in tqdm(enumerate(corpus_bow), desc=f"Calculating scores for doc {doc_index}", unit="doc"):
            if idx != doc_index:
                score = 0.0
                doc_dict = dict(doc_bow)
                for word_id, freq in document_bow:
                    if word_id in doc_dict:
                        idf = idf_cache[word_id]
                        tf = doc_dict[word_id]
                        denom = tf + self.bm25_model.k1 * \
                            (1 - self.bm25_model.b + self.bm25_model.b *
                             (self.document_lengths[idx] / self.avgdl))
                        score += freq * \
                            (idf * tf * (self.bm25_model.k1 + 1)) / denom
                if score > 0:
                    scores.append(
                        {"doc_id": self.doc_ids[idx], "score": score})
        return scores

    def fit(self, texts: list[str] = None):
        if not self.documents:
            self._setup_documents(texts)
        if not self.corpus:
            self._setup_corpus()
        if not self.dictionary:
            self._setup_dictionary()
        if not self.bow_corpus:
            self._setup_bow_corpus()
        if not self.bm25_model:
            self._setup_bm25_model()
        if not self.scores_cache:
            scores_path = f"{self.output_dir}/bm25_scores"
        #     self._precompute_scores()
            if os.path.exists(scores_path):
                print("Loading existing scores...")
            self.scores_cache = load_data(scores_path) if os.path.exists(
                scores_path) else {}

    @time_it
    def get_topics(self, doc_id: str, top_n: int = 5) -> List[str]:
        if self.bm25_model is None:
            raise ValueError(
                "Model not fitted. Call fit() with a corpus before getting topics.")
        if doc_id not in self.doc_ids:
            raise ValueError("Document ID not found in the corpus.")

        scores = 0.0
        if doc_id in self.scores_cache:
            scores = self.scores_cache[doc_id]
        else:
            doc_index = self.doc_ids.index(doc_id)
            doc_bow = self.dictionary.doc2bow(self.corpus[doc_index])
            scores = self._calculate_scores_for_document(
                doc_index, doc_bow)
            self.scores_cache[doc_id] = scores
            if self.output_dir:
                scores_path = f"{self.output_dir}/bm25_scores"
                save_data(scores_path, self.scores_cache, is_binary=True)

        print("Length of scores: ", len(scores))
        top_indexes = sorted(
            scores, key=lambda x: x['score'], reverse=True)[:top_n]
        top_topics = []
        doc_id_content = self.documents[int(doc_id)]['content']
        logger.debug(
            f"Top {top_n} topics for document id ({doc_id}):\n{doc_id_content}")
        for idx in top_indexes:
            doc_id = idx['doc_id']
            doc = self.documents[int(doc_id)]
            content = doc['content']
            score = idx['score']
            top_topics.append({
                "doc_id": doc_id,
                "score": score,
                "content": content,
            })
        return top_topics

    @time_it
    def get_bm25_similarities(self, queries: List[str], top_n: int = 5) -> List[Dict[str, float]]:
        corpus = self.corpus
        dictionary = self.dictionary
        # Convert corpus to tuples
        bow_corpus = [dictionary.doc2bow(line) for line in corpus]
        query_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
        document_model = document_model = OkapiBM25Model(
            dictionary=dictionary)  # fit bm25 model
        bm25_corpus = document_model[bow_corpus]
        index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                       normalize_queries=False, normalize_documents=False)
        bow_query = dictionary.doc2bow(queries)

        bm25_query = query_model[bow_query]
        # calculate similarity of queries to each doc from bow_corpus
        similarities = index[bm25_query]

        logger.debug(f"BM25 Similarities of queries: {queries}")
        data_with_similarities = []
        for idx, sim in enumerate(similarities):
            if sim > 0:
                doc_id = self.doc_ids[idx]
                document = self.documents[int(doc_id)]
                data_with_similarities.append({
                    "doc_id": doc_id,
                    "score": float(sim),
                    "content": document['content'],
                })
        # Sort by similarity score in descending order
        data_with_similarities = sorted(
            data_with_similarities, key=lambda x: x['score'], reverse=True)[:top_n]

        return data_with_similarities

    @time_it
    def get_cosine_similarities(self, queries: List[str], top_n: int = 5) -> List[Dict[str, float]]:
        corpus = self.corpus
        dictionary = self.dictionary
        # convert corpus to BoW format
        bow_corpus = [dictionary.doc2bow(line) for line in corpus]
        index = SparseMatrixSimilarity(
            bow_corpus, num_docs=len(corpus), num_terms=len(dictionary))

        bow_query = dictionary.doc2bow(queries)
        # calculate similarity of queries to each doc from bow_corpus
        similarities = index[bow_query]

        logger.debug(f"Cosine Similarities of queries: {queries}")
        data_with_similarities = []
        for idx, sim in enumerate(similarities):
            if sim > 0:
                doc_id = self.doc_ids[idx]
                document = self.documents[int(doc_id)]
                data_with_similarities.append({
                    "doc_id": doc_id,
                    "score": float(sim),
                    "content": document['content'],
                })
        # Sort by similarity score in descending order
        data_with_similarities = sorted(
            data_with_similarities, key=lambda x: x['score'], reverse=True)[:top_n]

        return data_with_similarities


# Example usage
if __name__ == '__main__':
    top_n = 5
    print("Loading word dictionary...")
    model_path = 'generated/gensim_jet_phrase_model.pkl'
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)

    texts = [item["details"] for item in data]

    output_dir = 'generated/topics'
    classifier = BM25TopicClassifier('english', output_dir=output_dir)
    classifier.fit(texts)
    # classifier.fit()

    logger.info("Getting topics...")
    topics = classifier.get_topics("1", top_n=top_n)
    logger.success(json.dumps(topics, indent=2))

    queries = [
        "Web development",
        "Mobile development",
        "React.js",
        "React Native",
        "Node.js",
    ]

    logger.info(f"Cosine Similarities:")
    similarities = classifier.get_cosine_similarities(queries)
    logger.success(json.dumps(similarities, indent=2))
    print("\n")
    logger.info(f"BM25 Similarities:")
    similarities = classifier.get_bm25_similarities(queries)
    logger.success(json.dumps(similarities, indent=2))

    print("DONE!")
