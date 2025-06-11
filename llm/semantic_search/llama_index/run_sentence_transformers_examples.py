class SentenceTransformerExamples:
    def embedding_generation(self):
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = self.embedding_model.encode(
            ["This is a sentence.", "This is another sentence."])
        return embeddings

    def semantic_textual_similarity(self):
        import numpy as np
        embeddings = self.embedding_generation()
        cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return cosine_similarity

    def text_classification(self):
        from sentence_transformers import SentenceTransformer
        self.classification_model = SentenceTransformer(
            'distilbert-base-nli-stsb-mean-tokens')
        sentences = ["I love this product!", "I hate the customer service."]
        labels = self.classification_model.predict(sentences)
        return labels

    def question_answering(self, context, question):
        from sentence_transformers import CrossEncoder
        self.re_ranking_model = CrossEncoder('ms-marco-MiniLM-L-6-v2')
        # Note: This method may not exist in the CrossEncoder class
        answer = self.re_ranking_model.answer_question(context, question)
        return answer

    def text_clustering(self):
        from sklearn.cluster import KMeans
        from sentence_transformers import SentenceTransformer
        self.clustering_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = self.clustering_model.encode(
            ["Article 1 text", "Article 2 text", "Article 3 text"])
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(embeddings)
        return kmeans.labels_

    def sentence_pair_classification(self):
        from sentence_transformers import CrossEncoder
        self.re_ranking_model = CrossEncoder('ms-marco-MiniLM-L-6-v2')
        score = self.re_ranking_model.predict(
            [["Is it raining?", "Will it rain today?"]])
        return score

    def re_ranking(self, queries, candidate_docs):
        from sentence_transformers import CrossEncoder
        self.re_ranking_model = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L6-v2', max_length=512)

        ranked_docs = self.re_ranking_model.predict(
            [(query, doc) for query in queries for doc in candidate_docs])
        result_tuples = [(doc, score) for doc, score in zip(
            candidate_docs * len(queries), ranked_docs)]

        return result_tuples


# Usage examples
if __name__ == "__main__":
    examples = SentenceTransformerExamples()

    # # Example usage of embedding_generation
    # embeddings = examples.embedding_generation()
    # print("Embeddings:", embeddings)

    # # Example usage of semantic_textual_similarity
    # similarity = examples.semantic_textual_similarity()
    # print("Semantic Textual Similarity:", similarity)

    # # Example usage of text_classification
    # labels = examples.text_classification()
    # print("Text Classification Labels:", labels)

    # # Example usage of question_answering
    # context = "The capital of France is Paris."
    # question = "What is the capital of France?"
    # answer = examples.question_answering(context, question)
    # print("Question Answering:", answer)

    # # Example usage of text_clustering
    # clusters = examples.text_clustering()
    # print("Text Clustering Labels:", clusters)

    # # Example usage of sentence_pair_classification
    # score = examples.sentence_pair_classification()
    # print("Sentence Pair Classification Score:", score)

    # Example usage of re_ranking
    queries = ["What is the capital of France?"]
    candidate_docs = ["Paris is the capital of France.",
                      "Berlin is the capital of Germany."]
    ranked_docs = examples.re_ranking(queries, candidate_docs)
    print("Re-ranked Documents:", ranked_docs)
