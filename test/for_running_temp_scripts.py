from gensim.models import Phrases, Phraser
from gensim.corpora import Dictionary


class ExtendedDictionary:
    def __init__(self, min_count=1, bigram_threshold=1):
        """
        Initializes an extended dictionary with bigram detection.

        :param min_count: Minimum frequency for phrases to be considered.
        :param bigram_threshold: Threshold for forming bigrams.
        """
        self.min_count = min_count
        self.bigram_threshold = bigram_threshold
        self.bigram_model = None
        self.dictionary = Dictionary()

    def preprocess_texts(self, texts):
        """
        Manually merge known multi-word terms before training Phrases.

        :param texts: List of tokenized sentences.
        :return: Preprocessed texts with known phrases.
        """
        known_phrases = {
            ("react", "native"): "react_native",
            ("node", "js"): "node_js",
            ("cross", "platform"): "cross_platform"
        }

        # Replace known phrases in texts
        processed_texts = []
        for text in texts:
            new_text = []
            skip = False
            for i in range(len(text)):
                if skip:
                    skip = False
                    continue
                # Merge known phrases
                if i < len(text) - 1 and (text[i], text[i + 1]) in known_phrases:
                    new_text.append(known_phrases[(text[i], text[i + 1])])
                    skip = True  # Skip next word
                else:
                    new_text.append(text[i])
            processed_texts.append(new_text)

        return processed_texts

    def train_phrases(self, texts):
        """
        Trains the bigram model on the given texts.

        :param texts: List of tokenized sentences.
        """
        self.bigram_model = Phraser(
            Phrases(texts, min_count=self.min_count, threshold=self.bigram_threshold))

    def process_texts(self, texts):
        """
        Processes and updates the dictionary with new texts.

        :param texts: List of tokenized sentences.
        """
        # Preprocess texts to force multi-word phrases
        texts = self.preprocess_texts(texts)

        # Train bigram model if not already trained
        if not self.bigram_model:
            self.train_phrases(texts)

        # Apply bigram transformation
        processed_texts = [self.bigram_model[text] for text in texts]

        # Update the dictionary
        self.dictionary.add_documents(processed_texts)

        return processed_texts

    def get_dictionary(self):
        """
        Returns the gensim Dictionary object.
        """
        return self.dictionary

    def get_corpus(self, processed_texts):
        """
        Converts processed texts into a bag-of-words corpus.

        :param processed_texts: List of processed tokenized texts.
        :return: Corpus (list of bag-of-words representations)
        """
        return [self.dictionary.doc2bow(text) for text in processed_texts]
