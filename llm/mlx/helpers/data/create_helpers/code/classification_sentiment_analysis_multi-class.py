from typing import List, Dict, Optional, TypedDict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmater
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, stopwords
from nltk import PorterStemmer
from nltk import PorterStemmer
from nltk import WordNetLemmater
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import stopwords
import nltk
nltk.download('punkt')
nltk.download('averill')
nltk.download('wordnet')
nltk.download('stopwords')

class SentimentAnalysis:
    def __init__(self, text: str):
        self.text = text

    def preprocess_text(self):
        """Preprocess the text by tokenizing, stemming and lemmatizing."""
        tokens = word_tokenize(self.text)
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        tokens = [token for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        tokens = [token for token in tokens if token.isalpha()]
        self.text = ' '.join(tokens)

    def lemmatize(self):
        """Lemmatize the text to get the base form of the words."""
        lemmatizer = WordNetLemmater()
        self.text = lemmatizer.lemmatize(self.text)

    def get_sentiment(self):
        """Get the sentiment of the text by using the NLTK library."""
        self.preprocess_text()
        self.lemmatize()
        self.preprocess_text()
        self.get_sents()

    def get_sents(self):
        """Get the sentiments of the text by using the NLTK library."""
        sents = sent_tokenize(self.text)
        sentiments = []
        for sent in sents:
            if sent.startswith('I love'):
                sentiments.append('positive')
            elif sent.startswith('I hate'):
                sentiments.append('negative')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                    sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                    sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                    sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                    sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append('neutral')
            elif sent.startswith('I am'):
                sentiments.append