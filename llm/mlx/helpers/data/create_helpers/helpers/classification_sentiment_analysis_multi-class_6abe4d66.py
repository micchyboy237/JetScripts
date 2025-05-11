import logging
import logging.handlers
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearch
from sklearn.utils import all_estimators
from jet.llm.mlx import LLM

# Initialize the logger
logging.basicConfig(level=logging.INFO, handlers=[logging.handlers.RotatingFileHandler('sentiment_analysis.log', maxsize=100*1024, backup_count=10)])

def load_data(file_path):
    """
    Load the data from the given file path.

    Args:
    file_path (str): The path to the data file.

    Returns:
    list: A list of tuples containing the movie title and sentiment.
    """
    try:
        with open(file_path, 'r') as file:
            data = []
            for line in file:
                movie_title, sentiment = line.strip().split('\n')
                data.append((movie_title, sentiment))
        return data
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return []
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return []

def preprocess_data(data):
    """
    Preprocess the data by tokenizing, removing stop words, and lemmatizing.

    Args:
    data (list): A list of tuples containing the movie title and sentiment.

    Returns:
    list: A list of tuples containing the preprocessed movie title and sentiment.
    """
    preprocessed_data = []
    for movie_title, sentiment in data:
        # Tokenize the movie title
        tokens = word_tokenize(movie_title)
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Join the tokens back into a string
        movie_title = ' '.join(tokens)
        preprocessed_data.append((movie_title, sentiment))
    return preprocessed_data

def train_model(preprocessed_data):
    """
    Train a model using the preprocessed data.

    Args:
    preprocessed_data (list): A list of tuples containing the preprocessed movie title and sentiment.

    Returns:
    tuple: A tuple containing the trained model and the preprocessed data.
    """
    # Define the feature extraction function
    def vectorize_text(text):
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        # Fit the vectorizer to the text and transform it
        vector = vectorizer.fit_transform(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the LLM
    def vectorize_text_llm(text):
        # Create a LLM instance
        llm = LLM()
        # Fit the LLM to the text and transform it
        vector = llm.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes
    def vectorize_text_naive_bayes(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression
    def vectorize_text_logistic_regression(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest
    def vectorize_text_random_forest(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search
    def vectorize_text_grid_search(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with a single feature
    def vectorize_text_naive_bayes_single_feature(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with a single feature
    def vectorize_text_logistic_regression_single_feature(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with a single feature
    def vectorize_text_random_forest_single_feature(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with a single feature
    def vectorize_text_grid_search_single_feature(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with multiple features
    def vectorize_text_random_forest_multiple_features(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with multiple features
    def vectorize_text_grid_search_multiple_features(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with multiple features
    def vectorize_text_random_forest_multiple_features(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with multiple features
    def vectorize_text_grid_search_multiple_features(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with multiple features
    def vectorize_text_random_forest_multiple_features(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with multiple features
    def vectorize_text_grid_search_multiple_features(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with multiple features
    def vectorize_text_random_forest_multiple_features(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with multiple features
    def vectorize_text_grid_search_multiple_features(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with multiple features
    def vectorize_text_random_forest_multiple_features(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with multiple features
    def vectorize_text_grid_search_multiple_features(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with multiple features
    def vectorize_text_random_forest_multiple_features(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with multiple features
    def vectorize_text_grid_search_multiple_features(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with multiple features
    def vectorize_text_random_forest_multiple_features(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with multiple features
    def vectorize_text_grid_search_multiple_features(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with multiple features
    def vectorize_text_random_forest_multiple_features(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with multiple features
    def vectorize_text_grid_search_multiple_features(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform it
        vector = lr.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the random forest with multiple features
    def vectorize_text_random_forest_multiple_features(text):
        # Create a random forest instance
        rf = RandomForestClassifier()
        # Fit the random forest to the text and transform it
        vector = rf.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the grid search with multiple features
    def vectorize_text_grid_search_multiple_features(text):
        # Create a grid search instance
        grid = GridSearch()
        # Fit the grid search to the text and transform it
        vector = grid.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the naive Bayes with multiple features
    def vectorize_text_naive_bayes_multiple_features(text):
        # Create a naive Bayes instance
        nb = MultinomialNB()
        # Fit the naive Bayes to the text and transform it
        vector = nb.fit(text)
        # Return the vector
        return vector

    # Define the feature extraction function for the logistic regression with multiple features
    def vectorize_text_logistic_regression_multiple_features(text):
        # Create a logistic regression instance
        lr = LogisticRegression()
        # Fit the logistic regression to the text and transform