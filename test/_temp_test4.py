import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk import ngrams

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def get_mixed_ngrams(documents: list[str], n: int = 2, formats: list[str] = None, keep_numbers: bool = False, keep_dates: bool = False) -> dict[str, dict[str, int]]:
    """
    Extract mixed word and POS n-grams in multiple formats, optionally retaining numbers or 'release date' phrases.

    Args:
        documents (list[str]): List of input document texts.
        n (int): Size of n-grams.
        formats (list[str]): List of n-gram formats ('word/POS', 'POS-word', 'word-POS', 'word-word', 'POS-POS').
        keep_numbers (bool): If True, treat numbers as tokens (e.g., '12' as 'number/NN').
        keep_dates (bool): If True, treat 'release date: <date>' as a single token 'release_date/NN'.

    Returns:
        dict[str, dict[str, int]]: Dictionary mapping format to n-gram counts, sorted by frequency.
    """
    if formats is None:
        formats = ['word/POS']
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    all_ngrams = {fmt: [] for fmt in formats}

    for doc in documents:
        # Replace 'release date: <date>' with a single token if keep_dates is True
        if keep_dates:
            doc = re.sub(
                r'release date: \d{4}-\d{2}-\d{2}', 'release_date', doc)
        tokens = word_tokenize(doc.lower())

        # Filter tokens, optionally keeping numbers
        filtered_tokens = []
        for token in tokens:
            if token.isalpha() and token not in stop_words:
                filtered_tokens.append(token)
            elif keep_numbers and re.match(r'^\d+$', token):
                filtered_tokens.append('number')

        pos_tags = nltk.pos_tag(filtered_tokens)
        # Adjust POS for 'number' and 'release_date'
        lemmatized_pairs = []
        for word, pos in pos_tags:
            if word == 'number' or word == 'release_date':
                lemmatized_pairs.append((word, 'NN'))  # Treat as noun
            else:
                lemmatized_pairs.append((lemmatizer.lemmatize(word), pos))

        for ngram in ngrams(lemmatized_pairs, n):
            words = [word for word, _ in ngram]
            pos = [pos for _, pos in ngram]
            for fmt in formats:
                if fmt == 'word/POS':
                    ngram_str = "-".join(f"{word}/{pos}" for word,
                                         pos in ngram)
                elif fmt == 'POS-word':
                    ngram_str = "-".join(f"{pos}-{word}" for word,
                                         pos in zip(words, pos))
                elif fmt == 'word-POS':
                    ngram_str = "-".join(f"{word}-{pos}" for word,
                                         pos in zip(words, pos))
                elif fmt == 'word-word':
                    ngram_str = "-".join(words)
                elif fmt == 'POS-POS':
                    ngram_str = "-".join(pos)
                all_ngrams[fmt].append(ngram_str)

    return {fmt: dict(sorted(Counter(ngrams).items(), key=lambda x: x[1], reverse=True))
            for fmt, ngrams in all_ngrams.items()}


def search_documents_by_ngrams(documents: list[str], target_ngram: str, n: int = 2, format: str = 'word/POS', keep_numbers: bool = False, keep_dates: bool = False) -> list[str]:
    """
    Search for documents containing a specific n-gram in the given format.

    Args:
        documents (list[str]): List of input document texts.
        target_ngram (str): Target n-gram (e.g., 'eps/NNS-number/NN').
        n (int): Size of n-grams.
        format (str): N-gram format ('word/POS', 'POS-word', 'word-POS', 'word-word', 'POS-POS').
        keep_numbers (bool): If True, treat numbers as tokens.
        keep_dates (bool): If True, treat 'release date: <date>' as 'release_date/NN'.

    Returns:
        list[str]: List of documents containing the target n-gram.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    matching_docs = []

    for doc in documents:
        if keep_dates:
            doc = re.sub(
                r'release date: \d{4}-\d{2}-\d{2}', 'release_date', doc)
        tokens = word_tokenize(doc.lower())
        filtered_tokens = []
        for token in tokens:
            if token.isalpha() and token not in stop_words:
                filtered_tokens.append(token)
            elif keep_numbers and re.match(r'^\d+$', token):
                filtered_tokens.append('number')

        pos_tags = nltk.pos_tag(filtered_tokens)
        lemmatized_pairs = []
        for word, pos in pos_tags:
            if word == 'number' or word == 'release_date':
                lemmatized_pairs.append((word, 'NN'))
            else:
                lemmatized_pairs.append((lemmatizer.lemmatize(word), pos))

        for ngram in ngrams(lemmatized_pairs, n):
            words = [word for word, _ in ngram]
            pos = [pos for _, pos in ngram]
            if format == 'word/POS':
                ngram_str = "-".join(f"{word}/{pos}" for word, pos in ngram)
            elif format == 'POS-word':
                ngram_str = "-".join(f"{pos}-{word}" for word,
                                     pos in zip(words, pos))
            elif format == 'word-POS':
                ngram_str = "-".join(f"{word}-{pos}" for word,
                                     pos in zip(words, pos))
            elif format == 'word-word':
                ngram_str = "-".join(words)
            elif format == 'POS-POS':
                ngram_str = "-".join(pos)
            if ngram_str == target_ngram:
                matching_docs.append(doc)
                break

    return matching_docs


# Example usage
if __name__ == "__main__":
    from jet.file.utils import load_file

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    headers: list[dict] = load_file(docs_file)
    documents = [header["text"] for header in headers]

    # Extract common mixed bigrams
    mixed_ngrams = get_mixed_ngrams(documents, n=2)
    print("Common mixed word/POS bigrams:", mixed_ngrams)

    # Search for documents with a specific mixed n-gram
    target_ngram = "cat/NN-run/VBZ"
    matching_docs = search_documents_by_ngrams(documents, target_ngram)
    print("\nDocuments with n-gram 'cat/NN-run/VBZ':")
    for doc in matching_docs:
        print(f"- {doc}")
