from typing import List
from jet.code.markdown_types import HeaderDoc
from jet.logger import logger
from jet.wordnet.text_chunker import ChunkResult, chunk_texts_with_data
import openai
from keybert import KeyBERT, KeyLLM
from sentence_transformers import SentenceTransformer
from jet.file.utils import load_file, save_file
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.utils.url_utils import clean_links
import os
import shutil

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Sample document for keyword extraction
SAMPLE_DOC = """
    Supervised learning is the machine learning task of learning a function that
    maps an input to an output based on example input-output pairs. It infers a
    function from labeled training data consisting of a set of training examples.
    In supervised learning, each example is a pair consisting of an input object
    (typically a vector) and a desired output value (also called the supervisory signal).
    A supervised learning algorithm analyzes the training data and produces an inferred function,
    which can be used for mapping new examples. An optimal scenario will allow for the
    algorithm to correctly determine the class labels for unseen instances. This requires
    the learning algorithm to generalize from the training data to unseen situations in a
    'reasonable' way (see inductive bias).
"""


def extract_basic_keywords(doc: str, model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Extract basic keywords from a document using KeyBERT.

    Args:
        doc (str): The input document for keyword extraction.
        model_name (str): The name of the sentence-transformer model to use.

    Returns:
        list: A list of tuples containing keywords and their scores.
    """
    kw_model = KeyBERT(model=model_name)
    return kw_model.extract_keywords(doc)


def extract_ngram_keywords(doc: str, ngram_range: tuple = (1, 1), stop_words: str = None, model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Extract keywords or keyphrases with specified n-gram range.

    Args:
        doc (str): The input document.
        ngram_range (tuple): The range of n-grams for keyphrases (e.g., (1, 1) for single words, (1, 2) for bigrams).
        stop_words (str): Stop words to filter out (e.g., 'english' or None).
        model_name (str): The name of the sentence-transformer model.

    Returns:
        list: A list of tuples containing keywords/keyphrases and their scores.
    """
    kw_model = KeyBERT(model=model_name)
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=ngram_range, stop_words=stop_words)


def highlight_keywords(doc: str, model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Extract and highlight keywords in the document.

    Args:
        doc (str): The input document.
        model_name (str): The name of the sentence-transformer model.

    Returns:
        list: A list of tuples containing keywords and their scores (highlighting is printed).
    """
    kw_model = KeyBERT(model=model_name)
    return kw_model.extract_keywords(doc, highlight=True)


def extract_maxsum_keywords(doc: str, ngram_range: tuple = (3, 3), stop_words: str = 'english', nr_candidates: int = 20, top_n: int = 5, model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Extract diversified keywords using Max Sum Distance.

    Args:
        doc (str): The input document.
        ngram_range (tuple): The range of n-grams for keyphrases.
        stop_words (str): Stop words to filter out.
        nr_candidates (int): Number of candidates to consider.
        top_n (int): Number of top keywords to return.
        model_name (str): The name of the sentence-transformer model.

    Returns:
        list: A list of tuples containing diversified keywords and their scores.
    """
    kw_model = KeyBERT(model=model_name)
    return kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=ngram_range,
        stop_words=stop_words,
        use_maxsum=True,
        nr_candidates=nr_candidates,
        top_n=top_n
    )


def extract_mmr_keywords(doc: str, ngram_range: tuple = (3, 3), stop_words: str = 'english', diversity: float = 0.7, model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Extract diversified keywords using Maximal Marginal Relevance (MMR).

    Args:
        doc (str): The input document.
        ngram_range (tuple): The range of n-grams for keyphrases.
        stop_words (str): Stop words to filter out.
        diversity (float): Diversity parameter for MMR (0 to 1).
        model_name (str): The name of the sentence-transformer model.

    Returns:
        list: A list of tuples containing diversified keywords and their scores.
    """
    kw_model = KeyBERT(model=model_name)
    return kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=ngram_range,
        stop_words=stop_words,
        use_mmr=True,
        diversity=diversity
    )


def extract_keywords_with_flair(doc: str, model_name: str = 'roberta-base') -> list:
    """
    Extract keywords using Flair's TransformerDocumentEmbeddings.

    Args:
        doc (str): The input document.
        model_name (str): The name of the Flair transformer model.

    Returns:
        list: A list of tuples containing keywords and their scores.
    """
    from flair.embeddings import TransformerDocumentEmbeddings
    roberta = TransformerDocumentEmbeddings(model_name)
    kw_model = KeyBERT(model=roberta)
    return kw_model.extract_keywords(doc)


def extract_keywords_with_llm(doc: str, api_key: str, model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Extract keywords using a Large Language Model (OpenAI).

    Args:
        doc (str): The input document.
        api_key (str): OpenAI API key.
        model_name (str): The name of the sentence-transformer model (for fallback or compatibility).

    Returns:
        list: A list of extracted keywords.
    """
    client = openai.OpenAI(api_key=api_key)
    llm = OpenAI(client)
    kw_model = KeyLLM(llm)
    return kw_model.extract_keywords(doc)


def extract_keywords_for_similar_docs(docs: list, api_key: str, threshold: float = 0.75, model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Extract keywords for similar documents using embeddings and LLM.

    Args:
        docs (list): List of input documents.
        api_key (str): OpenAI API key.
        threshold (float): Similarity threshold for grouping documents.
        model_name (str): The name of the sentence-transformer model.

    Returns:
        list: A list of keywords for similar documents.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, convert_to_tensor=True)
    client = openai.OpenAI(api_key=api_key)
    llm = OpenAI(client)
    kw_model = KeyLLM(llm)
    return kw_model.extract_keywords(docs, embeddings=embeddings, threshold=threshold)


def load_sample_data():
    """Load sample dataset from local for topic modeling."""
    embed_model = "embeddinggemma"
    headers_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/all_headers.json"
    
    logger.info("Loading sample dataset...")
    headers_dict = load_file(headers_file)
    # headers: List[HeaderDoc] = [h for h_list in headers_dict.values() for h in h_list]
    headers: List[HeaderDoc] = headers_dict["https://gamerant.com/new-isekai-anime-2025"]
    documents = [f"{doc["header"]}\n\n{doc['content']}" for doc in headers]

    # Clean all links
    documents = [clean_markdown_links(doc) for doc in documents]
    documents = [clean_links(doc) for doc in documents]

    documents: List[ChunkResult] = chunk_texts_with_data(
        documents,
        chunk_size=64,
        chunk_overlap=32,
        model=embed_model,
    )
    save_file(documents, f"{OUTPUT_DIR}/documents.json")
    return documents


def main():
    """
    Main function to demonstrate KeyBERT usage examples.
    """
    # text = SAMPLE_DOC
    documents = load_sample_data()
    text = [doc["content"] for doc in documents]

    # Basic keyword extraction
    print("Basic Keywords:")
    print(extract_basic_keywords(text))
    save_file(extract_basic_keywords(text), f"{OUTPUT_DIR}/extract_basic_keywords.json")

    # Single-word keywords
    print("\nSingle-word Keywords:")
    print(extract_ngram_keywords(text, ngram_range=(1, 1), stop_words=None))
    save_file(extract_ngram_keywords(text, ngram_range=(1, 1), stop_words=None), f"{OUTPUT_DIR}/extract_single_word_keywords.json")

    # Bigram keyphrases
    print("\nBigram Keyphrases:")
    print(extract_ngram_keywords(text, ngram_range=(1, 2), stop_words=None))
    save_file(extract_ngram_keywords(text, ngram_range=(1, 2), stop_words=None), f"{OUTPUT_DIR}/extract_bigram_keyphrases.json")

    # Highlight keywords
    print("\nHighlighted Keywords:")
    print(highlight_keywords(text))
    save_file(highlight_keywords(text), f"{OUTPUT_DIR}/highlighted_keywords.json")

    # Max Sum Distance
    print("\nMax Sum Distance Keywords:")
    print(extract_maxsum_keywords(text, ngram_range=(3, 3)))
    save_file(extract_maxsum_keywords(text, ngram_range=(3, 3)), f"{OUTPUT_DIR}/extract_maxsum_keywords.json")

    # MMR with high diversity
    print("\nMMR Keywords (High Diversity):")
    print(extract_mmr_keywords(text, diversity=0.7))
    save_file(extract_mmr_keywords(text, diversity=0.7), f"{OUTPUT_DIR}/extract_mmr_keywords_high_diversity.json")

    # MMR with low diversity
    print("\nMMR Keywords (Low Diversity):")
    print(extract_mmr_keywords(text, diversity=0.2))
    save_file(extract_mmr_keywords(text, diversity=0.2), f"{OUTPUT_DIR}/extract_mmr_keywords_low_diversity.json")

    # Flair-based extraction
    # print("\nFlair-based Keywords:")
    # print(extract_keywords_with_flair(text))
    # save_file(extract_keywords_with_flair(text), f"{OUTPUT_DIR}/extract_keywords_with_flair.json")

    # LLM-based extraction (requires API key)
    # print("\nLLM-based Keywords:")
    # print(extract_keywords_with_llm(text, api_key="MY_API_KEY"))
    # save_file(extract_keywords_with_llm(text, api_key="MY_API_KEY"), f"{OUTPUT_DIR}/extract_keywords_with_llm.json")

    # Similar documents extraction (requires API key and multiple documents)
    # docs = [text, "Another document about machine learning..."]
    # print("\nKeywords for Similar Documents:")
    # print(extract_keywords_for_similar_docs(docs, api_key="MY_API_KEY"))
    # save_file(extract_keywords_for_similar_docs(docs, api_key="MY_API_KEY"), f"{OUTPUT_DIR}/extract_keywords_for_similar_docs.json")


if __name__ == "__main__":
    main()
