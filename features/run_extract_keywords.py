from typing import List, TypedDict
from jet.code.markdown_types import HeaderDoc
from jet.logger import logger
from jet.wordnet.text_chunker import ChunkResult, chunk_texts_with_data
import openai
from jet.adapters.keybert import KeyBERT, KeyLLM
from sentence_transformers import SentenceTransformer
from jet.file.utils import load_file, save_file
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.utils.url_utils import clean_links
import os
import shutil

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class KeywordEntry(TypedDict):
    keyword: str
    score: float

class KeywordResult(TypedDict):
    doc_index: int
    chunk_index: int
    num_tokens: int
    content: str
    keywords: List[KeywordEntry]

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

def extract_basic_keywords(docs: List[ChunkResult], model_name: str = "google/embeddinggemma-300m") -> List[KeywordResult]:
    kw_model = KeyBERT(model=model_name)
    results = []
    texts = [doc["content"] for doc in docs]
    keywords = kw_model.extract_keywords(texts)
    for doc, kws in zip(docs, keywords):
        results.append({
            "doc_index": doc["doc_index"],
            "chunk_index": doc["chunk_index"],
            "num_tokens": doc["num_tokens"],
            "content": doc["content"],
            "keywords": [{"keyword": kw[0], "score": kw[1]} for kw in kws]
        })
    return results

def extract_ngram_keywords(docs: List[ChunkResult], ngram_range: tuple = (1, 1), stop_words: str = None, model_name: str = "google/embeddinggemma-300m") -> List[KeywordResult]:
    kw_model = KeyBERT(model=model_name)
    results = []
    texts = [doc["content"] for doc in docs]
    keywords = kw_model.extract_keywords(texts, keyphrase_ngram_range=ngram_range, stop_words=stop_words)
    for doc, kws in zip(docs, keywords):
        results.append({
            "doc_index": doc["doc_index"],
            "chunk_index": doc["chunk_index"],
            "num_tokens": doc["num_tokens"],
            "content": doc["content"],
            "keywords": [{"keyword": kw[0], "score": kw[1]} for kw in kws]
        })
    return results

def highlight_keywords(docs: List[ChunkResult], model_name: str = "google/embeddinggemma-300m") -> List[KeywordResult]:
    kw_model = KeyBERT(model=model_name)
    results = []
    texts = [doc["content"] for doc in docs]
    keywords = kw_model.extract_keywords(texts, highlight=True)
    for doc, kws in zip(docs, keywords):
        results.append({
            "doc_index": doc["doc_index"],
            "chunk_index": doc["chunk_index"],
            "num_tokens": doc["num_tokens"],
            "content": doc["content"],
            "keywords": [{"keyword": kw[0], "score": kw[1]} for kw in kws]
        })
    return results

def extract_maxsum_keywords(docs: List[ChunkResult], ngram_range: tuple = (3, 3), stop_words: str = 'english', nr_candidates: int = 20, top_n: int = 5, model_name: str = "google/embeddinggemma-300m") -> List[KeywordResult]:
    kw_model = KeyBERT(model=model_name)
    results = []
    texts = [doc["content"] for doc in docs]
    keywords = kw_model.extract_keywords(
        texts,
        keyphrase_ngram_range=ngram_range,
        stop_words=stop_words,
        use_maxsum=True,
        nr_candidates=nr_candidates,
        top_n=top_n
    )
    for doc, kws in zip(docs, keywords):
        results.append({
            "doc_index": doc["doc_index"],
            "chunk_index": doc["chunk_index"],
            "num_tokens": doc["num_tokens"],
            "content": doc["content"],
            "keywords": [{"keyword": kw[0], "score": kw[1]} for kw in kws]
        })
    return results

def extract_mmr_keywords(docs: List[ChunkResult], ngram_range: tuple = (3, 3), stop_words: str = 'english', diversity: float = 0.7, model_name: str = "google/embeddinggemma-300m") -> List[KeywordResult]:
    kw_model = KeyBERT(model=model_name)
    results = []
    texts = [doc["content"] for doc in docs]
    keywords = kw_model.extract_keywords(
        texts,
        keyphrase_ngram_range=ngram_range,
        stop_words=stop_words,
        use_mmr=True,
        diversity=diversity
    )
    for doc, kws in zip(docs, keywords):
        results.append({
            "doc_index": doc["doc_index"],
            "chunk_index": doc["chunk_index"],
            "num_tokens": doc["num_tokens"],
            "content": doc["content"],
            "keywords": [{"keyword": kw[0], "score": kw[1]} for kw in kws]
        })
    return results

def extract_keywords_with_flair(docs: List[ChunkResult], model_name: str = 'roberta-base') -> List[KeywordResult]:
    from flair.embeddings import TransformerDocumentEmbeddings
    roberta = TransformerDocumentEmbeddings(model_name)
    kw_model = KeyBERT(model=roberta)
    results = []
    texts = [doc["content"] for doc in docs]
    keywords = kw_model.extract_keywords(texts)
    for doc, kws in zip(docs, keywords):
        results.append({
            "doc_index": doc["doc_index"],
            "chunk_index": doc["chunk_index"],
            "num_tokens": doc["num_tokens"],
            "content": doc["content"],
            "keywords": [{"keyword": kw[0], "score": kw[1]} for kw in kws]
        })
    return results

def extract_keywords_with_llm(docs: List[ChunkResult], api_key: str, model_name: str = "google/embeddinggemma-300m") -> List[KeywordResult]:
    client = openai.OpenAI(api_key=api_key)
    llm = OpenAI(client)
    kw_model = KeyLLM(llm)
    results = []
    texts = [doc["content"] for doc in docs]
    keywords = kw_model.extract_keywords(texts)
    for doc, kws in zip(docs, keywords):
        results.append({
            "doc_index": doc["doc_index"],
            "chunk_index": doc["chunk_index"],
            "num_tokens": doc["num_tokens"],
            "content": doc["content"],
            "keywords": [{"keyword": kw[0], "score": kw[1]} for kw in kws]
        })
    return results

def extract_keywords_for_similar_docs(docs: List[ChunkResult], api_key: str, threshold: float = 0.75, model_name: str = "google/embeddinggemma-300m") -> List[KeywordResult]:
    model = SentenceTransformer(model_name)
    client = openai.OpenAI(api_key=api_key)
    llm = OpenAI(client)
    kw_model = KeyLLM(llm)
    texts = [doc["content"] for doc in docs]
    embeddings = model.encode(texts, convert_to_tensor=True)
    keywords = kw_model.extract_keywords(texts, embeddings=embeddings, threshold=threshold)
    results = []
    for doc, kws in zip(docs, keywords):
        results.append({
            "doc_index": doc["doc_index"],
            "chunk_index": doc["chunk_index"],
            "num_tokens": doc["num_tokens"],
            "content": doc["content"],
            "keywords": [{"keyword": kw[0], "score": kw[1]} for kw in kws]
        })
    return results

def load_sample_data():
    """Load sample dataset from local for topic modeling."""
    embed_model = "google/embeddinggemma-300m"
    headers_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/all_headers.json"
    
    logger.info("Loading sample dataset...")
    headers_dict = load_file(headers_file)
    # headers: List[HeaderDoc] = [h for h_list in headers_dict.values() for h in h_list]
    headers: List[HeaderDoc] = headers_dict["https://gamerant.com/new-isekai-anime-2025"]
    documents = [f"{doc['header']}\n\n{doc['content']}" for doc in headers]

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
    documents = load_sample_data()

    # Basic keyword extraction
    print("Basic Keywords:")
    results = extract_basic_keywords(documents)
    print(results)
    save_file(results, f"{OUTPUT_DIR}/extract_basic_keywords.json")

    # Single-word keywords
    print("\nSingle-word Keywords:")
    results = extract_ngram_keywords(documents, ngram_range=(1, 1), stop_words=None)
    print(results)
    save_file(results, f"{OUTPUT_DIR}/extract_single_word_keywords.json")

    # Bigram keyphrases
    print("\nBigram Keyphrases:")
    results = extract_ngram_keywords(documents, ngram_range=(1, 2), stop_words=None)
    print(results)
    save_file(results, f"{OUTPUT_DIR}/extract_bigram_keyphrases.json")

    # Highlight keywords
    print("\nHighlighted Keywords:")
    results = highlight_keywords(documents)
    print(results)
    save_file(results, f"{OUTPUT_DIR}/highlighted_keywords.json")

    # Max Sum Distance
    print("\nMax Sum Distance Keywords:")
    results = extract_maxsum_keywords(documents, ngram_range=(3, 3))
    print(results)
    save_file(results, f"{OUTPUT_DIR}/extract_maxsum_keywords.json")

    # MMR with high diversity
    print("\nMMR Keywords (High Diversity):")
    results = extract_mmr_keywords(documents, diversity=0.7)
    print(results)
    save_file(results, f"{OUTPUT_DIR}/extract_mmr_keywords_high_diversity.json")

    # MMR with low diversity
    print("\nMMR Keywords (Low Diversity):")
    results = extract_mmr_keywords(documents, diversity=0.2)
    print(results)
    save_file(results, f"{OUTPUT_DIR}/extract_mmr_keywords_low_diversity.json")

    # # Flair-based extraction
    # print("\nFlair-based Keywords:")
    # results = extract_keywords_with_flair(documents)
    # print(results)
    # save_file(results, f"{OUTPUT_DIR}/extract_keywords_with_flair.json")

    # # LLM-based extraction (requires API key)
    # print("\nLLM-based Keywords:")
    # results = extract_keywords_with_llm(documents, api_key="MY_API_KEY")
    # print(results)
    # save_file(results, f"{OUTPUT_DIR}/extract_keywords_with_llm.json")

    # # Similar documents extraction (requires API key and multiple documents)
    # docs = [doc["content"] for doc in documents] + ["Another document about machine learning..."]
    # print("\nKeywords for Similar Documents:")
    # results = extract_keywords_for_similar_docs(documents, api_key="MY_API_KEY")
    # print(results)
    # save_file(results, f"{OUTPUT_DIR}/extract_keywords_for_similar_docs.json")

if __name__ == "__main__":
    main()
