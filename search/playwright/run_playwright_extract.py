import os
import shutil
from jet.code.markdown_types import HeaderSearchResult
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.scrapers.utils import search_data
from jet.utils.text import format_sub_dir
from jet.vectors.semantic_search.search_docs import search_docs
from jet.wordnet.text_chunker import ChunkResult, chunk_texts_with_data
from typing import List, TypedDict
from jet.file.utils import save_file
from jet.search.playwright.playwright_extract import PlaywrightExtract, convert_html_to_markdown

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)

class ContextItem(TypedDict):
    doc_idx: int
    tokens: int
    text: str

class Topic(TypedDict):
    rank: int
    doc_index: int
    score: float
    text: str
    
class SearchResult(TypedDict):
    id: str
    rank: int
    doc_index: int
    score: float
    tokens: int
    text: str

def sync_example(urls):
    """Demonstrate synchronous usage of PlaywrightExtract."""
    extractor = PlaywrightExtract()
    try:
        result = extractor._run(
            urls=urls,
            extract_depth="basic",
            include_images=False,
            include_favicon=False,
            format="markdown"
        )
        print("Basic extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Content length: {len(item['raw_content'])})")
        print(f"Response time: {result['response_time']:.2f} seconds")
        save_file(result, f"{OUTPUT_DIR}/example_1_result.json")
    except Exception as e:
        print(f"Error in basic extract: {e}")

    try:
        result = extractor._run(
            urls=urls,
            extract_depth="advanced",
            include_images=True,
            include_favicon=True,
            format="text"
        )
        print("\nAdvanced extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Images: {len(item['images'])}, Favicon: {item['favicon']})")
        save_file(result, f"{OUTPUT_DIR}/example_2_result.json")
    except Exception as e:
        print(f"Error in advanced extract: {e}")

async def async_example(urls):
    """Demonstrate asynchronous usage of PlaywrightExtract."""
    extractor = PlaywrightExtract()
    try:
        result = await extractor._arun(
            urls=urls,
            extract_depth="basic",
            include_images=True,
            include_favicon=False,
            format="markdown"
        )
        print("\nAsync basic extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Images: {len(item['images'])})")
        print(f"Response time: {result['response_time']:.2f} seconds")
        save_file(result, f"{OUTPUT_DIR}/example_3_result.json")
    except Exception as e:
        print(f"Error in async basic extract: {e}")

    try:
        result = await extractor._arun(
            urls=urls,
            extract_depth="advanced",
            include_images=False,
            include_favicon=True,
            format="text"
        )
        print("\nAsync advanced extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Favicon: {item['favicon']})")
        save_file(result, f"{OUTPUT_DIR}/example_4_result.json")
    except Exception as e:
        print(f"Error in async advanced extract: {e}")

def extract_doc_chunks(html: str, url: str, chunk_size: int = 200, chunk_overlap: int = 50) -> List[ChunkResult]:
    md_content = convert_html_to_markdown(html, ignore_links=False)
    # original_docs = derive_by_header_hierarchy(md_content, ignore_links=True)
    chunks = chunk_texts_with_data(md_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model)
    return chunks

def extract_topics(
    query: str,
    documents: List[str],
    model: str = "embeddinggemma",
    top_k: int = None
) -> List[Topic]:
    """Extract topics from documents using BERTopic.
    
    Args:
        query: Search query to find relevant topics
        documents: List of documents to analyze
        model: Embedding model to use
        top_k: Number of top topics to return (if None, return all)
        
    Returns:
        List of Topic objects with rank, doc_index, score, and text
    """
    if not documents:
        return []
    
    try:
        from jet.adapters.bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        import logging
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting topic extraction for {len(documents)} documents")
        
        # Create BERTopic model with custom embedding model
        # Use the specified model for embeddings
        # embedding_model = SentenceTransformer(model)
        topic_model = BERTopic(
            # embedding_model=embedding_model,
            calculate_probabilities=True,
        )
        
        # Fit the model to documents
        logger.info("Fitting BERTopic model...")
        topics, probs = topic_model.fit_transform(documents)
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        logger.info(f"Found {len(topic_info)} topics")
        
        # Find topics similar to the query
        logger.info(f"Finding topics similar to query: '{query}'")
        similar_topics, similarities = topic_model.find_topics(query, top_n=len(topic_info))
        
        # Create topic results
        results = []
        for rank, (topic_id, similarity) in enumerate(zip(similar_topics, similarities)):
            if topic_id == -1:  # Skip outlier topic
                continue
                
            # Get topic words
            topic_words = topic_model.get_topic(topic_id)
            if not topic_words:
                continue
                
            # Create topic text from top words
            topic_text = " ".join([word[0] for word in topic_words[:5]])
            
            # Find the best document for this topic
            topic_docs = [i for i, t in enumerate(topics) if t == topic_id]
            if not topic_docs:
                continue
                
            # Get the document with highest probability for this topic
            best_doc_idx = None
            best_score = 0.0
            for doc_idx in topic_docs:
                if probs is not None and doc_idx < len(probs):
                    doc_probs = probs[doc_idx]
                    if topic_id < len(doc_probs):
                        topic_prob = doc_probs[topic_id]
                        if topic_prob > best_score:
                            best_score = topic_prob
                            best_doc_idx = doc_idx
            
            if best_doc_idx is not None:
                results.append({
                    "rank": rank + 1,
                    "doc_index": best_doc_idx,
                    "score": float(similarity),
                    "text": topic_text
                })
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top_k filter if specified
        if top_k is not None:
            results = results[:top_k]
            
        logger.info(f"Returning {len(results)} topics")
        return results
        
    except ImportError as e:
        logger.error(f"BERTopic not available: {e}")
        # Fallback to simple keyword-based topic extraction
        return _fallback_topic_extraction(query, documents, top_k)
    except Exception as e:
        logger.error(f"Error in topic extraction: {e}")
        return _fallback_topic_extraction(query, documents, top_k)


def _fallback_topic_extraction(
    query: str, 
    documents: List[str], 
    top_k: int = None
) -> List[Topic]:
    """Fallback topic extraction using simple keyword matching."""
    import re
    from collections import Counter
    
    # Extract query keywords
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    
    results = []
    for doc_idx, doc in enumerate(documents):
        # Extract document words
        doc_words = re.findall(r'\b\w+\b', doc.lower())
        word_counts = Counter(doc_words)
        
        # Calculate similarity based on common words
        common_words = query_words.intersection(set(doc_words))
        if common_words:
            # Calculate score based on word frequency and commonality
            score = sum(word_counts[word] for word in common_words) / len(doc_words)
            
            # Create topic text from most frequent words
            top_words = [word for word, _ in word_counts.most_common(5)]
            topic_text = " ".join(top_words)
            
            results.append({
                "rank": len(results) + 1,
                "doc_index": doc_idx,
                "score": float(score),
                "text": topic_text
            })
    
    # Sort by score and apply top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        results = results[:top_k]
        
    return results


def test_extract_topics():
    """Test the extract_topics function with sample data."""
    print("Testing extract_topics function...")
    
    # Sample documents covering different topics
    test_documents = [
        "Machine learning algorithms are revolutionizing data analysis and pattern recognition in various industries.",
        "Deep learning neural networks require large datasets and significant computational power for training.",
        "Natural language processing enables computers to understand and generate human language effectively.",
        "Computer vision applications can identify and classify objects in images and videos with high accuracy.",
        "Data science combines statistical analysis, programming skills, and domain expertise to extract insights.",
        "Artificial intelligence is transforming healthcare, finance, and transportation sectors worldwide.",
        "Reinforcement learning agents learn optimal strategies through trial and error interactions with environments.",
        "Supervised learning algorithms use labeled training data to make accurate predictions on new examples.",
        "Unsupervised learning discovers hidden patterns in data without requiring labeled examples.",
        "Transfer learning allows models trained on one task to be adapted for related tasks efficiently."
    ]
    
    # Test different queries
    test_queries = [
        "machine learning algorithms",
        "neural networks and deep learning", 
        "data science and analytics",
        "artificial intelligence applications"
    ]
    
    for query in test_queries:
        print(f"\nTesting with query: '{query}'")
        try:
            topics = extract_topics(
                query=query,
                documents=test_documents,
                model="embeddinggemma",
                top_k=3
            )
            
            print(f"Found {len(topics)} topics:")
            for topic in topics:
                print(f"  - {topic['text']} (Score: {topic['score']:.3f})")
                
        except Exception as e:
            print(f"Error: {e}")

def search_contexts(query: str, html: str, url: str, model: str) -> List[HeaderSearchResult]:
    chunks = extract_doc_chunks(html, url)
    texts = [chunk["content"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    
    top_k = None
    threshold = 0.0
    chunk_size = 128
    chunk_overlap = 64
    search_results = list(
        search_docs(
            query,
            texts,
            model,
            top_k=top_k,
            ids=ids,
            threshold=threshold,
        )
    )
    # contexts: List[ContextItem] = []
    # for idx, doc in enumerate(original_docs):
    #     doc["source"] = url
    #     text = doc["header"] + "\n\n" + doc["content"]
    #     contexts.append({
    #         "doc_idx": idx,
    #         "tokens": token_counter(text, model),
    #         "text": text,
    #     })
    return search_results

def scrape_urls_data(query: str, urls: List[str], model: str, use_cache: bool = True, urls_limit: int = 10):
    sub_dir_query = format_sub_dir(query)
    base_output_dir = f"{OUTPUT_DIR}/{sub_dir_query}"
    shutil.rmtree(base_output_dir, ignore_errors=True)

    # Fetch URLs if empty
    if not urls:
        search_engine_results = search_data(query, use_cache=use_cache)
        urls = [r["url"] for r in search_engine_results][:urls_limit]
        save_file(search_engine_results, f"{base_output_dir}/search_engine_results.json")
    save_file(urls, f"{base_output_dir}/urls.json")

    extractor = PlaywrightExtract()
    result_stream = extractor._stream(
        urls=urls,
        extract_depth="advanced",
        include_images=True,
        include_favicon=True,
        format="text"
    )
    print("\nAdvanced extract results stream:")
    count = 0
    all_documents = {}
    # results = []
    for result in result_stream:
        count += 1
        meta = result.copy().pop("meta")
        chunks = extract_doc_chunks(meta["html"], result['url'])
        documents = [doc["content"] for doc in chunks]
        topics = extract_topics(query, documents, model, top_k=5)
        search_results = search_contexts(query, meta["html"], result['url'], model)
        sub_dir_url = format_sub_dir(result['url'])
        print(f"URL: {sub_dir_url} (Images: {len(result['images'])}, Favicon: {result['favicon']})")
        # results.extend(result)
        all_documents[result["url"]] = chunks
        save_file({
            "query": query,
            "count": len(chunks),
            "chunks": chunks,
        }, f"{base_output_dir}/{sub_dir_url}/chunks.json")
        save_file({
            "query": query,
            "count": len(documents),
            "documents": documents,
        }, f"{base_output_dir}/{sub_dir_url}/documents.json")
        save_file(result, f"{base_output_dir}/{sub_dir_url}/results.json")
        save_file(topics, f"{base_output_dir}/{sub_dir_url}/topics.json")
        save_file(search_results, f"{base_output_dir}/{sub_dir_url}/search_results.json")
        save_file({
            "url": result["url"],
            "tokens": meta["tokens"],
        }, f"{base_output_dir}/{sub_dir_url}/info.json")
        save_file(meta["analysis"], f"{base_output_dir}/{sub_dir_url}/analysis.json")
        save_file(meta["text_links"], f"{base_output_dir}/{sub_dir_url}/text_links.json")
        save_file(meta["image_links"], f"{base_output_dir}/{sub_dir_url}/image_links.json")
        save_file(meta["html"], f"{base_output_dir}/{sub_dir_url}/page.html")
        save_file(meta["markdown"], f"{base_output_dir}/{sub_dir_url}/markdown.md")
        save_file(clean_markdown_links(meta["markdown"]), f"{base_output_dir}/{sub_dir_url}/markdown_no_links.md")
        save_file(meta["md_tokens"], f"{base_output_dir}/{sub_dir_url}/md_tokens.json")
        save_file(meta["screenshot"], f"{base_output_dir}/{sub_dir_url}/screenshot.png")
    return all_documents

if __name__ == "__main__":
    # urls = [
    #     "https://docs.tavily.com/documentation/api-reference/endpoint/crawl",
    # ]
    # query = "How to change max depth?"

    urls = []
    query = "Top isekai anime 2025"

    model = "embeddinggemma"
    urls_limit = 10
    use_cache = True

    sub_dir_query = format_sub_dir(query)
    # print("Running stream examples...")
    all_documents = scrape_urls_data(query, urls, model, use_cache=use_cache, urls_limit=urls_limit)
    save_file(all_documents, f"{OUTPUT_DIR}/{sub_dir_query}/all_documents.json")

    
    # # Example of using extract_topics function
    # print("\nRunning topic extraction example...")
    # sample_documents = [
    #     "Machine learning algorithms are used for data analysis and pattern recognition.",
    #     "Deep learning neural networks require large datasets and computational power.",
    #     "Natural language processing helps computers understand human language.",
    #     "Computer vision applications can identify objects in images and videos.",
    #     "Data science combines statistics, programming, and domain expertise.",
    #     "Artificial intelligence is transforming various industries worldwide.",
    #     "Reinforcement learning agents learn through trial and error interactions.",
    #     "Supervised learning uses labeled training data to make predictions."
    # ]


    # topics = extract_topics(
    #     query="machine learning and AI",
    #     documents=sample_documents,
    #     model=model,
    #     top_k=5
    # )
    
    # print(f"Found {len(topics)} topics:")
    # for topic in topics:
    #     print(f"  Rank {topic['rank']}: {topic['text']} (Score: {topic['score']:.3f}, Doc: {topic['doc_index']})")
    
    # save_file({
    #     "query": "machine learning and AI",
    #     "topics": topics,
    #     "document_count": len(sample_documents)
    # }, f"{OUTPUT_DIR}/topic_extraction_results.json")


    # Test the extract_topics function
    # print("\n" + "="*50)
    # test_extract_topics()
    
    # texts = [doc["text"] for doc in all_contexts]
    # search_results = search(query, texts, model)
    # save_file({
    #     "query": query,
    #     "count": len(search_results),
    #     "results": search_results,
    # }, f"{OUTPUT_DIR}/search_results.json")

    # print("Running synchronous examples...")
    # sync_example(urls)
    # print("\nRunning asynchronous examples...")
    # asyncio.run(async_example(urls))
