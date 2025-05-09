import json
from typing import Any, Callable, Union, List, Dict, Optional, Literal, TypedDict, DefaultDict
from jet.code.utils import ProcessedResult, process_markdown_file
from jet.file.utils import load_file, save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import ModelKey, ModelType
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.markdown import extract_block_content
import numpy as np
from collections import defaultdict
import faiss
from sentence_transformers import SentenceTransformer
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from bs4 import BeautifulSoup
import trafilatura
import re
# from fast_langdetect import detect
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize, word_tokenize

MODEL: ModelKey = "qwen3-0.6b-4bit"
mlx = MLX(MODEL)


def main():
    """
    Demonstrates semantic reranking for diverse, long, and unstructured web-scraped content.
    """
    # Example 1: News Article Search
    logger.orange("\n=== Example 1: News Article Search ===")
    news_query = "AI in healthcare"
    news_articles = [
        """
        <h1>AI Revolution in Healthcare</h1>
        <p>Published on 2025-02-01 by Jane Smith</p>
        <p>Artificial intelligence is transforming healthcare with predictive diagnostics and personalized treatments. Machine learning models analyze patient data to predict diseases like cancer with high accuracy. Hospitals are adopting AI to streamline operations. Click here to read more about AI trends.</p>
        <p>Subscribe to our newsletter for daily updates!</p>
        """,
        """
        <h2>Tech Stocks Surge in 2025</h2>
        <p>Published on 2025-01-15 by John Doe</p>
        <p>Tech companies, including those in AI, are seeing record stock gains. However, healthcare remains a challenging sector for investors. Read more about market trends.</p>
        <div>Follow us on social media!</div>
        """,
        """
        <h1>New Gadgets for 2025</h1>
        <p>From smartwatches to AI-powered home devices, new gadgets are hitting the market. Learn about the latest tech innovations. Sign up for our tech newsletter!</p>
        """
    ]
    news_ids = ["news1", "news2", "news3"]
    news_results = news_article_search(
        news_query, news_articles, news_ids, top_k=2)
    logger.gray("Query:")
    logger.debug(news_query)
    for result in news_results:
        logger.log(
            f"Rank {result['rank']}:",
            f"(Score: {result['score']:.3f}, Words: {result['word_count']})",
            f"{result['text'][:100]}...",
            colors=["DEBUG", "SUCCESS", "WHITE"],
        )
        logger.newline()

    # Example 2: Tech Blog Recommendation
    logger.orange("\n=== Example 2: Tech Blog Recommendation ===")
    blog_queries = ["machine learning advancements", "AI trends 2025"]
    blog_posts = [
        """
        <div class="post">
        <h2>The Future of Machine Learning</h2>
        <p>Machine learning advancements in 2025 include larger language models and better interpretability. Neural networks are now more efficient, thanks to sparsity techniques. This blog explores how ML is reshaping industries like finance and healthcare.</p>
        <p>Join our webinar to learn more! Sign up today.</p>
        <p>We also cover AI ethics, a critical topic for 2025. From bias mitigation to transparency, ethical AI is a priority.</p>
        </div>
        """,
        """
        <h1>AI Trends to Watch</h1>
        <p>Generative AI is booming, with applications in content creation and design. In 2025, expect AI to integrate with IoT for smarter homes. Click here to subscribe to our tech updates.</p>
        <p>Other trends include AI in autonomous vehicles and robotics.</p>
        <div>Follow our blog for more insights!</div>
        """,
        """
        <h2>Python for Beginners</h2>
        <p>Learn Python programming with our step-by-step guide. While not directly about AI, Python is the backbone of many ML frameworks.</p>
        <p>Sign up for our coding bootcamp!</p>
        """
    ]
    blog_ids = ["blog1", "blog2", "blog3"]
    blog_results = tech_blog_recommendation(
        blog_queries, blog_posts, blog_ids, top_k=2)
    logger.gray("Query:")
    logger.debug(format_json(blog_queries))
    for result in blog_results:
        logger.log(
            f"Rank {result['rank']}:",
            f"(Score: {result['score']:.3f}, Words: {result['word_count']})",
            f"{result['text'][:100]}...",
            colors=["DEBUG", "SUCCESS", "WHITE"],
        )
        logger.newline()

    # Example 3: E-Commerce Product Scraping
    logger.orange("\n=== Example 3: E-Commerce Product Scraping ===")
    product_query = "wireless earbuds"
    product_listings = [
        """
        <div class="product">
        <h2>AirPods Pro 2 - $249.99</h2>
        <p>Experience immersive sound with the AirPods Pro 2. Features include active noise cancellation, spatial audio, and up to 6 hours of battery life. Perfect for music lovers and professionals. Add to cart now!</p>
        <p>20% off this week only!</p>
        <p>Compatible with iOS and Android devices. Includes wireless charging case.</p>
        </div>
        """,
        """
        <h2>Sony WF-1000XM5 - $299.99</h2>
        <p>The Sony WF-1000XM5 wireless earbuds offer industry-leading noise cancellation and high-resolution audio. With 8 hours of battery life and a compact design, they’re ideal for travel. Buy now and save 10%!</p>
        <p>Free shipping on orders over $50.</p>
        """,
        """
        <h2>LED Desk Lamp - $29.99</h2>
        <p>Brighten your workspace with this energy-efficient LED desk lamp. Adjustable brightness and color temperature. Not wireless earbuds, but a great deal! Click here to purchase.</p>
        """
    ]
    listing_ids = ["prod1", "prod2", "prod3"]
    product_results = ecommerce_product_scraping(
        product_query, product_listings, listing_ids, top_k=2)
    logger.gray("Query:")
    logger.debug(product_query)
    for result in product_results:
        logger.log(
            f"Rank {result['rank']}:",
            f"(Score: {result['score']:.3f}, Words: {result['word_count']})",
            f"{result['text'][:100]}...",
            colors=["DEBUG", "SUCCESS", "WHITE"],
        )
        logger.newline()

    # Example 4: Forum Thread Prioritization
    logger.orange("\n=== Example 4: Forum Thread Prioritization ===")
    forum_query = "Python error handling"
    forum_threads = [
        """
        <div class="thread">
        <p>Posted by User123: I'm getting a TypeError in Python when handling exceptions. Here's my code: <code>try: x = int(input()) except: print("Error")</code>. Any tips on proper error handling?</p>
        <p>Posted by DevGuru: You should specify the exception type, like <code>except ValueError:</code>. This avoids catching unrelated errors. Also, log the error with <code>logging</code>.</p>
        <p>Posted by User123: Thanks! That fixed it. Any libraries for advanced error handling?</p>
        </div>
        """,
        """
        <div class="thread">
        <p>Posted by CodeMaster: How do I optimize Python loops? My script is slow when processing large datasets.</p>
        <p>Posted by DataNerd: Use NumPy for vectorized operations or list comprehensions. Avoid nested loops where possible.</p>
        <p>Re: Check out the <code>multiprocessing</code> module for parallel processing.</p>
        </div>
        """,
        """
        <div class="thread">
        <p>Posted by Newbie: What's the best Python IDE? I'm new to coding.</p>
        <p>Posted by ProCoder: Try VS Code with Python extensions or PyCharm for advanced features.</p>
        </div>
        """
    ]
    thread_ids = ["thread1", "thread2", "thread3"]
    thread_results = forum_thread_prioritization(
        forum_query, forum_threads, thread_ids, top_k=2)
    logger.gray("Query:")
    logger.debug(forum_query)
    for result in thread_results:
        logger.log(
            f"Rank {result['rank']}:",
            f"(Score: {result['score']:.3f}, Words: {result['word_count']})",
            f"{result['text'][:100]}...",
            colors=["DEBUG", "SUCCESS", "WHITE"],
        )
        logger.newline()


def main2():
    import os
    import shutil

    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/docs/5_contextual_chunk_headers_rag.md"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    threshold = 0.0
    top_k = 10
    use_bm25 = True

    # data: list[dict] = load_file(data_file)
    data: list[ProcessedResult] = process_markdown_file(md_file)
    texts = [item["text"] for item in data]

    query = write_query(texts)

    logger.info(
        f"Reranking {len(texts)} web scraped contents for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=texts,
        # ids=article_ids,
        threshold=threshold,
        model=["all-MiniLM-L12-v2", "distilbert-base-nli-stsb-mean-tokens"],
        fuse_method="average",
        metrics="cosine",
        domain=None,
        use_index=len(texts) > 1000,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100
    )

    token_counts: list[int] = mlx.count_tokens(texts, prevent_total=True)
    for result in results:
        tokens = token_counts[result["doc_index"]]
        result["tokens"] = tokens

    logger.gray("Query:")
    logger.debug(query)
    for result in results[:5]:
        logger.log(
            f"Rank {result['rank']}:",
            f"Doc: {result['doc_index']}, Tokens: {result['tokens']}",
            f"\nScore: {result['score']:.3f}",
            f"\n{json.dumps(result['text'])[:100]}...",
            colors=["ORANGE", "DEBUG", "SUCCESS", "WHITE"],
        )

    save_file({
        "query": query,
        "model": mlx.model_path,
        "results": results
    }, f"{output_dir}/query_scores.json")


if __name__ == "__main__":
    # main()
    main2()
