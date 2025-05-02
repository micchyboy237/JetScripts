import asyncio
import os
import shutil
from typing import Generator, Optional
from datetime import datetime
from jet.wordnet.analyzers.text_analysis import analyze_text
from tqdm import tqdm
from jet.code.splitter_markdown_utils import Header, extract_md_header_contents, get_md_header_contents
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.logger import logger
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import extract_texts_by_hierarchy, safe_path_from_url, scrape_links, scrape_title_and_metadata, search_data
from jet.scrapers.hrequests_utils import scrape_urls
from jet.transformers.formatters import format_json
from jet.utils.url_utils import normalize_url
from jet.vectors.hybrid_search_engine import HybridSearchEngine
from jet.wordnet.similarity import compute_info, query_similarity_scores

logger.info("Initializing MLX and embedding function")
mlx = MLX()


def get_url_html_tuples(urls: list[str], top_n: int = 3, num_parallel: int = 3, min_header_count: int = 10, min_avg_word_count: int = 10, output_dir: Optional[str] = None) -> Generator[list[Header], None, None]:
    urls = [normalize_url(url) for url in urls]

    for url, html in scrape_multiple_urls(urls, top_n=top_n, num_parallel=num_parallel, min_header_count=min_header_count, min_avg_word_count=min_avg_word_count):

        headers = get_md_header_contents(html)

        yield {
            "url": url,
            "headers": headers,
            "html": html,
        }


def decompose_query(original_query, num_subqueries=None, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    # Get current date dynamically
    current_date = datetime.now().strftime("%B %d, %Y")

    # Define system prompt based on whether num_subqueries is provided
    if num_subqueries is None:
        system_prompt = """
        You are an AI assistant specialized in breaking down complex browser-based queries for web search and information retrieval, as of {current_date}. Your task is to decompose a query into simpler sub-questions that, when answered together, fully and comprehensively address all components of the original query. Each sub-question should target a specific, answerable aspect of the query to guide web searches or scraping efforts, ensuring no part of the original query is overlooked.

        Format your response as a numbered list of sub-questions, with a reasonable number of sub-questions (typically 2 to 4, depending on the query's complexity), each on a new line, with the following structure:
        1. Sub-question text
        2. Sub-question text
        ...

        Ensure each sub-question:
        - Starts with a number followed by a period (e.g., '1.', '2.').
        - Is a clear, standalone question ending with a question mark.
        - Targets a distinct aspect of the original query, collectively covering all its components, including explicit and implicit elements.
        - Is specific and designed for web search or content analysis, avoiding vague or overly broad questions.
        - Reflects the current date ({current_date}) when relevant, especially for queries about trends, recent events, or time-sensitive information.
        - Contains no additional text, headings, explanations, or mentions of specific entities outside the numbered list.
        - Ends with [TERMINATE] on a new line after the list.
        """.format(current_date=current_date)
    else:
        system_prompt = """
        You are an AI assistant specialized in breaking down complex browser-based queries for web search and information retrieval, as of {current_date}. Your task is to decompose a query into simpler sub-questions that, when answered together, fully and comprehensively address all components of the original query. Each sub-question should target a specific, answerable aspect of the query to guide web searches or scraping efforts, ensuring no part of the original query is overlooked.

        Format your response as a numbered list of exactly {num_subqueries} sub-questions, each on a new line, with the following structure:
        1. Sub-question text
        2. Sub-question text
        ...

        Ensure each sub-question:
        - Starts with a number followed by a period (e.g., '1.', '2.').
        - Is a clear, standalone question ending with a question mark.
        - Targets a distinct aspect of the original query, collectively covering all its components, including explicit and implicit elements.
        - Is specific and designed for web search or content analysis, avoiding vague or overly broad questions.
        - Reflects the current date ({current_date}) when relevant, especially for queries about trends, recent events, or time-sensitive information.
        - Contains no additional text, headings, explanations, or mentions of specific entities outside the numbered list.
        - Ends with [TERMINATE] on a new line after the list.
        """.format(current_date=current_date, num_subqueries=num_subqueries)

    logger.debug(original_query)
    stream_response = mlx.stream_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",
                "content": f"Decompose this query into {'exactly ' + str(num_subqueries) if num_subqueries else 'a reasonable number of'} sub-questions: {original_query}"}
        ],
        model=model,
        temperature=0.2,
        stop=["[TERMINATE]", "\n\n"]
    )
    content = ""
    for chunk in stream_response:
        chunk_content = chunk["choices"][0]["message"]["content"]
        logger.success(chunk_content, flush=True)
        content += chunk_content
    logger.newline()
    lines = content.split("\n")
    sub_queries = []
    for line in lines:
        line = line.strip()
        if line and any(line.startswith(f"{i}.") for i in range(1, (num_subqueries or 4) + 1)):
            query = line[line.find(".") + 1:].strip()
            if query.endswith("?"):  # Ensure it's a question
                sub_queries.append(query)
    # If num_subqueries is specified, ensure exactly that number are returned
    return sub_queries[:num_subqueries] if num_subqueries else sub_queries


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "generated",
                              os.path.splitext(os.path.basename(__file__))[0])

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    query = "List trending isekai anime this year."
    embed_models = ["mxbai-embed-large"]

    # Search web engine
    search_results = search_data(query)
    save_file({"query": query, "results": search_results}, os.path.join(
        output_dir, "search_results.json"))

    # Decompose query to sub-queries
    sub_queries = decompose_query(query)
    save_file({"query": query, "sub_queries": sub_queries}, os.path.join(
        output_dir, "queries.json"))

    # Rerank docs
    queries = [query, *sub_queries]
    search_result_docs = [
        f"Title: {result["title"]}\nContent: {result["content"]}" for result in search_results]
    top_n = len(search_result_docs)

    query_scores = query_similarity_scores(
        queries, search_result_docs, model=embed_models)
    save_file({"queries": queries, "results": query_scores},
              os.path.join(output_dir, "search_query_scores.json"))

    # Use hybrid search
    # engine = HybridSearchEngine()
    # engine.fit(search_result_docs)

    # print("\n🔎 Hybrid Search Results:\n")
    # simple_results = engine.search(
    #     query, top_n=top_n, alpha=0.5, use_mmr=False)
    # for r in simple_results:
    #     print(f"Score: {r['score']:.4f} | Document: {r['document'][:100]}")
    # save_file(simple_results, os.path.join(output_dir, "hybrid_search.json"))

    # print("\n🔎 Hybrid Search Results w/ MMR Diversity:\n")
    # mmr_results = engine.search(
    #     query, top_n=top_n, alpha=0.5, use_mmr=True, lambda_param=0.7)
    # for r in mmr_results:
    #     print(f"Score: {r['score']:.4f} | Document: {r['document'][:100]}")
    # save_file(mmr_results, os.path.join(
    #     output_dir, "hybrid_search_with_diversity.json"))

    sub_dir = os.path.join(output_dir, "searched_html")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Convert html to docs
    urls = [item["url"] for item in search_results]
    html_list = asyncio.run(scrape_urls(urls, num_parallel=3))
    all_url_html_tuples = zip(urls, html_list)

    html_info = []
    for url, html_str in all_url_html_tuples:
        output_dir_url = safe_path_from_url(url, sub_dir)

        all_links = scrape_links(html_str, base_url=url)
        save_file(all_links, os.path.join(
            output_dir_url, "links.json"))

        title_and_metadata = scrape_title_and_metadata(html_str)
        save_file(title_and_metadata, os.path.join(
            output_dir_url, "title_and_metadata.json"))

    url_html_tuples = []
    for url, html_str in tqdm(all_url_html_tuples):
        if html_str:
            headers = get_md_header_contents(html_str)
            logger.debug(
                f"Scraped {url}, headers length: {len(headers)}")

            output_dir_url = safe_path_from_url(url, sub_dir)

            url_html_tuples.append((url, html_str))

            save_file(html_str, os.path.join(output_dir_url, "doc.html"))

            headings = extract_texts_by_hierarchy(html_str)
            save_file(headings, f"{output_dir_url}/doc.json")

            headers = [item["text"].splitlines()[0].strip()
                       for item in headings]
            save_file("\n".join(headers), os.path.join(
                output_dir_url, "headers.md"))

            html_docs = [item["text"] for item in headings]
            md_text = "\n\n".join(html_docs)
            save_file(md_text, os.path.join(output_dir_url, "doc.md"))

            # Analyze doc
            stats = analyze_text(md_text)
            save_file(stats, f"{output_dir_url}/stats.json")

            # Rerank docs
            query_scores = query_similarity_scores(
                queries, html_docs, model=embed_models)
            save_file({"queries": queries, "results": query_scores},
                      os.path.join(output_dir_url, "query_scores.json"))

            save_file({
                "url": url,
                "headers": len(headers),
                "info": compute_info(query_scores),
                "top_header": {
                    "doc_index": query_scores[0]
                }
            }, os.path.join(output_dir_url, "info.json"))

        else:
            logger.error(f"Failed to fetch {url}")

    logger.success(f"Done scraping urls {len(url_html_tuples)}")

    # url_html_tuples = []
    # for item in get_url_html_tuples(urls, top_n=5):
    #     url = item["url"]
    #     headers = item["headers"]
    #     html = item["html"]

    #     url_html_tuples.append((url, html))

    #     html_docs = [header["content"] for header in headers]

    #     output_dir_url = safe_path_from_url(url, sub_dir)
    #     save_file({
    #         "url": url,
    #         "headers": len(headers),
    #     }, os.path.join(output_dir_url, "info.json"))
    #     save_file(html, os.path.join(output_dir_url, "doc.html"))
    #     save_file("\n\n".join(html_docs),
    #               os.path.join(output_dir_url, "doc.md"))

    #     # Rerank docs
    #     query_scores = query_similarity_scores(
    #         queries, html_docs, model=embed_models)
    #     save_file({"queries": queries, "results": query_scores},
    #               os.path.join(output_dir_url, "query_scores.json"))

    #     # Use hybrid search
    #     # engine = HybridSearchEngine()
    #     # engine.fit(html_docs)

    #     # print("\n🔎 Docs Hybrid Search Results:\n")
    #     # simple_results = engine.search(
    #     #     query, top_n=top_n, alpha=0.5, use_mmr=False)
    #     # for r in simple_results:
    #     #     print(f"Score: {r['score']:.4f} | Document: {r['document'][:100]}")
    #     # save_file(simple_results, os.path.join(
    #     #     output_dir_url, "hybrid_search.json"))

    #     # print("\n🔎 Docs Hybrid Search Results w/ MMR Diversity:\n")
    #     # mmr_results = engine.search(
    #     #     query, top_n=top_n, alpha=0.5, use_mmr=True, lambda_param=0.7)
    #     # for r in mmr_results:
    #     #     print(f"Score: {r['score']:.4f} | Document: {r['document'][:100]}")
    #     # save_file(mmr_results, os.path.join(
    #     #     output_dir_url, "hybrid_search_with_diversity.json"))

    # logger.success(f"Done scraping urls {len(url_html_tuples)}")
