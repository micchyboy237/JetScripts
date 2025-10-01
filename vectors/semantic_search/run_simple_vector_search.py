import os
import shutil
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.vectors.semantic_search.vector_search_simple import VectorSearch
import stanza

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Real-world demonstration
if __name__ == "__main__":
    # 1. Specify preffered dimensions
    dimensions = None
    model_name: OLLAMA_MODEL_NAMES = "embeddinggemma"
    # Same example queries
    queries = [
        "How to change max depth?",
    ]
    long_text = "max_depth\ninteger\ndefault: 1\nMax depth of the crawl. Defines how far from the base URL the crawler can explore.\nRequired range: ` x >= 1 `\nmax_breadth\ninteger\ndefault: 20\nMax number of links to follow per level of the tree (i.e., per page).\nRequired range: ` x >= 1 `\nlimit\ninteger\ndefault: 50\nTotal number of links the crawler will process before stopping.\nRequired range: ` x >= 1 `\nselect_paths\nstring[]\nRegex patterns to select only URLs with specific path patterns (e.g., ` /docs/.* ` , ` /api/v1.* ` ).\nselect_domains\nstring[]\nRegex patterns to select crawling to specific domains or subdomains (e.g., ` ^docs.example.com$ ` ).\nexclude_paths\nstring[]\nRegex patterns to exclude URLs with specific path patterns (e.g., ` /private/.* ` , ` /admin/.* ` ).\nexclude_domains\nstring[]\nRegex patterns to exclude specific domains or subdomains from crawling (e.g., ` ^private.example.com$ ` ).\nallow_external\nboolean\ndefault: true\nWhether to include external domain links in the final results list.\ninclude_images\nboolean\ndefault: false\nWhether to include images in the crawl results.\nextract_depth\nenum<string>\ndefault: basic\nAdvanced extraction retrieves more data, including tables and embedded content, with higher success but may increase latency. ` basic ` extraction costs 1 credit per 5 successful extractions, while ` advanced ` extraction costs 2 credits per 5 successful extractions.\nAvailable options:\n` basic ` ,\n` advanced `\nformat\nenum<string>\ndefault: markdown\nThe format of the extracted web page content. ` markdown ` returns content in markdown format. ` text ` returns plain text and may increase latency.\nAvailable options:\n` markdown ` ,\n` text `\ninclude_favicon\nboolean\ndefault: false\nWhether to include the favicon URL for each result"

    nlp = stanza.Pipeline('en', dir=os.path.expanduser("~/.cache/stanza_resources"), processors='tokenize,pos', verbose=True, logging_level="DEBUG")
    doc = nlp(long_text)
    sentences = [sent.text.strip() for sent in doc.sentences]
    # sentences = split_sentences(long_text)
    sample_docs = [
        "##### Help\n\n- Help Center",
        "##### Legal\n\n- Security & Compliance\n- Privacy Policy",
        "##### Partnerships\n\n- IBM",
        "##### Tavily MCP Server\n\n- Tavily MCP Server",
        *sentences,
    ]

    search_engine = VectorSearch(model_name, truncate_dim=dimensions)
    search_engine.add_documents(sample_docs)

    for query in queries:
        results = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for num, (doc, score) in enumerate(results[:10], 1):
            print(f"\n{colorize_log(f"{num}.", "ORANGE")} (Score: {
                  colorize_log(f"{score:.3f}", "SUCCESS")})")
            print(f"{doc}")

    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")
