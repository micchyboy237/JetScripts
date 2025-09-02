import os
from typing import List

from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils import parse_markdown
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType
from jet.utils.print_utils import print_dict_types
from jet.vectors.semantic_search.header_vector_search import search_headers
from jet.vectors.semantic_search.vector_search_simple import VectorSearch


def get_noun_phrases(texts: List[str]) -> List[List[str]]:
    noun_phrases: List[List[str]] = []
    extractor = ConllExtractor()
    for text in texts:
        textblob = TextBlob(text, np_extractor=None)
        noun_phrases.append(list(textblob.noun_phrases))
    return noun_phrases


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/code/extraction/generated/run_extract_notebook_texts/GenAI_Agents/docs/Academic_Task_Learning_Agent_LangGraph.md"
    md_content: str = load_file(md_file)
    save_file(md_content, f"{output_dir}/md_content.md")

    query = "notes taking"
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    top_k = None
    threshold = 0.0
    chunk_size = 500
    chunk_overlap = 100
    merge_chunks = False

    markdown_tokens = base_parse_markdown(md_content)
    save_file(markdown_tokens, f"{output_dir}/markdown_tokens.json")

    docs: List[HeaderDoc] = derive_by_header_hierarchy(
        md_content, ignore_links=True)
    save_file(docs, f"{output_dir}/docs.json")

    texts = [f"{doc["header"]}\n{doc["content"]}" for doc in docs]
    noun_phrases: List[List[str]] = get_noun_phrases(texts)
    save_file(noun_phrases, f"{output_dir}/noun_phrases.json")

    # Prepare documents for vector search
    search_engine = VectorSearch(embed_model)
    for phrases in noun_phrases:
        search_engine.add_documents(noun_phrases)

    # Run search
    results = search_engine.search(query, top_k=10)
    logger.debug(f"\nQuery: {query}")
    logger.info("Top matches:")
    for num, (doc, score) in enumerate(results, 1):
        print(f"\n{colorize_log(f"{num}.", "ORANGE")} (Score: {
            colorize_log(f"{score:.3f}", "SUCCESS")})")
        print(f"{doc}")
    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{output_dir}/search_results.json")
