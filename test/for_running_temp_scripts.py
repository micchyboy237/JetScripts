from llama_index.core.schema import Document
from llama_index.core.node_parser.text.semantic_double_merging_splitter import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
from jet.features.scrape_search_chat import get_docs_from_html
from jet.file.utils import load_file
from jet.logger import logger
from jet.utils.commands import copy_to_clipboard


if __name__ == "__main__":
    # Setup docs

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_anime_scraper/honeysanime_com/scraped_html.html"
    html = load_file(data_file)
    headers = get_docs_from_html(html)
    docs = [Document(text=header.text) for header in headers]

    # doc = Document(
    #     text="Warsaw: Warsaw, the capital city of Poland, is a bustling metropolis located on the banks of the Vistula River. "
    #     "It is known for its rich history, vibrant culture, and resilient spirit. Warsaw's skyline is characterized by a mix of historic architecture and modern skyscrapers. "
    #     "The Old Town, with its cobblestone streets and colorful buildings, is a UNESCO World Heritage Site.\n\n"
    #     "Football: Football, also known as soccer, is a popular sport played by millions of people worldwide. "
    #     "It is a team sport that involves two teams of eleven players each. The objective of the game is to score goals by kicking the ball into the opposing team's goal. "
    #     "Football matches are typically played on a rectangular field called a pitch, with goals at each end. "
    #     "The game is governed by a set of rules known as the Laws of the Game. Football is known for its passionate fanbase and intense rivalries between clubs and countries. "
    #     "The FIFA World Cup is the most prestigious international football tournament.\n\n"
    #     "Mathematics: Mathematics is a fundamental discipline that deals with the study of numbers, quantities, and shapes. "
    #     "Its branches include algebra, calculus, geometry, and statistics."
    # )
    # docs = [doc]

    splitter = SemanticDoubleMergingSplitterNodeParser(
        initial_threshold=0.7,
        appending_threshold=0.8,
        merging_threshold=0.7,
        max_chunk_size=1000,
        language_config=LanguageConfig(
            language="english", spacy_model="en_core_web_md"),
    )
    splitter.language_config.load_model()

    nodes = splitter.get_nodes_from_documents(docs)

    copy_to_clipboard(nodes)
    logger.success(f"Nodes: {len(nodes)}")
