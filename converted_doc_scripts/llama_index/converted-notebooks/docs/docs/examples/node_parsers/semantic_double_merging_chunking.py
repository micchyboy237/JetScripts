from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
SemanticDoubleMergingSplitterNodeParser,
LanguageConfig,
)
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Semantic Double Merging and Chunking

Download spacy
"""
logger.info("# Semantic Double Merging and Chunking")

# !pip install spacy

"""
Download required spacy model
"""
logger.info("Download required spacy model")

# !python3 -m spacy download en_core_web_md

"""
Download sample data:
"""
logger.info("Download sample data:")

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'pg_essay.txt'


"""
Load document and create sample splitter:
"""
logger.info("Load document and create sample splitter:")

documents = SimpleDirectoryReader(input_files=["pg_essay.txt"]).load_data()

config = LanguageConfig(language="english", spacy_model="en_core_web_md")
splitter = SemanticDoubleMergingSplitterNodeParser(
    language_config=config,
    initial_threshold=0.4,
    appending_threshold=0.5,
    merging_threshold=0.5,
    max_chunk_size=5000,
)

"""
Get the nodes:
"""
logger.info("Get the nodes:")

nodes = splitter.get_nodes_from_documents(documents)

"""
Sample nodes:
"""
logger.info("Sample nodes:")

logger.debug(nodes[0].get_content())

logger.debug(nodes[5].get_content())

"""
Remember that different spaCy models and various parameter values can perform differently on specific texts. A text that clearly changes its subject matter should have lower threshold values to easily detect these changes. Conversely, a text with a very uniform subject matter should have high threshold values to help split the text into a greater number of chunks. For more information and comparison with different chunking methods check https://bitpeak.pl/chunking-methods-in-rag-methods-comparison/
"""
logger.info("Remember that different spaCy models and various parameter values can perform differently on specific texts. A text that clearly changes its subject matter should have lower threshold values to easily detect these changes. Conversely, a text with a very uniform subject matter should have high threshold values to help split the text into a greater number of chunks. For more information and comparison with different chunking methods check https://bitpeak.pl/chunking-methods-in-rag-methods-comparison/")

logger.info("\n\n[DONE]", bright=True)