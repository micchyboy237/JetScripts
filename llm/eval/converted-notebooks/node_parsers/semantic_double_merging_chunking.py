from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Download spacy

# !pip install spacy

# Download required spacy model

# !python3 -m spacy download en_core_web_md

# Download sample data:

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'pg_essay.txt'


# Load document and create sample splitter:

documents = SimpleDirectoryReader(input_files=["pg_essay.txt"]).load_data()

config = LanguageConfig(language="english", spacy_model="en_core_web_md")
splitter = SemanticDoubleMergingSplitterNodeParser(
    language_config=config,
    initial_threshold=0.4,
    appending_threshold=0.5,
    merging_threshold=0.5,
    max_chunk_size=5000,
)

# Get the nodes:

nodes = splitter.get_nodes_from_documents(documents)

# Sample nodes:

print(nodes[0].get_content())

print(nodes[5].get_content())

# Remember that different spaCy models and various parameter values can perform differently on specific texts. A text that clearly changes its subject matter should have lower threshold values to easily detect these changes. Conversely, a text with a very uniform subject matter should have high threshold values to help split the text into a greater number of chunks. For more information and comparison with different chunking methods check https://bitpeak.pl/chunking-methods-in-rag-methods-comparison/

logger.info("\n\n[DONE]", bright=True)
