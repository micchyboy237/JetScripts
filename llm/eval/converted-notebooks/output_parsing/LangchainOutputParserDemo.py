from jet.llm.ollama.base import Ollama
from llama_index.core.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from llama_index.core.output_parsers import LangchainOutputParser
import os
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import sys
import logging
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/LangchainOutputParserDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Langchain Output Parsing

# Download Data

# %pip install llama-index-llms-ollama

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# Load documents, build the VectorStoreIndex


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# os.environ["OPENAI_API_KEY"] = "sk-..."

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries/").load_data()

index = VectorStoreIndex.from_documents(documents, chunk_size=512)

# Define Query + Langchain Output Parser


# **Define custom QA and Refine Prompts**

response_schemas = [
    ResponseSchema(
        name="Education",
        description=(
            "Describes the author's educational experience/background."
        ),
    ),
    ResponseSchema(
        name="Work",
        description="Describes the author's work experience/background.",
    ),
]

lc_output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)
output_parser = LangchainOutputParser(lc_output_parser)


fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
print(fmt_qa_tmpl)

# Query Index


llm = Ollama(output_parser=output_parser)

query_engine = index.as_query_engine(
    llm=llm,
)
response = query_engine.query(
    "What are a few things the author did growing up?",
)

print(response)

logger.info("\n\n[DONE]", bright=True)
