from llama_index.core.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
)
from jet.llm.ollama.base import Ollama
import guardrails as gd
from typing import List
from pydantic import BaseModel, Field
from llama_index.output_parsers.guardrails import GuardrailsOutputParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import sys
import logging
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/GuardrailsDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Guardrails Output Parsing

# First, set your openai api keys


# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-llms-ollama
# %pip install llama-index-output-parsers-guardrails

# %pip install guardrails-ai

# Download Data

# !mkdir -p 'data/paul_graham/'
# !curl 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' > 'data/paul_graham/paul_graham_essay.txt'

# Load documents, build the VectorStoreIndex


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents, chunk_size=512)

# Define Query + Guardrails Spec


# **Define custom QA and Refine Prompts**

# **Define Guardrails Spec**


class BulletPoints(BaseModel):
    explanation: str = Field()
    explanation2: str = Field()
    explanation3: str = Field()


class Explanation(BaseModel):
    points: BulletPoints = Field(
        description="Bullet points regarding events in the author's life."
    )


prompt = """
Query string here.

${gr.xml_prefix_prompt}

${output_schema}

${gr.json_suffix_prompt_v2_wo_none}
"""


guard = gd.Guard.from_pydantic(output_class=Explanation, prompt=prompt)

output_parser = GuardrailsOutputParser(guard)

llm = Ollama(output_parser=output_parser)


fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
print(fmt_qa_tmpl)

# Query Index

query_engine = index.as_query_engine(
    llm=llm,
)
response = query_engine.query(
    "What are the three items the author did growing up?",
)

print(response)

guard.history.last.tree

logger.info("\n\n[DONE]", bright=True)
