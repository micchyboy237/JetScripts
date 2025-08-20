from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts.default_prompts import (
DEFAULT_TEXT_QA_PROMPT_TMPL,
)
from llama_index.output_parsers.guardrails import GuardrailsOutputParser
from pydantic import BaseModel, Field
from typing import List
import guardrails as gd
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/GuardrailsDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Guardrails Output Parsing

First, set your openai api keys
"""
logger.info("# Guardrails Output Parsing")



"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.")

# %pip install llama-index-llms-ollama
# %pip install llama-index-output-parsers-guardrails

# %pip install guardrails-ai

"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !curl 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' > 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load documents, build the VectorStoreIndex
"""
logger.info("#### Load documents, build the VectorStoreIndex")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents, chunk_size=512)

"""
#### Define Query + Guardrails Spec
"""
logger.info("#### Define Query + Guardrails Spec")


"""
**Define custom QA and Refine Prompts**

**Define Guardrails Spec**
"""



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

llm = MLX(output_parser=output_parser)


fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
logger.debug(fmt_qa_tmpl)

"""
#### Query Index
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine(
    llm=llm,
)
response = query_engine.query(
    "What are the three items the author did growing up?",
)

logger.debug(response)

guard.history.last.tree

logger.info("\n\n[DONE]", bright=True)