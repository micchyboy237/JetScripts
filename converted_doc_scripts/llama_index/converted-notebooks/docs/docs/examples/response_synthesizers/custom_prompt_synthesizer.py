from jet.logger import CustomLogger
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import TreeSummarize, Refine
from llama_index.core.types import BaseModel
from typing import List
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/response_synthesizers/custom_prompt_synthesizer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pydantic Tree Summarize

In this notebook, we demonstrate how to use tree summarize with structured outputs. Specifically, tree summarize is used to output pydantic objects.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Pydantic Tree Summarize")

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data
"""
logger.info("## Load Data")


reader = SimpleDirectoryReader(
    input_files=["/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/paul_graham_essay.txt"]
)

docs = reader.load_data()

text = docs[0].text

"""
## Define Custom Prompt
"""
logger.info("## Define Custom Prompt")


qa_prompt_tmpl = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Please also write the answer in the style of {tone_name}.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(qa_prompt_tmpl)

refine_prompt_tmpl = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "Please also write the answer in the style of {tone_name}.\n"
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
refine_prompt = PromptTemplate(refine_prompt_tmpl)

"""
## Try out Response Synthesis with Custom Prompt

We try out a few different response synthesis strategies with the custom prompt.
"""
logger.info("## Try out Response Synthesis with Custom Prompt")


summarizer = TreeSummarize(verbose=True, summary_template=qa_prompt)

response = summarizer.get_response(
    "who is Paul Graham?", [text], tone_name="a Shakespeare play"
)

logger.debug(str(response))

summarizer = Refine(
    verbose=True, text_qa_template=qa_prompt, refine_template=refine_prompt
)

response = summarizer.get_response(
    "who is Paul Graham?", [text], tone_name="a haiku"
)

logger.debug(str(response))

class Biography(BaseModel):
    """Data model for a biography."""

    name: str
    best_known_for: List[str]
    extra_info: str

summarizer = TreeSummarize(
    verbose=True, summary_template=qa_prompt, output_cls=Biography
)

response = summarizer.get_response(
    "who is Paul Graham?", [text], tone_name="a business memo"
)

logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)