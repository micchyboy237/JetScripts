from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/prompts/prompt_mixin.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Accessing/Customizing Prompts within Higher-Level Modules

LlamaIndex contains a variety of higher-level modules (query engines, response synthesizers, retrievers, etc.), many of which make LLM calls + use prompt templates.

This guide shows how you can 1) access the set of prompts for any module (including nested) with `get_prompts`, and 2) update these prompts easily with `update_prompts`.
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# !pip install llama-index

import os
import openai

# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
from IPython.display import Markdown, display

"""
## Setup: Load Data, Build Index, and Get Query Engine

Here we build a vector index over a toy dataset (PG's essay), and access the query engine.

The query engine is a simple RAG pipeline consisting of top-k retrieval + LLM synthesis.
"""

"""
Download Data
"""

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(response_mode="tree_summarize")

def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))

"""
## Accessing Prompts

Here we get the prompts from the query engine. Note that *all* prompts are returned, including ones used in sub-modules in the query engine. This allows you to centralize a view of these prompts!
"""

prompts_dict = query_engine.get_prompts()

display_prompt_dict(prompts_dict)

"""
#### Checking `get_prompts` on Response Synthesizer

You can also call `get_prompts` on the underlying response synthesizer, where you'll see the same list.
"""

prompts_dict = query_engine.response_synthesizer.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Checking `get_prompts` with a different response synthesis strategy

Here we try the default `compact` method.

We'll see that the set of templates used are different; a QA template and a refine template.
"""

query_engine = index.as_query_engine(response_mode="compact")

prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Put into query engine, get response
"""

response = query_engine.query("What did the author do growing up?")
print(str(response))

"""
## Customize the prompt

You can also update/customize the prompts with the `update_prompts` function. Pass in arg values with the keys equal to the keys you see in the prompt dictionary.

Here we'll change the summary prompt to use Shakespeare.
"""

from llama_index.core import PromptTemplate

query_engine = index.as_query_engine(response_mode="tree_summarize")

new_summary_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the style of a Shakespeare play.\n"
    "Query: {query_str}\n"
    "Answer: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:summary_template": new_summary_tmpl}
)

prompts_dict = query_engine.get_prompts()

display_prompt_dict(prompts_dict)

response = query_engine.query("What did the author do growing up?")
print(str(response))

"""
## Accessing Prompts from Other Modules

Here we take a look at some other modules: query engines, routers/selectors, evaluators, and others.
"""

from llama_index.core.query_engine import (
    RouterQueryEngine,
    FLAREInstructQueryEngine,
)
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.evaluation import FaithfulnessEvaluator, DatasetGenerator
from llama_index.core.postprocessor import LLMRerank

"""
#### Analyze Prompts: Router Query Engine
"""

from llama_index.core.tools import QueryEngineTool

query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine, description="test description"
)

router_query_engine = RouterQueryEngine.from_defaults([query_tool])

prompts_dict = router_query_engine.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: FLARE Query Engine
"""

flare_query_engine = FLAREInstructQueryEngine(query_engine)

prompts_dict = flare_query_engine.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: LLMMultiSelector
"""

from llama_index.core.selectors import LLMSingleSelector

selector = LLMSingleSelector.from_defaults()

prompts_dict = selector.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: FaithfulnessEvaluator
"""

evaluator = FaithfulnessEvaluator()

prompts_dict = evaluator.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: DatasetGenerator
"""

dataset_generator = DatasetGenerator.from_documents(documents)

prompts_dict = dataset_generator.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: LLMRerank
"""

llm_rerank = LLMRerank()

prompts_dict = dataset_generator.get_prompts()
display_prompt_dict(prompts_dict)

logger.info("\n\n[DONE]", bright=True)