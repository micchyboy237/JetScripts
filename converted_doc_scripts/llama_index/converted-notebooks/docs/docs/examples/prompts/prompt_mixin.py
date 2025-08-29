from jet.models.config import MODELS_CACHE_DIR
from IPython.display import Markdown, display
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.evaluation import FaithfulnessEvaluator, DatasetGenerator
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.selectors import LLMSingleSelector
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/prompts/prompt_mixin.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Accessing/Customizing Prompts within Higher-Level Modules

LlamaIndex contains a variety of higher-level modules (query engines, response synthesizers, retrievers, etc.), many of which make LLM calls + use prompt templates.

This guide shows how you can 1) access the set of prompts for any module (including nested) with `get_prompts`, and 2) update these prompts easily with `update_prompts`.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Accessing/Customizing Prompts within Higher-Level Modules")

# %pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."


Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2")

"""
## Setup: Load Data, Build Index, and Get Query Engine

Here we build a vector index over a toy dataset (PG's essay), and access the query engine.

The query engine is a simple RAG pipeline consisting of top-k retrieval + LLM synthesis.

Download Data
"""
logger.info("## Setup: Load Data, Build Index, and Get Query Engine")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(response_mode="tree_summarize")



def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        logger.debug(p.get_template())
        display(Markdown("<br><br>"))

"""
## Accessing Prompts

Here we get the prompts from the query engine. Note that *all* prompts are returned, including ones used in sub-modules in the query engine. This allows you to centralize a view of these prompts!
"""
logger.info("## Accessing Prompts")

prompts_dict = query_engine.get_prompts()

display_prompt_dict(prompts_dict)

"""
#### Checking `get_prompts` on Response Synthesizer

You can also call `get_prompts` on the underlying response synthesizer, where you'll see the same list.
"""
logger.info("#### Checking `get_prompts` on Response Synthesizer")

prompts_dict = query_engine.response_synthesizer.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Checking `get_prompts` with a different response synthesis strategy

Here we try the default `compact` method.

We'll see that the set of templates used are different; a QA template and a refine template.
"""
logger.info("#### Checking `get_prompts` with a different response synthesis strategy")

query_engine = index.as_query_engine(response_mode="compact")

prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Put into query engine, get response
"""
logger.info("#### Put into query engine, get response")

response = query_engine.query("What did the author do growing up?")
logger.debug(str(response))

"""
## Customize the prompt

You can also update/customize the prompts with the `update_prompts` function. Pass in arg values with the keys equal to the keys you see in the prompt dictionary.

Here we'll change the summary prompt to use Shakespeare.
"""
logger.info("## Customize the prompt")


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
logger.debug(str(response))

"""
## Accessing Prompts from Other Modules

Here we take a look at some other modules: query engines, routers/selectors, evaluators, and others.
"""
logger.info("## Accessing Prompts from Other Modules")


"""
#### Analyze Prompts: ReActAgent
"""
logger.info("#### Analyze Prompts: ReActAgent")

agent = ReActAgent(tools=[])

prompts_dict = agent.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: FLARE Query Engine
"""
logger.info("#### Analyze Prompts: FLARE Query Engine")

flare_query_engine = FLAREInstructQueryEngine(query_engine)

prompts_dict = flare_query_engine.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: LLMMultiSelector
"""
logger.info("#### Analyze Prompts: LLMMultiSelector")


selector = LLMSingleSelector.from_defaults()

prompts_dict = selector.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: FaithfulnessEvaluator
"""
logger.info("#### Analyze Prompts: FaithfulnessEvaluator")

evaluator = FaithfulnessEvaluator()

prompts_dict = evaluator.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: DatasetGenerator
"""
logger.info("#### Analyze Prompts: DatasetGenerator")

dataset_generator = DatasetGenerator.from_documents(documents)

prompts_dict = dataset_generator.get_prompts()
display_prompt_dict(prompts_dict)

"""
#### Analyze Prompts: LLMRerank
"""
logger.info("#### Analyze Prompts: LLMRerank")

llm_rerank = LLMRerank()

prompts_dict = dataset_generator.get_prompts()
display_prompt_dict(prompts_dict)

logger.info("\n\n[DONE]", bright=True)