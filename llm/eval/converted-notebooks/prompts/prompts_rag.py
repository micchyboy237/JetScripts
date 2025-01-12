from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
from llama_index.core.readers.file.base import SimpleDirectoryReader
initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/prompts/prompts_rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Prompt Engineering for RAG

In this notebook we show various prompt techniques you can try to customize your LlamaIndex RAG pipeline.

- Getting and setting prompts for query engines, etc.
- Defining template variable mappings (e.g. you have an existing QA prompt)
- Adding few-shot examples + performing query transformations/rewriting.
"""

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-file pymupdf

# !pip install llama-index

import os
import openai

# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
## Setup
"""

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
from IPython.display import Markdown, display

"""
#### Load Data
"""

# !mkdir data
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

from pathlib import Path
from llama_index.readers.file import PyMuPDFReader

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()

"""
#### Load into Vector Store
"""

from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

gpt35_llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)
gpt4_llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)

index = VectorStoreIndex.from_documents(documents)

"""
#### Setup Query Engine / Retriever
"""

query_str = "What are the potential risks associated with the use of Llama 2 as mentioned in the context?"

query_engine = index.as_query_engine(similarity_top_k=2, llm=gpt35_llm)
vector_retriever = index.as_retriever(similarity_top_k=2)

response = query_engine.query(query_str)
logger.debug(str(response))

"""
## Viewing/Customizing Prompts

First, let's take a look at the query engine prompts, and see how we can customize it.
"""

"""
### View Prompts
"""

def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        logger.debug(p.get_template())
        display(Markdown("<br><br>"))

prompts_dict = query_engine.get_prompts()

display_prompt_dict(prompts_dict)

"""
### Customize Prompts

What if we want to do something different than our standard question-answering prompts?

Let's try out the RAG prompt from [LangchainHub](https://smith.langchain.com/hub/rlm/rag-prompt)
"""

from langchain import hub

langchain_prompt = hub.pull("rlm/rag-prompt")

"""
One catch is that the template variables in the prompt are different than what's expected by our synthesizer in the query engine:
- the prompt uses `context` and `question`,
- we expect `context_str` and `query_str`

This is not a problem! Let's add our template variable mappings to map variables. We use our `LangchainPromptTemplate` to map to LangChain prompts.
"""

from llama_index.core.prompts import LangchainPromptTemplate

lc_prompt_tmpl = LangchainPromptTemplate(
    template=langchain_prompt,
    template_var_mappings={"query_str": "question", "context_str": "context"},
)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": lc_prompt_tmpl}
)

prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)

"""
### Try It Out

Let's re-run our query engine again.
"""

response = query_engine.query(query_str)
logger.debug(str(response))

"""
## Adding Few-Shot Examples

Let's try adding few-shot examples to the prompt, which can be dynamically loaded depending on the query! 

We do this by setting the `function_mapping` variable in our prompt template - this allows us to compute functions (e.g. return few-shot examples) during prompt formatting time.

As an example use case, through this we can coerce the model to output results in a structured format,
by showing examples of other structured outputs.

Let's parse a pre-generated question/answer file. For the sake of focus we'll skip how the file is generated (tl;dr we used a GPT-4 powered function calling RAG pipeline), but the qa pairs look like this:

```
{"query": "<query>", "response": "<output_json>"}
```

We embed/index these Q/A pairs, and retrieve the top-k.
"""

from llama_index.core.schema import TextNode

few_shot_nodes = []
for line in open("../llama2_qa_citation_events.jsonl", "r"):
    few_shot_nodes.append(TextNode(text=line))

few_shot_index = VectorStoreIndex(few_shot_nodes)
few_shot_retriever = few_shot_index.as_retriever(similarity_top_k=2)

import json


def few_shot_examples_fn(**kwargs):
    query_str = kwargs["query_str"]
    retrieved_nodes = few_shot_retriever.retrieve(query_str)

    result_strs = []
    for n in retrieved_nodes:
        raw_dict = json.loads(n.get_content())
        query = raw_dict["query"]
        response_dict = json.loads(raw_dict["response"])
        result_str = f"""\
Query: {query}
Response: {response_dict}"""
        result_strs.append(result_str)
    return "\n\n".join(result_strs)

qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, \
answer the query asking about citations over different topics.
Please provide your answer in the form of a structured JSON format containing \
a list of authors as the citations. Some examples are given below.

{few_shot_examples}

Query: {query_str}
Answer: \
"""

qa_prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str,
    function_mappings={"few_shot_examples": few_shot_examples_fn},
)

citation_query_str = (
    "Which citations are mentioned in the section on Safety RLHF?"
)

"""
Let's see what the formatted prompt looks like with the few-shot examples function.
(we fill in test context for brevity)
"""

logger.debug(
    qa_prompt_tmpl.format(
        query_str=citation_query_str, context_str="test_context"
    )
)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

display_prompt_dict(query_engine.get_prompts())

response = query_engine.query(citation_query_str)
logger.debug(str(response))

logger.debug(response.source_nodes[1].get_content())

"""
## Context Transformations - PII Example

We can also dynamically add context transformations as functions in the prompt variable. In this example we show how we can process the `context_str` before feeding to the context window - specifically in masking out PII (a step towards alleviating concerns around data privacy/security).

**NOTE**: You can do these as steps before feeding into the prompt as well, but this gives you flexibility to perform all this on the fly for any QA prompt you define!
"""

from llama_index.core.postprocessor import (
    NERPIINodePostprocessor,
    SentenceEmbeddingOptimizer,
)
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode

pii_processor = NERPIINodePostprocessor(llm=gpt4_llm)

def filter_pii_fn(**kwargs):
    query_bundle = QueryBundle(query_str=kwargs["query_str"])

    new_nodes = pii_processor.postprocess_nodes(
        [NodeWithScore(node=TextNode(text=kwargs["context_str"]))],
        query_bundle=query_bundle,
    )
    new_node = new_nodes[0]
    return new_node.get_content()

qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str, function_mappings={"context_str": filter_pii_fn}
)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

retrieved_nodes = vector_retriever.retrieve(query_str)
context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])

logger.debug(qa_prompt_tmpl.format(query_str=query_str, context_str=context_str))

response = query_engine.query(query_str)
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)