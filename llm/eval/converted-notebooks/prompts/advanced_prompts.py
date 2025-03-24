from jet.llm.ollama.base import Ollama
from llama_index.core import PromptTemplate
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/prompts/advanced_prompts.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Advanced Prompt Techniques (Variable Mappings, Functions)

In this notebook we show some advanced prompt techniques. These features allow you to define more custom/expressive prompts, re-use existing ones, and also express certain operations in fewer lines of code.


We show the following features:
1. Partial formatting
2. Prompt template variable mappings
3. Prompt function mappings
"""

# %pip install llama-index-llms-ollama


"""
## 1. Partial Formatting

Partial formatting (`partial_format`) allows you to partially format a prompt, filling in some variables while leaving others to be filled in later.

This is a nice convenience function so you don't have to maintain all the required prompt variables all the way down to `format`, you can partially format as they come in.

This will create a copy of the prompt template.
"""

qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Please write the answer in the style of {tone_name}
Query: {query_str}
Answer: \
"""

prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

partial_prompt_tmpl = prompt_tmpl.partial_format(tone_name="Shakespeare")

partial_prompt_tmpl.kwargs

fmt_prompt = partial_prompt_tmpl.format(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
print(fmt_prompt)

"""
## 2. Prompt Template Variable Mappings

Template var mappings allow you to specify a mapping from the "expected" prompt keys (e.g. `context_str` and `query_str` for response synthesis), with the keys actually in your template. 

This allows you re-use your existing string templates without having to annoyingly change out the template variables.
"""

qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{my_context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {my_query}
Answer: \
"""

template_var_mappings = {"context_str": "my_context", "query_str": "my_query"}

prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str, template_var_mappings=template_var_mappings
)

fmt_prompt = prompt_tmpl.format(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
print(fmt_prompt)

"""
### 3. Prompt Function Mappings

You can also pass in functions as template variables instead of fixed values.

This allows you to dynamically inject certain values, dependent on other values, during query-time.

Here are some basic examples. We show more advanced examples (e.g. few-shot examples) in our Prompt Engineering for RAG guide.
"""

qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \
"""


def format_context_fn(**kwargs):
    context_list = kwargs["context_str"].split("\n\n")
    fmtted_context = "\n\n".join([f"- {c}" for c in context_list])
    return fmtted_context


prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str, function_mappings={"context_str": format_context_fn}
)

context_str = """\
In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters.

Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases.

Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models.
"""

fmt_prompt = prompt_tmpl.format(
    context_str=context_str, query_str="How many params does llama 2 have"
)
print(fmt_prompt)

logger.info("\n\n[DONE]", bright=True)
