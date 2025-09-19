from jet.models.config import MODELS_CACHE_DIR
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.schema import TextNode
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/prompts/advanced_prompts.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Advanced Prompt Techniques (Variable Mappings, Functions)

In this notebook we show some advanced prompt techniques. These features allow you to define more custom/expressive prompts, re-use existing ones, and also express certain operations in fewer lines of code.


We show the following features:
1. Partial formatting
2. Prompt template variable mappings
3. Prompt function mappings
4. Dynamic few-shot examples
"""
logger.info("# Advanced Prompt Techniques (Variable Mappings, Functions)")

# %pip install llama-index-llms-ollama

"""
## 1. Partial Formatting

Partial formatting (`partial_format`) allows you to partially format a prompt, filling in some variables while leaving others to be filled in later.

This is a nice convenience function so you don't have to maintain all the required prompt variables all the way down to `format`, you can partially format as they come in.

This will create a copy of the prompt template.
"""
logger.info("## 1. Partial Formatting")


qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{{ context_str }}
---------------------
Given the context information and not prior knowledge, answer the query.
Please write the answer in the style of {{ tone_name }}
Query: {{ query_str }}
Answer: \
"""

prompt_tmpl = RichPromptTemplate(qa_prompt_tmpl_str)

partial_prompt_tmpl = prompt_tmpl.partial_format(tone_name="Shakespeare")

partial_prompt_tmpl.kwargs

fmt_prompt = partial_prompt_tmpl.format(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
logger.debug(fmt_prompt)

"""
We can also use `format_messages` to format the prompt into `ChatMessage` objects.
"""
logger.info("We can also use `format_messages` to format the prompt into `ChatMessage` objects.")

fmt_prompt = partial_prompt_tmpl.format_messages(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
logger.debug(fmt_prompt)

"""
## 2. Prompt Template Variable Mappings

Template var mappings allow you to specify a mapping from the "expected" prompt keys (e.g. `context_str` and `query_str` for response synthesis), with the keys actually in your template. 

This allows you re-use your existing string templates without having to annoyingly change out the template variables.
"""
logger.info("## 2. Prompt Template Variable Mappings")


qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{{ my_context }}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {{ my_query }}
Answer: \
"""

template_var_mappings = {"context_str": "my_context", "query_str": "my_query"}

prompt_tmpl = RichPromptTemplate(
    qa_prompt_tmpl_str, template_var_mappings=template_var_mappings
)

fmt_prompt = prompt_tmpl.format(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
logger.debug(fmt_prompt)

"""
### 3. Prompt Function Mappings

You can also pass in functions as template variables instead of fixed values.

This allows you to dynamically inject certain values, dependent on other values, during query-time.

Here are some basic examples. We show more advanced examples (e.g. few-shot examples) in our Prompt Engineering for RAG guide.
"""
logger.info("### 3. Prompt Function Mappings")


qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{{ context_str }}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {{ query_str }}
Answer: \
"""


def format_context_fn(**kwargs):
    context_list = kwargs["context_str"].split("\n\n")
    fmtted_context = "\n\n".join([f"- {c}" for c in context_list])
    return fmtted_context


prompt_tmpl = RichPromptTemplate(
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
logger.debug(fmt_prompt)

"""
### 4. Dynamic few-shot examples

Using the function mappings, you can also dynamically inject few-shot examples based on other prompt variables.

Here's an example that uses a vector store to dynamically inject few-shot text-to-sql examples based on the query.

First, lets define a text-to-sql prompt template.
"""
logger.info("### 4. Dynamic few-shot examples")

text_to_sql_prompt_tmpl_str = """\
You are a SQL expert. You are given a natural language query, and your job is to convert it into a SQL query.

Here are some examples of how you should convert natural language to SQL:
<examples>
{{ examples }}
</examples>

Now it's your turn.

Query: {{ query_str }}
SQL:
"""

"""
Given this prompt template, lets define and index some few-shot text-to-sql examples.
"""
logger.info("Given this prompt template, lets define and index some few-shot text-to-sql examples.")


# os.environ["OPENAI_API_KEY"] = "sk-..."


Settings.llm = OllamaFunctionCalling(model="llama3.2")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

example_nodes = [
    TextNode(
        text="Query: How many params does llama 2 have?\nSQL: SELECT COUNT(*) FROM llama_2_params;"
    ),
    TextNode(
        text="Query: How many layers does llama 2 have?\nSQL: SELECT COUNT(*) FROM llama_2_layers;"
    ),
]

index = VectorStoreIndex(nodes=example_nodes)

retriever = index.as_retriever(similarity_top_k=1)

"""
With our retriever, we can create our prompt template with function mappings to dynamically inject few-shot examples based on the query.
"""
logger.info("With our retriever, we can create our prompt template with function mappings to dynamically inject few-shot examples based on the query.")



def get_examples_fn(**kwargs):
    query = kwargs["query_str"]
    examples = retriever.retrieve(query)
    return "\n\n".join(node.text for node in examples)


prompt_tmpl = RichPromptTemplate(
    text_to_sql_prompt_tmpl_str,
    function_mappings={"examples": get_examples_fn},
)

prompt = prompt_tmpl.format(
    query_str="What are the number of parameters in the llama 2 model?"
)
logger.debug(prompt)

response = Settings.llm.complete(prompt)
logger.debug(response.text)

logger.info("\n\n[DONE]", bright=True)