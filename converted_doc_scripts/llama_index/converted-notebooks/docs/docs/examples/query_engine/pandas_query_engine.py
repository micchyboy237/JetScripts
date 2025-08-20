from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.experimental.query_engine import PandasQueryEngine
import logging
import os
import pandas as pd
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/pandas_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pandas Query Engine

This guide shows you how to use our `PandasQueryEngine`: convert natural language to Pandas python code using LLMs.

The input to the `PandasQueryEngine` is a Pandas dataframe, and the output is a response. The LLM infers dataframe operations to perform in order to retrieve the result.

**WARNING:** This tool provides the LLM access to the `eval` function.
Arbitrary code execution is possible on the machine running this tool.
While some level of filtering is done on code, this tool is not recommended 
to be used in a production setting without heavy sandboxing or virtual machines.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Pandas Query Engine")

# !pip install llama-index llama-index-experimental




logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
## Let's start on a Toy DataFrame

Here let's load a very simple dataframe containing city and population pairs, and run the `PandasQueryEngine` on it.

By setting `verbose=True` we can see the intermediate generated instructions.
"""
logger.info("## Let's start on a Toy DataFrame")

df = pd.DataFrame(
    {
        "city": ["Toronto", "Tokyo", "Berlin"],
        "population": [2930000, 13960000, 3645000],
    }
)

query_engine = PandasQueryEngine(df=df, verbose=True)

response = query_engine.query(
    "What is the city with the highest population?",
)

display(Markdown(f"<b>{response}</b>"))

logger.debug(response.metadata["pandas_instruction_str"])

"""
We can also take the step of using an LLM to synthesize a response.
"""
logger.info("We can also take the step of using an LLM to synthesize a response.")

query_engine = PandasQueryEngine(df=df, verbose=True, synthesize_response=True)
response = query_engine.query(
    "What is the city with the highest population? Give both the city and population",
)
logger.debug(str(response))

"""
## Analyzing the Titanic Dataset

The Titanic dataset is one of the most popular tabular datasets in introductory machine learning
Source: https://www.kaggle.com/c/titanic

#### Download Data
"""
logger.info("## Analyzing the Titanic Dataset")

# !wget 'https://raw.githubusercontent.com/jerryjliu/llama_index/main/docs/docs/examples/data/csv/titanic_train.csv' -O 'titanic_train.csv'

df = pd.read_csv("./titanic_train.csv")

query_engine = PandasQueryEngine(df=df, verbose=True)

response = query_engine.query(
    "What is the correlation between survival and age?",
)

display(Markdown(f"<b>{response}</b>"))

logger.debug(response.metadata["pandas_instruction_str"])

"""
## Additional Steps

### Analyzing / Modifying prompts

Let's look at the prompts!
"""
logger.info("## Additional Steps")


query_engine = PandasQueryEngine(df=df, verbose=True)
prompts = query_engine.get_prompts()
logger.debug(prompts["pandas_prompt"].template)

logger.debug(prompts["response_synthesis_prompt"].template)

"""
You can update prompts as well:
"""
logger.info("You can update prompts as well:")

new_prompt = PromptTemplate(
    """\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `logger.debug(df.head())`:
{df_str}

Follow these instructions:
{instruction_str}
Query: {query_str}

Expression: """
)

query_engine.update_prompts({"pandas_prompt": new_prompt})

"""
This is the instruction string (that you can customize by passing in `instruction_str` on initialization)
"""
logger.info("This is the instruction string (that you can customize by passing in `instruction_str` on initialization)")

instruction_str = """\
1. Convert the query to executable Python code using Pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code should represent a solution to the query.
4. PRINT ONLY THE EXPRESSION.
5. Do not quote the expression.
"""

"""
### Implementing Query Engine using Query Pipeline Syntax

If you want to learn to construct your own Pandas Query Engine using our Query Pipeline syntax and the prompt components above, check out our below tutorial.

[Setting up a Pandas DataFrame query engine with Query Pipelines](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_pandas.html)
"""
logger.info("### Implementing Query Engine using Query Pipeline Syntax")

logger.info("\n\n[DONE]", bright=True)