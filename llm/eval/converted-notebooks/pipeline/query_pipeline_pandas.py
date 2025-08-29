import pandas as pd
from llama_index.core import PromptTemplate
from jet.llm.ollama.base import Ollama
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Query Pipeline over Pandas DataFrames
#
# This is a simple example that builds a query pipeline that can perform structured operations over a Pandas DataFrame to satisfy a user query, using LLMs to infer the set of operations.
#
# This can be treated as the "from-scratch" version of our `PandasQueryEngine`.
#
# WARNING: This tool provides the LLM access to the `eval` function.
# Arbitrary code execution is possible on the machine running this tool.
# This tool is not recommended to be used in a production setting, and would
# require heavy sandboxing or virtual machines.

# %pip install llama-index-llms-ollama llama-index-experimental


# Download Data
#
# Here we load the Titanic CSV dataset.

# !wget 'https://raw.githubusercontent.com/jerryjliu/llama_index/main/docs/docs/examples/data/csv/titanic_train.csv' -O 'titanic_train.csv'


df = pd.read_csv("./titanic_train.csv")

# Define Modules
#
# Here we define the set of modules:
# 1. Pandas prompt to infer pandas instructions from user query
# 2. Pandas output parser to execute pandas instructions on dataframe, get back dataframe
# 3. Response synthesis prompt to synthesize a final response given the dataframe
# 4. LLM
#
# The pandas output parser specifically is designed to safely execute Python code. It includes a lot of safety checks that may be annoying to write from scratch. This includes only importing from a set of approved modules (e.g. no modules that would alter the file system like `os`), and also making sure that no private/dunder methods are being called.

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = Ollama(model="llama3.2")

# Build Query Pipeline
#
# Looks like this:
# input query_str -> pandas_prompt -> llm1 -> pandas_output_parser -> response_synthesis_prompt -> llm2
#
# Additional connections to response_synthesis_prompt: llm1 -> pandas_instructions, and pandas_output_parser -> pandas_output.

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
    ]
)
qp.add_link("response_synthesis_prompt", "llm2")

# Run Query

response = qp.run(
    query_str="What is the correlation between survival and age?",
)

print(response.message.content)

logger.info("\n\n[DONE]", bright=True)
