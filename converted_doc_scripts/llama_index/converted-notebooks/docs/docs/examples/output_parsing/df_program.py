from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core.program import (
DFFullProgram,
DataFrame,
DataFrameRowsOnly,
)
from llama_index.core.program import DFFullProgram, DFRowsProgram
from llama_index.program.openai import OllamaFunctionCallingAdapterPydanticProgram
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/df_program.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# DataFrame Structured Data Extraction

This demo shows how you can extract tabular DataFrames from raw text.

This was directly inspired by jxnl's dataframe example here: https://github.com/jxnl/openai_function_call/blob/main/auto_dataframe.py.

We show this with different levels of complexity, all backed by the OllamaFunctionCallingAdapter Function API:
- (more code) How to build an extractor yourself using our OllamaFunctionCallingAdapterPydanticProgram
- (less code) Using our out-of-the-box `DFFullProgram` and `DFRowsProgram` objects

## Build a DF Extractor Yourself (Using OllamaFunctionCallingAdapterPydanticProgram)

Our OllamaFunctionCallingAdapterPydanticProgram is a wrapper around an OllamaFunctionCallingAdapter LLM that supports function calling - it will return structured
outputs in the form of a Pydantic object.

We import our `DataFrame` and `DataFrameRowsOnly` objects.

To create an output extractor, you just need to 1) specify the relevant Pydantic object, and 2) Add the right prompt

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# DataFrame Structured Data Extraction")

# %pip install llama-index-llms-ollama
# %pip install llama-index-program-openai

# !pip install llama-index


program = OllamaFunctionCallingAdapterPydanticProgram.from_defaults(
    output_cls=DataFrame,
    llm=OllamaFunctionCallingAdapter(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096),
    prompt_template_str=(
        "Please extract the following query into a structured data according"
        " to: {input_str}.Please extract both the set of column names and a"
        " set of rows."
    ),
    verbose=True,
)

response_obj = program(
    input_str="""My name is John and I am 25 years old. I live in
        New York and I like to play basketball. His name is
        Mike and he is 30 years old. He lives in San Francisco
        and he likes to play baseball. Sarah is 20 years old
        and she lives in Los Angeles. She likes to play tennis.
        Her name is Mary and she is 35 years old.
        She lives in Chicago."""
)
response_obj

program = OllamaFunctionCallingAdapterPydanticProgram.from_defaults(
    output_cls=DataFrameRowsOnly,
    llm=OllamaFunctionCallingAdapter(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096),
    prompt_template_str=(
        "Please extract the following text into a structured data:"
        " {input_str}. The column names are the following: ['Name', 'Age',"
        " 'City', 'Favorite Sport']. Do not specify additional parameters that"
        " are not in the function schema. "
    ),
    verbose=True,
)

program(
    input_str="""My name is John and I am 25 years old. I live in
        New York and I like to play basketball. His name is
        Mike and he is 30 years old. He lives in San Francisco
        and he likes to play baseball. Sarah is 20 years old
        and she lives in Los Angeles. She likes to play tennis.
        Her name is Mary and she is 35 years old.
        She lives in Chicago."""
)

"""
## Use our DataFrame Programs

We provide convenience wrappers for `DFFullProgram` and `DFRowsProgram`. This allows a simpler object creation interface than specifying all details through the `OllamaFunctionCallingAdapterPydanticProgram`.
"""
logger.info("## Use our DataFrame Programs")


df = pd.DataFrame(
    {
        "Name": pd.Series(dtype="str"),
        "Age": pd.Series(dtype="int"),
        "City": pd.Series(dtype="str"),
        "Favorite Sport": pd.Series(dtype="str"),
    }
)

df_rows_program = DFRowsProgram.from_defaults(
    pydantic_program_cls=OllamaFunctionCallingAdapterPydanticProgram, df=df
)

result_obj = df_rows_program(
    input_str="""My name is John and I am 25 years old. I live in
        New York and I like to play basketball. His name is
        Mike and he is 30 years old. He lives in San Francisco
        and he likes to play baseball. Sarah is 20 years old
        and she lives in Los Angeles. She likes to play tennis.
        Her name is Mary and she is 35 years old.
        She lives in Chicago."""
)

result_obj.to_df(existing_df=df)

df_full_program = DFFullProgram.from_defaults(
    pydantic_program_cls=OllamaFunctionCallingAdapterPydanticProgram,
)

result_obj = df_full_program(
    input_str="""My name is John and I am 25 years old. I live in
        New York and I like to play basketball. His name is
        Mike and he is 30 years old. He lives in San Francisco
        and he likes to play baseball. Sarah is 20 years old
        and she lives in Los Angeles. She likes to play tennis.
        Her name is Mary and she is 35 years old.
        She lives in Chicago."""
)

result_obj.to_df()

df = pd.DataFrame(
    {
        "City": pd.Series(dtype="str"),
        "State": pd.Series(dtype="str"),
        "Population": pd.Series(dtype="int"),
    }
)

df_rows_program = DFRowsProgram.from_defaults(
    pydantic_program_cls=OllamaFunctionCallingAdapterPydanticProgram, df=df
)

input_text = """San Francisco is in California, has a population of 800,000.
New York City is the most populous city in the United States. \
With a 2020 population of 8,804,190 distributed over 300.46 square miles (778.2 km2), \
New York City is the most densely populated major city in the United States.
New York City is in New York State.
Boston (US: /ˈbɔːstən/),[8] officially the City of Boston, is the capital and largest city of the Commonwealth of Massachusetts \
and the cultural and financial center of the New England region of the Northeastern United States. \
The city boundaries encompass an area of about 48.4 sq mi (125 km2)[9] and a population of 675,647 as of 2020.[4]
"""

result_obj = df_rows_program(input_str=input_text)

new_df = result_obj.to_df(existing_df=df)
new_df

logger.info("\n\n[DONE]", bright=True)