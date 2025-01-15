"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/pydantic_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Query Engine with Pydantic Outputs

Every query engine has support for integrated structured responses using the following `response_mode`s in `RetrieverQueryEngine`:
- `refine`
- `compact`
- `tree_summarize`
- `accumulate` (beta, requires extra parsing to convert to objects)
- `compact_accumulate` (beta, requires extra parsing to convert to objects)

In this notebook, we walk through a small example demonstrating the usage.

Under the hood, every LLM response will be a pydantic object. If that response needs to be refined or summarized, it is converted into a JSON string for the next response. Then, the final response is returned as a pydantic object.

**NOTE:** This can technically work with any LLM, but non-openai is support is still in development and considered beta.
"""

"""
## Setup
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-llms-anthropic
# %pip install llama-index-llms-ollama

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
Download Data
"""

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


from llama_index.core import VectorStoreIndex,
from llama_index.llms.anthropic import Anthropic
from jet.llm.ollama import Ollama
from llama_index.core import VectorStoreIndex
from pydantic import BaseModel
from typing import List
import os
import openai
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()

"""
### Create our Pydanitc Output Object
"""


class Biography(BaseModel):
    """Data model for a biography."""

    name: str
    best_known_for: List[str]
    extra_info: str


"""
## Create the Index + Query Engine (Ollama)

When using Ollama, the function calling API will be leveraged for reliable structured outputs.
"""


llm = Ollama(model="llama3.2", request_timeout=300.0,
             context_window=4096, temperature=0.1)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(
    output_cls=Biography, response_mode="compact", llm=llm
)

response = query_engine.query("Who is Paul Graham?")

print(response.name)
print(response.best_known_for)
print(response.extra_info)

print(type(response.response))

"""
## Create the Index + Query Engine (Non-Ollama, Beta)

When using an LLM that does not support function calling, we rely on the LLM to write the JSON itself, and we parse the JSON into the proper pydantic object.
"""


# os.environ["ANTHROPIC_API_KEY"] = "sk-..."


llm = Anthropic(model="claude-instant-1.2", temperature=0.1)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(
    output_cls=Biography, response_mode="tree_summarize", llm=llm
)

response = query_engine.query("Who is Paul Graham?")

print(response.name)
print(response.best_known_for)
print(response.extra_info)

print(type(response.response))

"""
## Accumulate Examples (Beta)

Accumulate with pydantic objects requires some extra parsing. This is still a beta feature, but it's still possible to get accumulate pydantic objects.
"""


class Company(BaseModel):
    """Data model for a companies mentioned."""

    company_name: str
    context_info: str


llm = Ollama(model="llama3.2", request_timeout=300.0,
             context_window=4096, temperature=0.1)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(
    output_cls=Company, response_mode="accumulate", llm=llm
)

response = query_engine.query("What companies are mentioned in the text?")

"""
In accumulate, responses are separated by a default separator, and prepended with a prefix.
"""

companies = []

for response_str in str(response).split("\n---------------------\n"):
    response_str = response_str[response_str.find("{"):]
    companies.append(Company.parse_raw(response_str))

print(companies)
