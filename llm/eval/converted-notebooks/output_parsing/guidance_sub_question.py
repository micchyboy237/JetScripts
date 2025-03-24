from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import QueryBundle
from llama_index.core.tools import ToolMetadata
from guidance.llms import Ollama as GuidanceOllama
from llama_index.question_gen.guidance import GuidanceQuestionGenerator
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/guidance_sub_question.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Guidance for Sub-Question Query Engine

# In this notebook, we showcase how to use [**guidance**](https://github.com/microsoft/guidance) to improve the robustness of our sub-question query engine.

# The sub-question query engine is designed to accept swappable question generators that implement the `BaseQuestionGenerator` interface.
# To leverage the power of [**guidance**](https://github.com/microsoft/guidance), we implemented a new `GuidanceQuestionGenerator` (powered by our `GuidancePydanticProgram`)

# Guidance Question Generator

# Unlike the default `LLMQuestionGenerator`, guidance guarantees that we will get the desired structured output, and eliminate output parsing errors.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-question-gen-guidance

# !pip install llama-index


question_gen = GuidanceQuestionGenerator.from_defaults(
    guidance_llm=GuidanceOllama("text-davinci-003"), verbose=False
)

# Let's test it out!


tools = [
    ToolMetadata(
        name="lyft_10k",
        description="Provides information about Lyft financials for year 2021",
    ),
    ToolMetadata(
        name="uber_10k",
        description="Provides information about Uber financials for year 2021",
    ),
]

sub_questions = question_gen.generate(
    tools=tools,
    query=QueryBundle("Compare and contrast Uber and Lyft financial in 2021"),
)

sub_questions

# Using Guidance Question Generator with Sub-Question Query Engine

# Prepare data and base query engines


# Download Data

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

lyft_docs = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

# Construct sub-question query engine and run some queries!

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021"
            ),
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(
    question_gen=question_gen,  # use guidance based question_gen defined above
    query_engine_tools=query_engine_tools,
)

response = s_engine.query(
    "Compare and contrast the customer segments and geographies that grew the"
    " fastest"
)

print(response)

logger.info("\n\n[DONE]", bright=True)
