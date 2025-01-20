from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent import ReActChatFormatter
from llama_index.core import QueryBundle
from llama_index.question_gen.openai import OpenAIQuestionGenerator
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from jet.llm.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from IPython.display import Markdown, display
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

"""
# Query Transform Cookbook 

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_transformations/query_transform_cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A user query can be transformed and decomposed in many ways before being executed as part of a RAG query engine, agent, or any other pipeline.

In this guide we show you different ways to transform, decompose queries, and find the set of relevant tools. Each technique might be applicable for different use cases!

For naming purposes, we define the underlying pipeline as a "tool". Here are the different query transformations:

1. **Routing**: Keep the query, but identify the relevant subset of tools that the query applies to. Output those tools as the relevant choices.
2. **Query-Rewriting**: Keep the tools, but rewrite the query in a variety of different ways to execute against the same tools.
3. **Sub-Questions**: Decompose queries into multiple sub-questions over different tools (identified by their metadata).
4. **ReAct Agent Tool Picking**: Given the initial query, identify 1) the tool to pick, and 2) the query to execute on the tool.

The goal of this guide is to show you how to use these query transforms as **modular** components. Of course, each of these components plug into a bigger system (e.g. the sub-question generator is a part of our `SubQuestionQueryEngine`) - and the guides for each of these are linked below.

Take a look and let us know your thoughts!
"""

# %pip install llama-index-question-gen-openai
# %pip install llama-index-llms-ollama


def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))


"""
## Routing

In this example, we show how a query can be used to select the set of relevant tool choices. 

We use our `selector` abstraction to pick the relevant tool(s) - it can be a single tool, or a multiple tool depending on the abstraction.

We have four selectors: combination of (LLM or function calling) x (single selection or multi-selection)
"""


selector = LLMMultiSelector.from_defaults()


tool_choices = [
    ToolMetadata(
        name="covid_nyt",
        description=("This tool contains a NYT news article about COVID-19"),
    ),
    ToolMetadata(
        name="covid_wiki",
        description=("This tool contains the Wikipedia page about COVID-19"),
    ),
    ToolMetadata(
        name="covid_tesla",
        description=("This tool contains the Wikipedia page about apples"),
    ),
]

display_prompt_dict(selector.get_prompts())

selector_result = selector.select(
    tool_choices, query="Tell me more about COVID-19"
)

selector_result.selections

"""
Learn more about our routing abstractions in our [dedicated Router page](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/root.html).
"""

"""
## Query Rewriting

In this section, we show you how to rewrite queries into multiple queries. You can then execute all these queries against a retriever. 

This is a key step in advanced retrieval techniques. By doing query rewriting, you can generate multiple queries for [ensemble retrieval] and [fusion], leading to higher-quality retrieved results.

Unlike the sub-question generator, this is just a prompt call, and exists independently of tools.
"""

"""
### Query Rewriting (Custom)

Here we show you how to use a prompt to generate multiple queries, using our LLM and prompt abstractions.
"""


query_gen_str = """\
You are a helpful assistant that generates multiple search queries based on a \
single input query. Generate {num_queries} search queries, one on each line, \
related to the following input query:
Query: {query}
Queries:
"""
query_gen_prompt = PromptTemplate(query_gen_str)

llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)


def generate_queries(query: str, llm, num_queries: int = 4):
    response = llm.predict(
        query_gen_prompt, num_queries=num_queries, query=query
    )
    queries = response.split("\n")
    queries_str = "\n".join(queries)
    print(f"Generated queries:\n{queries_str}")
    return queries


queries = generate_queries("What happened at Interleaf and Viaweb?", llm)

queries

"""
For more details about an e2e implementation with a retriever, check out our guides on our fusion retriever:
- [Module Guide](https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion.html)
- [Build a Fusion Retriever from Scratch](https://docs.llamaindex.ai/en/latest/examples/low_level/fusion_retriever.html)
"""

"""
### Query Rewriting (using QueryTransform)

In this section we show you how to do query transformations using our QueryTransform class.
"""


hyde = HyDEQueryTransform(include_original=True)
llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)

query_bundle = hyde.run("What is Bel?")

"""
This generates a query bundle that contains the original query, but also `custom_embedding_strs` representing the queries that should be embedded.
"""

# new_query.custom_embedding_strs

"""
## Sub-Questions

Given a set of tools and a user query, decide both the 1) set of sub-questions to generate, and 2) the tools that each sub-question should run over.

We run through an example using the `OpenAIQuestionGenerator`, which depends on function calling, and also the `LLMQuestionGenerator`, which depends on prompting.
"""


llm = Ollama()
question_gen = OpenAIQuestionGenerator.from_defaults(llm=llm)

display_prompt_dict(question_gen.get_prompts())


tool_choices = [
    ToolMetadata(
        name="uber_2021_10k",
        description=(
            "Provides information about Uber financials for year 2021"
        ),
    ),
    ToolMetadata(
        name="lyft_2021_10k",
        description=(
            "Provides information about Lyft financials for year 2021"
        ),
    ),
]


query_str = "Compare and contrast Uber and Lyft"
choices = question_gen.generate(tool_choices, QueryBundle(query_str=query_str))

"""
The outputs are `SubQuestion` Pydantic objects.
"""

choices

"""
For details on how to plug this into your RAG pipeline in a more packaged fashion, check out our [SubQuestionQueryEngine](https://docs.llamaindex.ai/en/latest/examples/query_engine/sub_question_query_engine.html).
"""

"""
## Query Transformation with ReAct Prompt

ReAct is a popular framework for agents, and here we show how the core ReAct prompt can be used to transform queries.

We use the `ReActChatFormatter` to get the set of input messages for the LLM.
"""


def execute_sql(sql: str) -> str:
    """Given a SQL input string, execute it."""
    return f"Executed {sql}"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tool1 = FunctionTool.from_defaults(fn=execute_sql)
tool2 = FunctionTool.from_defaults(fn=add)
tools = [tool1, tool2]

"""
Here we get the input prompt messages to pass to the LLM. Take a look!
"""

chat_formatter = ReActChatFormatter()
output_parser = ReActOutputParser()
input_msgs = chat_formatter.format(
    tools,
    [
        ChatMessage(
            content="Can you find the top three rows from the table named `revenue_years`",
            role="user",
        )
    ],
)
input_msgs

"""
Next we get the output from the model.
"""

llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)

response = llm.chat(input_msgs)

"""
Finally we use our ReActOutputParser to parse the content into a structured output, and analyze the action inputs.
"""

reasoning_step = output_parser.parse(response.message.content)

reasoning_step.action_input

logger.info("\n\n[DONE]", bright=True)
