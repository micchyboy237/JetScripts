from jet.logger import logger
from langchain.agents import AgentExecutor
from langchain.chains import SequentialChain
from langchain_timbr import (
ExecuteTimbrQueryChain,
GenerateAnswerChain,
TimbrSqlAgent,
LlmWrapper,
LlmTypes,
)
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
---
sidebar_label: timbr
---

# Timbr

[Timbr](https://docs.timbr.ai/doc/docs/integration/langchain-sdk/) integrates natural language inputs with Timbr's ontology-driven semantic layer. Leveraging Timbr's robust ontology capabilities, the SDK integrates with Timbr data models and leverages semantic relationships and annotations, enabling users to query data using business-friendly language.

This notebook provides a quick overview for getting started with Timbr tools and agents. For more information about Timbr visit [Timbr.ai](https://timbr.ai/) or the [Timbr Documentation](https://docs.timbr.ai/doc/docs/integration/langchain-sdk/)

## Overview

### Integration details

Timbr package for LangChain is [langchain-timbr](https://pypi.org/project/langchain-timbr), which provides seamless integration with Timbr's semantic layer for natural language to SQL conversion.

### Tool features

| Tool Name | Description |
| :--- | :--- |
| `IdentifyTimbrConceptChain` | Identify relevant concepts from user prompts |
| `GenerateTimbrSqlChain` | Generate SQL queries from natural language prompts |
| `ValidateTimbrSqlChain` | Validate SQL queries against Timbr knowledge graph schemas |
| `ExecuteTimbrQueryChain` | Execute SQL queries against Timbr knowledge graph databases |
| `GenerateAnswerChain` | Generate human-readable answers from query results |
| `TimbrSqlAgent` | End-to-end SQL agent for natural language queries |

### TimbrSqlAgent Parameters

The `TimbrSqlAgent` is a pre-built agent that combines all the above tools for end-to-end natural language to SQL processing.

For the complete list of parameters and detailed documentation, see: [TimbrSqlAgent Documentation](https://docs.timbr.ai/doc/docs/integration/langchain-sdk/#timbr-sql-agent)

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `llm` | BaseChatModel | Yes | Language model instance (ChatOllama, ChatOllama, etc.) |
| `url` | str | Yes | Timbr application URL |
| `token` | str | Yes | Timbr API token |
| `ontology` | str | Yes | Knowledge graph ontology name |
| `schema` | str | No | Database schema name |
| `concept` | str | No | Specific concept to focus on |
| `concepts_list` | List[str] | No | List of relevant concepts |
| `views_list` | List[str] | No | List of available views |
| `note` | str | No | Additional context or instructions |
| `retries` | int | No | Number of retry attempts (default: 3) |
| `should_validate_sql` | bool | No | Whether to validate generated SQL (default: True) |

## Setup

The integration lives in the `langchain-timbr` package.

In this example, we'll use Ollama for the LLM provider.
"""
logger.info("# Timbr")

# %pip install --quiet -U langchain-timbr[ollama]

"""
### Credentials

You'll need Timbr credentials to use the tools. Get your API token from your Timbr application's API settings.
"""
logger.info("### Credentials")

# import getpass

if not os.environ.get("TIMBR_URL"):
    os.environ["TIMBR_URL"] = input("Timbr URL:\n")

if not os.environ.get("TIMBR_TOKEN"):
#     os.environ["TIMBR_TOKEN"] = getpass.getpass("Timbr API Token:\n")

if not os.environ.get("TIMBR_ONTOLOGY"):
    os.environ["TIMBR_ONTOLOGY"] = input("Timbr Ontology:\n")

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:\n")

"""
## Instantiation

Instantiate Timbr tools and agents. First, let's set up the LLM and basic Timbr chains:
"""
logger.info("## Instantiation")



llm = LlmWrapper(
#     llm_type=LlmTypes.Ollama, api_key=os.environ["OPENAI_API_KEY"], model="llama3.2"
)

execute_timbr_query_chain = ExecuteTimbrQueryChain(
    llm=llm,
    url=os.environ["TIMBR_URL"],
    token=os.environ["TIMBR_TOKEN"],
    ontology=os.environ["TIMBR_ONTOLOGY"],
)

generate_answer_chain = GenerateAnswerChain(
    llm=llm, url=os.environ["TIMBR_URL"], token=os.environ["TIMBR_TOKEN"]
)

"""
## Invocation

### Execute SQL queries from natural language

You can use the individual chains to perform specific operations:
"""
logger.info("## Invocation")

result = execute_timbr_query_chain.invoke(
    {"prompt": "What are the total sales for last month?"}
)

logger.debug("SQL Query:", result["sql"])
logger.debug("Results:", result["rows"])
logger.debug("Concept:", result["concept"])

answer_result = generate_answer_chain.invoke(
    {"prompt": "What are the total sales for last month?", "rows": result["rows"]}
)

logger.debug("Human-readable answer:", answer_result["answer"])

"""
## Use within an agent

### Using TimbrSqlAgent

The `TimbrSqlAgent` provides an end-to-end solution that combines concept identification, SQL generation, validation, execution, and answer generation:
"""
logger.info("## Use within an agent")


timbr_agent = TimbrSqlAgent(
    llm=llm,
    url=os.environ["TIMBR_URL"],
    token=os.environ["TIMBR_TOKEN"],
    ontology=os.environ["TIMBR_ONTOLOGY"],
    concepts_list=["Sales", "Orders"],  # optional
    views_list=["sales_view"],  # optional
    note="Focus on monthly aggregations",  # optional
    retries=3,  # optional
    should_validate_sql=True,  # optional
)

agent_result = AgentExecutor.from_agent_and_tools(
    agent=timbr_agent,
    tools=[],  # No tools needed as we're directly using the chain
    verbose=True,
).invoke("Show me the top 5 customers by total sales amount this year")

logger.debug("Final Answer:", agent_result["answer"])
logger.debug("Generated SQL:", agent_result["sql"])
logger.debug("Usage Metadata:", agent_result.get("usage_metadata", {}))

"""
### Sequential Chains

You can combine multiple Timbr chains using LangChain's SequentialChain for custom workflows:
"""
logger.info("### Sequential Chains")


pipeline = SequentialChain(
    chains=[execute_timbr_query_chain, generate_answer_chain],
    input_variables=["prompt"],
    output_variables=["answer", "sql", "rows"],
)

pipeline_result = pipeline.invoke(
    {"prompt": "What are the average order values by customer segment?"}
)

logger.debug("Pipeline Result:", pipeline_result)

result_with_metadata = execute_timbr_query_chain.invoke(
    {"prompt": "How many orders were placed last quarter?"}
)

usage_metadata = result_with_metadata.get("execute_timbr_usage_metadata", {})
determine_concept_usage = usage_metadata.get("determine_concept", {})
generate_sql_usage = usage_metadata.get("generate_sql", {})

logger.debug(determine_concept_usage)

logger.debug(
    "Concept determination token estimate:",
    determine_concept_usage.get("approximate", "N/A"),
)
logger.debug(
    "Concept determination tokens:",
    determine_concept_usage.get("token_usage", {}).get("total_tokens", "N/A"),
)

logger.debug("SQL generation token estimate:", generate_sql_usage.get("approximate", "N/A"))
logger.debug(
    "SQL generation tokens:",
    generate_sql_usage.get("token_usage", {}).get("total_tokens", "N/A"),
)

"""
## API reference

- [PyPI](https://pypi.org/project/langchain-timbr)
- [GitHub](https://github.com/WPSemantix/langchain-timbr)
- [LangChain Timbr Documentation](https://docs.timbr.ai/doc/docs/integration/langchain-sdk/)
- [LangGraph Timbr Documentation](https://docs.timbr.ai/doc/docs/integration/langgraph-sdk)
- [Timbr Official Website](https://timbr.ai/)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)