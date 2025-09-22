from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import SequentialChain
from langchain_timbr import ExecuteTimbrQueryChain
from langchain_timbr import ExecuteTimbrQueryChain, GenerateAnswerChain
from langchain_timbr import LlmWrapper, LlmTypes
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
# Timbr

[Timbr](https://docs.timbr.ai/doc/docs/integration/langchain-sdk/) integrates natural language inputs with Timbr's ontology-driven semantic layer. Leveraging Timbr's robust ontology capabilities, the SDK integrates with Timbr data models and leverages semantic relationships and annotations, enabling users to query data using business-friendly language.

Timbr provides a pre-built SQL agent, `TimbrSqlAgent`, which can be used for end-to-end purposes from user prompt, through semantic SQL query generation and validation, to query execution and result analysis.

For customizations and partial usage, you can use LangChain chains and LangGraph nodes with our 5 main tools:

- `IdentifyTimbrConceptChain` & `IdentifyConceptNode` - Identify relevant concepts from user prompts
- `GenerateTimbrSqlChain` & `GenerateTimbrSqlNode` - Generate SQL queries from natural language prompts
- `ValidateTimbrSqlChain` & `ValidateSemanticSqlNode` - Validate SQL queries against Timbr knowledge graph schemas
- `ExecuteTimbrQueryChain` & `ExecuteSemanticQueryNode` - Execute (semantic and regular) SQL queries against Timbr knowledge graph databases
- `GenerateAnswerChain` & `GenerateResponseNode` - Generate human-readable answers based on a given prompt and data rows

Additionally, `langchain-timbr` provides `TimbrLlmConnector` for manual integration with Timbr's semantic layer using LLM providers. This connector includes the following methods:

- `get_ontologies` - List Timbr's semantic knowledge graphs
- `get_concepts` - List selected knowledge graph ontology representation concepts
- `get_views` - List selected knowledge graph ontology representation views
- `determine_concept` - Identify relevant concepts from user prompts
- `generate_sql` - Generate SQL queries from natural language prompts
- `validate_sql` - Validate SQL queries against Timbr knowledge graph schemas
- `run_timbr_query` - Execute (semantic and regular) SQL queries against Timbr knowledge graph databases
- `run_llm_query` - Execute agent pipeline to determine concept, generate SQL, and run query from natural language prompt

## Quickstart

### Installation

#### Install the package
"""
logger.info("# Timbr")

pip install langchain-timbr

"""
#### Optional: Install with selected LLM provider

Choose one of: ollama, anthropic, google, azure_openai, snowflake, databricks (or 'all')
"""
logger.info("#### Optional: Install with selected LLM provider")

pip install 'langchain-timbr[<your selected providers, separated by comma without spaces>]'

"""
## Configuration

Starting from `langchain-timbr` v2.0.0, all chains, agents, and nodes support optional environment-based configuration. You can set the following environment variables to provide default values and simplify setup for the provided tools:

### Timbr Connection Parameters

- **TIMBR_URL**: Default Timbr server URL
- **TIMBR_TOKEN**: Default Timbr authentication token
- **TIMBR_ONTOLOGY**: Default ontology/knowledge graph name

When these environment variables are set, the corresponding parameters (`url`, `token`, `ontology`) become optional in all chain and agent constructors and will use the environment values as defaults.

### LLM Configuration Parameters

- **LLM_TYPE**: The type of LLM provider (one of langchain_timbr LlmTypes enum: 'ollama-chat', 'anthropic-chat', 'chat-google-generative-ai', 'azure-ollama-chat', 'snowflake-cortex', 'chat-databricks')
- **LLM_API_KEY**: The API key for authenticating with the LLM provider
- **LLM_MODEL**: The model name or deployment to use
- **LLM_TEMPERATURE**: Temperature setting for the LLM
- **LLM_ADDITIONAL_PARAMS**: Additional parameters as dict or JSON string

When LLM environment variables are set, the `llm` parameter becomes optional and will use the `LlmWrapper` with environment configuration.

Example environment setup:
"""
logger.info("## Configuration")

export TIMBR_URL="https://your-timbr-app.com/"
export TIMBR_TOKEN="tk_XXXXXXXXXXXXXXXXXXXXXXXX"
export TIMBR_ONTOLOGY="timbr_knowledge_graph"

export LLM_TYPE="ollama-chat"
export LLM_API_KEY="your-ollama-api-key"
export LLM_MODEL="gpt-4o"
export LLM_TEMPERATURE="0.1"
export LLM_ADDITIONAL_PARAMS='{"max_tokens": 1000}'

"""
## Usage

Import and utilize your intended chain/node, or use TimbrLlmConnector to manually integrate with Timbr's semantic layer. For a complete agent working example, see the [Timbr tool page](/docs/integrations/tools/timbr).

### ExecuteTimbrQueryChain example
"""
logger.info("## Usage")


llm = ChatOllama(model="llama3.2")

llm = LlmWrapper(llm_type=LlmTypes.Ollama, model="llama3.2")

execute_timbr_query_chain = ExecuteTimbrQueryChain(
    llm=llm,
    url="https://your-timbr-app.com/",
    token="tk_XXXXXXXXXXXXXXXXXXXXXXXX",
    ontology="timbr_knowledge_graph",
    schema="dtimbr",              # optional
    concept="Sales",              # optional
    concepts_list=["Sales","Orders"],  # optional
    views_list=["sales_view"],         # optional
    note="We only need sums",     # optional
    retries=3,                    # optional
    should_validate_sql=True      # optional
)

result = execute_timbr_query_chain.invoke({"prompt": "What are the total sales for last month?"})
rows = result["rows"]
sql = result["sql"]
concept = result["concept"]
schema = result["schema"]
error = result.get("error", None)

usage_metadata = result.get("execute_timbr_usage_metadata", {})
determine_concept_usage = usage_metadata.get('determine_concept', {})
generate_sql_usage = usage_metadata.get('generate_sql', {})

"""
### Multiple chains using SequentialChain example
"""
logger.info("### Multiple chains using SequentialChain example")


llm = ChatOllama(model="llama3.2")

llm = LlmWrapper(llm_type=LlmTypes.Ollama, model="llama3.2")

execute_timbr_query_chain = ExecuteTimbrQueryChain(
    llm=llm,
    url='https://your-timbr-app.com/',
    token='tk_XXXXXXXXXXXXXXXXXXXXXXXX',
    ontology='timbr_knowledge_graph',
)

generate_answer_chain = GenerateAnswerChain(
    llm=llm,
    url='https://your-timbr-app.com/',
    token='tk_XXXXXXXXXXXXXXXXXXXXXXXX',
)

pipeline = SequentialChain(
    chains=[execute_timbr_query_chain, generate_answer_chain],
    input_variables=["prompt"],
    output_variables=["answer", "sql"]
)

result = pipeline.invoke({"prompt": "What are the total sales for last month?"})

"""
## Additional Resources

- [PyPI](https://pypi.org/project/langchain-timbr)
- [GitHub](https://github.com/WPSemantix/langchain-timbr)
- [LangChain Timbr Docs](https://docs.timbr.ai/doc/docs/integration/langchain-sdk/)
- [LangGraph Timbr Docs](https://docs.timbr.ai/doc/docs/integration/langgraph-sdk)
"""
logger.info("## Additional Resources")

logger.info("\n\n[DONE]", bright=True)