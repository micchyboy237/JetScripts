from elasticsearch import Elasticsearch
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains.elasticsearch_database import ElasticsearchDatabaseChain
from langchain.prompts.prompt import PromptTemplate
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
# Elasticsearch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/qa_structured/integrations/elasticsearch.ipynb)

We can use LLMs to interact with Elasticsearch analytics databases in natural language.

This chain builds search queries via the Elasticsearch DSL API (filters and aggregations).

The Elasticsearch client must have permissions for index listing, mapping description and search queries.

See [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) for instructions on how to run Elasticsearch locally.
"""
logger.info("# Elasticsearch")

# ! pip install langchain langchain-experimental ollama elasticsearch


ELASTIC_SEARCH_SERVER = "https://elastic:pass@localhost:9200"
db = Elasticsearch(ELASTIC_SEARCH_SERVER)

"""
Uncomment the next cell to initially populate your db.
"""
logger.info("Uncomment the next cell to initially populate your db.")



llm = ChatOllama(model="llama3.2")
chain = ElasticsearchDatabaseChain.from_llm(llm=llm, database=db, verbose=True)

question = "What are the first names of all the customers?"
chain.run(question)

"""
We can customize the prompt.
"""
logger.info("We can customize the prompt.")


PROMPT_TEMPLATE = """Given an input question, create a syntactically correct Elasticsearch query to run. Unless the user specifies in their question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Unless told to do not query for all the columns from a specific index, only ask for a few relevant columns given the question.

Pay attention to use only the column names that you can see in the mapping description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which index. Return the query as valid json.

Use the following format:

Question: Question here
ESQuery: Elasticsearch Query formatted as json
"""

PROMPT = PromptTemplate.from_template(
    PROMPT_TEMPLATE,
)
chain = ElasticsearchDatabaseChain.from_llm(llm=llm, database=db, query_prompt=PROMPT)

logger.info("\n\n[DONE]", bright=True)