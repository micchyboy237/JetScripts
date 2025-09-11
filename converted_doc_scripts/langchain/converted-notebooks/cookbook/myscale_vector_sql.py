from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.retrievers.vector_sql_database import (
    VectorSQLDatabaseChainRetriever,
)
from langchain_experimental.sql.prompt import MYSCALE_PROMPT
from langchain_experimental.sql.vector_sql import (
    VectorSQLDatabaseChain,
    VectorSQLRetrieveAllOutputParser,
)
from langchain_experimental.sql.vector_sql import VectorSQLDatabaseChain
from langchain_experimental.sql.vector_sql import VectorSQLOutputParser
from os import environ
from sqlalchemy import MetaData, create_engine
import os
import pandas as pd
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
# Vector SQL Retriever with MyScale

>[MyScale](https://docs.myscale.com/en/) is an integrated vector database. You can access your database in SQL and also from here, LangChain. MyScale can make a use of [various data types and functions for filters](https://blog.myscale.com/2023/06/06/why-integrated-database-solution-can-boost-your-llm-apps/#filter-on-anything-without-constraints). It will boost up your LLM app no matter if you are scaling up your data or expand your system to broader application.
"""
logger.info("# Vector SQL Retriever with MyScale")

# !pip3 install clickhouse-sqlalchemy InstructorEmbedding sentence_transformers ollama langchain-experimental

# import getpass


MYSCALE_HOST = "msc-4a9e710a.us-east-1.aws.staging.myscale.cloud"
MYSCALE_PORT = 443
MYSCALE_USER = "chatdata"
MYSCALE_PASSWORD = "myscale_rocks"
# OPENAI_API_KEY = getpass.getpass("Ollama API Key:")

engine = create_engine(
    f"clickhouse://{MYSCALE_USER}:{MYSCALE_PASSWORD}@{MYSCALE_HOST}:{MYSCALE_PORT}/default?protocol=https"
)
metadata = MetaData(bind=engine)
# environ["OPENAI_API_KEY"] = OPENAI_API_KEY


output_parser = VectorSQLOutputParser.from_embeddings(
    model=HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
    )
)


chain = VectorSQLDatabaseChain(
    llm_chain=LLMChain(
        #         llm=Ollama(ollama_api_key=OPENAI_API_KEY, temperature=0),
        prompt=MYSCALE_PROMPT,
    ),
    top_k=10,
    return_direct=True,
    sql_cmd_parser=output_parser,
    database=SQLDatabase(engine, None, metadata),
)


pd.DataFrame(
    chain.run(
        "Please give me 10 papers to ask what is PageRank?",
        callbacks=[StdOutCallbackHandler()],
    )
)

"""
## SQL Database as Retriever
"""
logger.info("## SQL Database as Retriever")


output_parser_retrieve_all = VectorSQLRetrieveAllOutputParser.from_embeddings(
    output_parser.model
)

chain = VectorSQLDatabaseChain.from_llm(
    #     llm=Ollama(ollama_api_key=OPENAI_API_KEY, temperature=0),
    prompt=MYSCALE_PROMPT,
    top_k=10,
    return_direct=True,
    db=SQLDatabase(engine, None, metadata),
    sql_cmd_parser=output_parser_retrieve_all,
    native_format=True,
)

retriever = VectorSQLDatabaseChainRetriever(
    sql_db_chain=chain, page_content_key="abstract"
)

document_with_metadata_prompt = PromptTemplate(
    input_variables=["page_content", "id", "title",
                     "authors", "pubdate", "categories"],
    template="Content:\n\tTitle: {title}\n\tAbstract: {page_content}\n\tAuthors: {authors}\n\tDate of Publication: {pubdate}\n\tCategories: {categories}\nSOURCE: {id}",
)

chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOllama(
        #         model_name="gpt-3.5-turbo-16k", ollama_api_key=OPENAI_API_KEY, temperature=0.6
    ),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "document_prompt": document_with_metadata_prompt,
    },
    return_source_documents=True,
)
ans = chain(
    "Please give me 10 papers to ask what is PageRank?",
    callbacks=[StdOutCallbackHandler()],
)
logger.debug(ans["answer"])

logger.info("\n\n[DONE]", bright=True)
