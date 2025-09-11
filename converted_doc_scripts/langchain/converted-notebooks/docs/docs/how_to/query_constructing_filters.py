from jet.logger import logger
from langchain.chains.query_constructor.ir import (
Comparator,
Comparison,
Operation,
Operator,
StructuredQuery,
)
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_community.query_constructors.elasticsearch import ElasticsearchTranslator
from pydantic import BaseModel
from typing import Optional
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
sidebar_position: 6
---

# How to construct filters for query analysis

We may want to do query analysis to extract filters to pass into retrievers. One way we ask the LLM to represent these filters is as a Pydantic model. There is then the issue of converting that Pydantic model into a filter that can be passed into a retriever. 

This can be done manually, but LangChain also provides some "Translators" that are able to translate from a common syntax into filters specific to each retriever. Here, we will cover how to use those translators.
"""
logger.info("# How to construct filters for query analysis")



"""
In this example, `year` and `author` are both attributes to filter on.
"""
logger.info("In this example, `year` and `author` are both attributes to filter on.")

class Search(BaseModel):
    query: str
    start_year: Optional[int]
    author: Optional[str]

search_query = Search(query="RAG", start_year=2022, author="LangChain")

def construct_comparisons(query: Search):
    comparisons = []
    if query.start_year is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.GT,
                attribute="start_year",
                value=query.start_year,
            )
        )
    if query.author is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="author",
                value=query.author,
            )
        )
    return comparisons

comparisons = construct_comparisons(search_query)

_filter = Operation(operator=Operator.AND, arguments=comparisons)

ElasticsearchTranslator().visit_operation(_filter)

ChromaTranslator().visit_operation(_filter)

logger.info("\n\n[DONE]", bright=True)