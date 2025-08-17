from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: '🔍 search'
---

`.search()` enables you to uncover the most pertinent context by performing a semantic search across your data sources based on a given query. Refer to the function signature below:

### Parameters

<ParamField path="query" type="str">
    Question
</ParamField>
<ParamField path="num_documents" type="int" optional>
    Number of relevant documents to fetch. Defaults to `3`
</ParamField>
<ParamField path="where" type="dict" optional>
    Key value pair for metadata filtering.
</ParamField>
<ParamField path="raw_filter" type="dict" optional>
    Pass raw filter query based on your vector database.
    Currently, `raw_filter` param is only supported for Pinecone vector database.
</ParamField>

### Returns

<ResponseField name="answer" type="dict">
    Return list of dictionaries that contain the relevant chunk and their source information.
</ResponseField>

## Usage

### Basic

Refer to the following example on how to use the search api:
"""
logger.info("### Parameters")


app = App()
app.add("https://www.forbes.com/profile/elon-musk")

context = app.search("What is the net worth of Elon?", num_documents=2)
logger.debug(context)

"""
### Advanced

#### Metadata filtering using `where` params

Here is an advanced example of `search()` API with metadata filtering on pinecone database:
"""
logger.info("### Advanced")



os.environ["PINECONE_API_KEY"] = "xxx"

config = {
    "vectordb": {
        "provider": "pinecone",
        "config": {
            "metric": "dotproduct",
            "vector_dimension": 1536,
            "index_name": "ec-test",
            "serverless_config": {"cloud": "aws", "region": "us-west-2"},
        },
    }
}

app = App.from_config(config=config)

app.add("https://www.forbes.com/profile/bill-gates", metadata={"type": "forbes", "person": "gates"})
app.add("https://en.wikipedia.org/wiki/Bill_Gates", metadata={"type": "wiki", "person": "gates"})

results = app.search("What is the net worth of Bill Gates?", where={"person": "gates"})
logger.debug("Num of search results: ", len(results))

"""
#### Metadata filtering using `raw_filter` params

Following is an example of metadata filtering by passing the raw filter query that pinecone vector database follows:
"""
logger.info("#### Metadata filtering using `raw_filter` params")



os.environ["PINECONE_API_KEY"] = "xxx"

config = {
    "vectordb": {
        "provider": "pinecone",
        "config": {
            "metric": "dotproduct",
            "vector_dimension": 1536,
            "index_name": "ec-test",
            "serverless_config": {"cloud": "aws", "region": "us-west-2"},
        },
    }
}

app = App.from_config(config=config)

app.add("https://www.forbes.com/profile/bill-gates", metadata={"year": 2022, "person": "gates"})
app.add("https://en.wikipedia.org/wiki/Bill_Gates", metadata={"year": 2024, "person": "gates"})

logger.debug("Filter with person: gates and year > 2023")
raw_filter = {"$and": [{"person": "gates"}, {"year": {"$gt": 2023}}]}
results = app.search("What is the net worth of Bill Gates?", raw_filter=raw_filter)
logger.debug("Num of search results: ", len(results))

logger.info("\n\n[DONE]", bright=True)