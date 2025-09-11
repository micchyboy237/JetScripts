from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import Client
import json
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
# Optimization

This notebook goes over how to optimize chains using LangChain and [LangSmith](https://smith.langchain.com).

## Set up

We will set an environment variable for LangSmith, and load the relevant data
"""
logger.info("# Optimization")


os.environ["LANGSMITH_PROJECT"] = "movie-qa"


df = pd.read_csv("data/imdb_top_1000.csv")

df["Released_Year"] = df["Released_Year"].astype(int, errors="ignore")

"""
## Create the initial retrieval chain

We will use a self-query retriever
"""
logger.info("## Create the initial retrieval chain")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

records = df.to_dict("records")
documents = [Document(page_content=d["Overview"], metadata=d) for d in records]

vectorstore = Chroma.from_documents(documents, embeddings)


metadata_field_info = [
    AttributeInfo(
        name="Released_Year",
        description="The year the movie was released",
        type="int",
    ),
    AttributeInfo(
        name="Series_Title",
        description="The title of the movie",
        type="str",
    ),
    AttributeInfo(
        name="Genre",
        description="The genre of the movie",
        type="string",
    ),
    AttributeInfo(
        name="IMDB_Rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"
llm = ChatOllama(model="llama3.2")
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)



prompt = ChatPromptTemplate.from_template(
    """Answer the user's question based on the below information:

Information:

{info}

Question: {question}"""
)
generator = (prompt | ChatOllama(model="llama3.2") | StrOutputParser()).with_config(
    run_name="generator"
)

chain = (
    RunnablePassthrough.assign(info=(lambda x: x["question"]) | retriever) | generator
)

"""
## Run examples

Run examples through the chain. This can either be manually, or using a list of examples, or production traffic
"""
logger.info("## Run examples")

chain.invoke({"question": "what is a horror movie released in early 2000s"})

"""
## Annotate

Now, go to LangSmitha and annotate those examples as correct or incorrect

## Create Dataset

We can now create a dataset from those runs.

What we will do is find the runs marked as correct, then grab the sub-chains from them. Specifically, the query generator sub chain and the final generation step
"""
logger.info("## Annotate")


client = Client()

runs = list(
    client.list_runs(
        project_name="movie-qa",
        execution_order=1,
        filter="and(eq(feedback_key, 'correctness'), eq(feedback_score, 1))",
    )
)

len(runs)

gen_runs = []
query_runs = []
for r in runs:
    gen_runs.extend(
        list(
            client.list_runs(
                project_name="movie-qa",
                filter="eq(name, 'generator')",
                trace_id=r.trace_id,
            )
        )
    )
    query_runs.extend(
        list(
            client.list_runs(
                project_name="movie-qa",
                filter="eq(name, 'query_constructor')",
                trace_id=r.trace_id,
            )
        )
    )

runs[0].inputs

runs[0].outputs

query_runs[0].inputs

query_runs[0].outputs

gen_runs[0].inputs

gen_runs[0].outputs

"""
## Create datasets

We can now create datasets for the query generation and final generation step.
We do this so that (1) we can inspect the datapoints, (2) we can edit them if needed, (3) we can add to them over time
"""
logger.info("## Create datasets")

client.create_dataset("movie-query_constructor")

inputs = [r.inputs for r in query_runs]
outputs = [r.outputs for r in query_runs]

client.create_examples(
    inputs=inputs, outputs=outputs, dataset_name="movie-query_constructor"
)

client.create_dataset("movie-generator")

inputs = [r.inputs for r in gen_runs]
outputs = [r.outputs for r in gen_runs]

client.create_examples(inputs=inputs, outputs=outputs, dataset_name="movie-generator")

"""
## Use as few shot examples

We can now pull down a dataset and use them as few shot examples in a future chain
"""
logger.info("## Use as few shot examples")

examples = list(client.list_examples(dataset_name="movie-query_constructor"))



def filter_to_string(_filter):
    if "operator" in _filter:
        args = [filter_to_string(f) for f in _filter["arguments"]]
        return f"{_filter['operator']}({','.join(args)})"
    else:
        comparator = _filter["comparator"]
        attribute = json.dumps(_filter["attribute"])
        value = json.dumps(_filter["value"])
        return f"{comparator}({attribute}, {value})"

model_examples = []

for e in examples:
    if "filter" in e.outputs["output"]:
        string_filter = filter_to_string(e.outputs["output"]["filter"])
    else:
        string_filter = "NO_FILTER"
    model_examples.append(
        (
            e.inputs["query"],
            {"query": e.outputs["output"]["query"], "filter": string_filter},
        )
    )

retriever1 = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    chain_kwargs={"examples": model_examples},
)

chain1 = (
    RunnablePassthrough.assign(info=(lambda x: x["question"]) | retriever1) | generator
)

chain1.invoke(
    {"question": "what are good action movies made before 2000 but after 1997?"}
)

logger.info("\n\n[DONE]", bright=True)