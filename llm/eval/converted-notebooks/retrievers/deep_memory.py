from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()





# %pip install llama-index-vector-stores-deeplake
# %pip install llama-index-llms-ollama

import nest_asyncio
import os
import getpass

nest_asyncio.apply()

# !pip install deeplake beautifulsoup4 html2text tiktoken openai llama-index python-dotenv







import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def get_all_links(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the page: {url}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    links = [
        urljoin(url, a["href"])
        for a in soup.find_all("a", href=True)
        if a["href"]
    ]

    return links

from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from llama_index.core import Document


def load_documents(url):
    all_links = get_all_links(url)
    loader = AsyncHtmlLoader(all_links)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    docs = [Document.from_langchain_format(doc) for doc in docs_transformed]
    return docs


docs = load_documents("https://docs.deeplake.ai/en/latest/")

len(docs)

from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama


# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Ollama API token: ")
os.environ["ACTIVELOOP_TOKEN"] = getpass.getpass(
    "Enter your ActiveLoop API token: "
)  # Get your API token from https://app.activeloop.ai, click on your profile picture in the top right corner, and select "API Tokens"

token = os.getenv("ACTIVELOOP_TOKEN")

vector_store = DeepLakeVectorStore(
    dataset_path="hub://activeloop-test/deeplake_docs_deepmemory2",
    overwrite=False,  # set to True to overwrite the existing dataset
    runtime={"tensor_db": True},
    token=token,
)

def create_modules(vector_store, docs=[], populate_vector_store=True):
    if populate_vector_store:
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(docs)
    else:
        nodes = []

    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context, nodes, llm

(
    storage_context,
    nodes,
    llm,
) = create_modules(
    docs=docs,
    vector_store=vector_store,
)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
deep_memory_retriever = vector_index.as_retriever(
    similarity_top_k=4, deep_memory=True
)







from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
import random


def create_train_test_datasets(
    number_of_samples=600, llm=None, nodes=None, save=False
):
    random_indices = random.sample(range(len(nodes)), number_of_samples)

    ratio = int(len(random_indices) * 0.8)

    train_indices = random_indices[:ratio]
    test_indices = random_indices[ratio:]

    train_nodes = [nodes[i] for i in train_indices]
    test_nodes = [nodes[i] for i in test_indices]

    train_qa_dataset = generate_question_context_pairs(
        train_nodes, llm=llm, num_questions_per_chunk=1
    )

    test_qa_dataset = generate_question_context_pairs(
        test_nodes, llm=llm, num_questions_per_chunk=1
    )

    if save:
        train_qa_dataset.save_json(
            f"deeplake_docs_{number_of_samples}_train.json"
        )
        test_qa_dataset.save_json(
            f"deeplake_docs_{number_of_samples}_test.json"
        )
    return train_qa_dataset, test_qa_dataset

train_qa_dataset, test_qa_dataset = create_train_test_datasets(
    number_of_samples=600, llm=llm, nodes=nodes, save=True
)

train_qa_dataset = EmbeddingQAFinetuneDataset.from_json(
    "deeplake_docs_600_train.json"
)
test_qa_dataset = EmbeddingQAFinetuneDataset.from_json(
    "deeplake_docs_600_test.json"
)

def create_query_relevance(qa_dataset):
    """Function for converting llama-index dataset to correct format for deep memory training"""
    queries = [text for _, text in qa_dataset.queries.items()]
    relevant_docs = qa_dataset.relevant_docs
    relevance = []
    for doc in relevant_docs:
        relevance.append([(relevant_docs[doc][0], 1)])
    return queries, relevance

train_queries, train_relevance = create_query_relevance(train_qa_dataset)
test_queries, test_relevance = create_query_relevance(test_qa_dataset)

train_queries[:3]

train_relevance[:3]

test_queries[:3]

test_relevance[:3]

from langchain.embeddings.openai import OllamaEmbeddings

embeddings = OllamaEmbeddings()


job_id = vector_store.vectorstore.deep_memory.train(
    queries=train_queries,
    relevance=train_relevance,
    embedding_function=embeddings.embed_documents,
)

vector_store.vectorstore.deep_memory.status(job_id)





recalls = vector_store.vectorstore.deep_memory.evaluate(
    queries=test_queries,
    relevance=test_relevance,
    embedding_function=embeddings.embed_documents,
)



import pandas as pd


def display_results(eval_results):
    """Display results from evaluate."""
    hit_rates = []
    mrrs = []
    names = []
    for name, eval_result in eval_results.items():
        metric_dicts = []
        for er in eval_result:
            metric_dict = er.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()

        hit_rates.append(hit_rate)
        mrrs.append(mrr)
        names.append(name)

    metric_df = pd.DataFrame(
        [
            {"retrievers": names[i], "hit_rate": hit_rates[i], "mrr": mrrs[i]}
            for i in range(2)
        ],
    )

    return metric_df



from llama_index.core.evaluation import RetrieverEvaluator

deep_memory_retriever = vector_index.as_retriever(
    similarity_top_k=10, vector_store_kwargs={"deep_memory": True}
)
dm_retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=deep_memory_retriever
)

dm_eval_results = dm_retriever_evaluator.evaluate_dataset(
    test_qa_dataset, retriever=dm_retriever_evaluator
)

from llama_index.core.evaluation import RetrieverEvaluator

naive_retriever = vector_index.as_retriever(similarity_top_k=10)
naive_retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=naive_retriever
)

naive_eval_results = naive_retriever_evaluator.evaluate_dataset(
    test_qa_dataset, retriever=naive_retriever
)

eval_results = {
    f"{mode} with Deep Memory top-10 eval": eval_result
    for mode, eval_result in zip(
        ["with", "without"], [dm_eval_results, naive_eval_results]
    )
}

display_results(eval_results)





query_engine = vector_index.as_query_engine(
    vector_store_kwargs={"deep_memory": True}, llm=llm
)
response = query_engine.query(
    "How can you connect your own storage to the deeplake?"
)
print(response)

query_engine = vector_index.as_query_engine(
    vector_store_kwargs={"deep_memory": False}, llm=llm
)
response = query_engine.query(
    "How can you connect your own storage to the deeplake?"
)
print(response)


logger.info("\n\n[DONE]", bright=True)