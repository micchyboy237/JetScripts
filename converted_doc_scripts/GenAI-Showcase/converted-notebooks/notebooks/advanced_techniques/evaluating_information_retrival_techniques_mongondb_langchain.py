from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.base import BaseSearch
from datetime import datetime
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.schema import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasFullTextSearchRetriever
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
from pymongo.operations import SearchIndexModel
from tqdm import tqdm
from typing import Any, List, Tuple
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import ollama
import os
import pymongo
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Information Retrieval Evaluation With BEIR Benchmark and LangChain and MongoDB


---

# **Step 1: Install Libraires and Set Environment Variables**
"""
logger.info("# Information Retrieval Evaluation With BEIR Benchmark and LangChain and MongoDB")

# !pip install -q ollama pymongo langchain langchain_mongodb jet.llm.ollama.base_langchain beir

# import getpass

# OPENAI_API_KEY = getpass.getpass("Enter Ollama API Key: ")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

GPT_MODEL = "gpt-4o-2024-08-06"

EMBEDDING_MODEL = "mxbai-embed-large"
EMBEDDING_DIMENSION_SIZE = 256

# MONGO_URI = getpass.getpass("Enter MongoDB URI: ")
os.environ["MONGO_URI"] = MONGO_URI

metric_names = ["NDCG", "MAP", "Recall", "Precision"]
information_retrieval_search_methods = ["Lexical", "Vector", "Hybrid"]

"""
# **Step 2: Data Loading**
"""
logger.info("# **Step 2: Data Loading**")



def load_beir_dataset(dataset_name="scifact"):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

DATASET = "scifact"
corpus, queries, qrels = load_beir_dataset("scifact")

logger.debug(list(corpus.items())[0])
logger.debug()
logger.debug(list(queries.items())[0])
logger.debug()
logger.debug(list(qrels.items())[0])

"""
Corpus:
The corpus is a dictionary where each key is a document ID, and the value is another dictionary containing the document's text and title. For example:

```
'4983': {
    'text': 'Alterations of the architecture of cerebral white matter...',
    'title': 'Microstructural development of human newborn cerebral white matter...'
}
```


This corresponds to the scientific abstracts in our earlier example.

Queries:
The queries dictionary contains the scientific claims, where the key is a query ID and the value is the claim text. For example:

```
'1': '0-dimensional biomaterials show inductive properties.'
```

Qrels:
The qrels (query relevance) dictionary contains the ground truth relevance judgments. It's structured as a nested dictionary where the outer key is the query ID, the inner key is a document ID, and the value is the relevance score (typically 1 for relevant, 0 for non-relevant). For example:

```
'1': {'31715818': 1}
```
This indicates that for query '1', the document with ID '31715818' is relevant.

# **Step 3: Data Ingestion**
"""
logger.info("# **Step 3: Data Ingestion**")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "information_retrieval_testing"
CORPUS_COLLECTION_NAME = f"{DATASET}_corpus"
QUERIES_COLLECTION_NAME = f"{DATASET}_queries"
QRELS_COLLECTION_NAME = f"{DATASET}_qrels"
ATLAS_VECTOR_SEARCH_INDEX = "vector_index"
TEXT_SEARCH_INDEX = "text_search_index"



def get_mongo_client(mongo_uri):
    """Establish and validate connection to the MongoDB."""

    client = pymongo.MongoClient(
        mongo_uri, appname="devrel.showcase.information_retrieval_eval.python"
    )

    ping_result = client.admin.command("ping")
    if ping_result.get("ok") == 1.0:
        logger.debug("Connection to MongoDB successful")
        return client
    logger.debug("Connection to MongoDB failed")
    return None


if not MONGO_URI:
    logger.debug("MONGO_URI not set in environment variables")

def ingest_data(db, corpus=None, corpus_collection_name="", queries=None, qrels=None):
    """Ingest data into MongoDB collections."""
    if corpus and corpus_collection_name:
        corpus_docs = [
            {"_id": doc_id, "text": doc["text"], "title": doc["title"]}
            for doc_id, doc in corpus.items()
        ]
        db[corpus_collection_name].insert_many(corpus_docs)
        logger.debug(f"Ingested {len(corpus_docs)} documents into {corpus_collection_name}")

    if queries:
        query_docs = [
            {"_id": query_id, "text": query_text}
            for query_id, query_text in queries.items()
        ]
        db[QUERIES_COLLECTION_NAME].insert_many(query_docs)
        logger.debug(f"Ingested {len(query_docs)} queries into {QUERIES_COLLECTION_NAME}")

    if qrels:
        qrel_docs = [
            {"query_id": query_id, "doc_id": doc_id, "relevance": relevance}
            for query_id, relevance_dict in qrels.items()
            for doc_id, relevance in relevance_dict.items()
        ]
        db[QRELS_COLLECTION_NAME].insert_many(qrel_docs)
        logger.debug(
            f"Ingested {len(qrel_docs)} relevance judgments into {QRELS_COLLECTION_NAME}"
        )



def setup_vector_search_index_with_filter(
    collection, index_definition, index_name="vector_index"
):
    """
    Setup a vector search index for a MongoDB collection.

    Args:
    collection: MongoDB collection object
    index_definition: Dictionary containing the index definition
    index_name: Name of the index (default: "vector_index_with_filter")
    """
    new_vector_search_index_model = SearchIndexModel(
        definition=index_definition,
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=new_vector_search_index_model)
        logger.debug(f"Creating index '{index_name}'...")
        logger.debug(f"New index '{index_name}' created successfully:", result)
    except Exception as e:
        logger.debug(f"Error creating new vector search index '{index_name}': {e!s}")


def create_collection_search_index(collection, index_definition, index_name):
    """
    Create a search index for a MongoDB Atlas collection.

    Args:
    collection: MongoDB collection object
    index_definition: Dictionary defining the index mappings
    index_name: String name for the index

    Returns:
    str: Result of the index creation operation
    """

    try:
        search_index_model = SearchIndexModel(
            definition=index_definition, name=index_name
        )

        result = collection.create_search_index(model=search_index_model)
        logger.debug(f"Search index '{index_name}' created successfully")
        return result
    except Exception as e:
        logger.debug(f"Error creating search index: {e!s}")
        return None


def print_collection_search_indexes(collection):
    """
    Print all search indexes for a given collection.

    Args:
    collection: MongoDB collection object
    """
    logger.debug(f"\nSearch indexes for collection '{collection.name}':")
    for index in collection.list_search_indexes():
        logger.debug(f"Index: {index['name']}")

corpus_text_index_definition = {
    "mappings": {
        "dynamic": True,
        "fields": {"text": {"type": "string"}, "title": {"type": "string"}},
    }
}

corpus_vector_search_index_definition = {
    "mappings": {
        "dynamic": True,
        "fields": {
            "embedding": {
                "dimensions": 256,
                "similarity": "cosine",
                "type": "knnVector",
            },
        },
    }
}

mongo_client = get_mongo_client(MONGO_URI)

if mongo_client:
    db = mongo_client[DB_NAME]

for collection in [
    CORPUS_COLLECTION_NAME,
    QUERIES_COLLECTION_NAME,
    QRELS_COLLECTION_NAME,
]:
    db[collection].delete_many({})

ingest_data(db, corpus, CORPUS_COLLECTION_NAME, queries, qrels)

create_collection_search_index(
    db[CORPUS_COLLECTION_NAME], corpus_text_index_definition, TEXT_SEARCH_INDEX
)

setup_vector_search_index_with_filter(
    db[CORPUS_COLLECTION_NAME], corpus_vector_search_index_definition
)

print_collection_search_indexes(db[CORPUS_COLLECTION_NAME])

"""
# **Step 4: Embedding Generation**
"""
logger.info("# **Step 4: Embedding Generation**")



def generate_and_store_embeddings(corpus, db, collection_name):
    collection = db[collection_name]

    logger.debug("Checking for documents without embeddings...")

    all_doc_ids = set(corpus.keys())

    docs_with_embeddings = set(
        doc["_id"]
        for doc in collection.find(
            {"_id": {"$in": list(all_doc_ids)}, "embedding": {"$exists": True}},
            projection={"_id": 1},
        )
    )

    documents_to_embed = []
    for doc_id in tqdm(all_doc_ids, desc="Identifying documents to embed"):
        if doc_id not in docs_with_embeddings:
            documents_to_embed.append((doc_id, corpus[doc_id]))

    logger.debug(
        f"Found {len(documents_to_embed)} documents without embeddings out of {len(corpus)} total documents."
    )

    if documents_to_embed:
        logger.debug("Generating embeddings for documents without them...")
        for doc_id, doc in tqdm(documents_to_embed, desc="Embedding documents"):
            content = f"{doc.get('title', '')} {doc.get('text', '')}"
            try:
                embedding = (
                    ollama.embeddings.create(
                        input=content,
                        model=EMBEDDING_MODEL,
                        dimensions=EMBEDDING_DIMENSION_SIZE,
                    )
                    .data[0]
                    .embedding
                )

                collection.update_one(
                    {"_id": doc_id}, {"$set": {"embedding": embedding}}, upsert=True
                )
            except Exception as e:
                logger.debug(f"Error generating embedding for document {doc_id}: {e!s}")

        logger.debug("New embeddings generated and stored successfully.")
    else:
        logger.debug(
            "All documents already have embeddings. No new embeddings were generated."
        )

    docs_with_embeddings = collection.count_documents({"embedding": {"$exists": True}})
    logger.debug(f"Total documents with embeddings: {docs_with_embeddings}")

generate_and_store_embeddings(corpus, db, CORPUS_COLLECTION_NAME)

"""
# **Step 5: Testing Information Retrieval Mechanisms**
"""
logger.info("# **Step 5: Testing Information Retrieval Mechanisms**")

logger.debug(
    f"Number of documents in {CORPUS_COLLECTION_NAME}: {db[CORPUS_COLLECTION_NAME].count_documents({})}"
)
logger.debug(
    f"Number of queries in {QUERIES_COLLECTION_NAME}: {db[QUERIES_COLLECTION_NAME].count_documents({})}"
)
logger.debug(
    f"Number of relevance judgments in {QRELS_COLLECTION_NAME}: {db[QRELS_COLLECTION_NAME].count_documents({})}"
)

logger.debug("\nSample document from corpus:")
logger.debug(db[CORPUS_COLLECTION_NAME].find_one())
logger.debug("\nSample query:")
logger.debug(db[QUERIES_COLLECTION_NAME].find_one())
logger.debug("\nSample relevance judgment:")
logger.debug(db[QRELS_COLLECTION_NAME].find_one())

"""
### Full text search MongoDB Aggregation Pipeline Integration
"""
logger.info("### Full text search MongoDB Aggregation Pipeline Integration")

def full_text_search_aggregation_pipeline():
    pass

"""
### Full text search with LangChain<>MongoDB Integration
"""
logger.info("### Full text search with LangChain<>MongoDB Integration")




def full_text_search(collection, query: str, top_k: int = 10) -> List[Document]:
    full_text_search = MongoDBAtlasFullTextSearchRetriever(
        collection=collection,
        search_index_name=TEXT_SEARCH_INDEX,
        search_field="text",
        top_k=top_k,
    )
    return full_text_search.get_relevant_documents(query)

full_text_search(
    db[CORPUS_COLLECTION_NAME], "0-dimensional biomaterials show inductive properties"
)

"""
### Vector Search LangChain<>MongoDB Integration
"""
logger.info("### Vector Search LangChain<>MongoDB Integration")


embedding_model = OllamaEmbeddings(
    model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSION_SIZE
)

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=f"{DB_NAME}.{CORPUS_COLLECTION_NAME}",
    embedding=embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX,
    text_key="text",
)

def vector_search(query: str, top_k: int = 10) -> List[Tuple[Any, float]]:
    return vector_store.similarity_search_with_score(query=query, k=top_k)

vector_search("0-dimensional biomaterials show inductive properties")

"""
### Hybrid Search LangChain<>MongoDB Integration
"""
logger.info("### Hybrid Search LangChain<>MongoDB Integration")



def hybrid_search(query: str, top_k: int = 10) -> List[Document]:
    hybrid_search = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vector_store, search_index_name="text_search_index", top_k=top_k
    )
    return hybrid_search.get_relevant_documents(query)

hybrid_search("0-dimensional biomaterials show inductive properties")

"""
# Information Retrieval Evaluation Process Begins


---



---

# **Step 6: Custom Retrieval Class For Lexical Search**
"""
logger.info("# Information Retrieval Evaluation Process Begins")




class MongoDBSearch(BaseSearch):
    def __init__(
        self, collection, search_index_name, search_field="text", batch_size=128
    ):
        self.collection = collection
        self.search_index_name = search_index_name
        self.search_field = search_field
        self.batch_size = batch_size

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        score_function: str = "dot",
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for query_id, query_text in queries.items():
            full_text_search = MongoDBAtlasFullTextSearchRetriever(
                collection=self.collection,
                search_index_name=self.search_index_name,
                search_field=self.search_field,
                top_k=top_k,
            )
            documents = full_text_search.get_relevant_documents(query_text)
            results[query_id] = {
                doc.metadata["_id"]: doc.metadata["score"] for doc in documents
            }
        return results

model = MongoDBSearch(db[CORPUS_COLLECTION_NAME], TEXT_SEARCH_INDEX)

retriever = EvaluateRetrieval(model)

results = retriever.retrieve(corpus, queries)

logger.debug("Sample of retrieved results:")
for query_id, doc_scores in list(results.items())[:5]:  # First 5 queries
    logger.debug(f"Query ID: {query_id}")
    logger.debug(f"Query text: {queries[query_id]}")
    logger.debug("Top 3 retrieved documents:")
    for doc_id, score in list(doc_scores.items())[:3]:
        logger.debug(f"  Doc ID: {doc_id}, Score: {score}")
    logger.debug()

metrics = retriever.evaluate(qrels, results, retriever.k_values)

ndcg, _map, recall, precision = metrics

lexical_search_metric_dicts = [ndcg, _map, recall, precision]

for name, metric_dict in zip(metric_names, lexical_search_metric_dicts):
    logger.debug(f"\n{name}:")
    for k, score in metric_dict.items():
        logger.debug(f"  {k}: {score:.4f}")

"""
# **Step 7: Custom Retrieval Class For Vector Search**
"""
logger.info("# **Step 7: Custom Retrieval Class For Vector Search**")

class MongoDBVectorSearch(BaseSearch):
    def __init__(
        self,
        vector_store: MongoDBAtlasVectorSearch,
        embedding_model: OllamaEmbeddings,
        batch_size=128,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        score_function: str = "dot",
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for query_id, query_text in queries.items():
            vector_results = self.vector_store.similarity_search_with_score(
                query=query_text, k=top_k
            )
            results[query_id] = {
                str(doc.metadata.get("_id", i)): score
                for i, (doc, score) in enumerate(vector_results)
            }
        return results

mongodb_vector_search = MongoDBVectorSearch(vector_store, embedding_model)

vector_search_retriever = EvaluateRetrieval(mongodb_vector_search)

vector_search_eval_results = vector_search_retriever.retrieve(corpus, queries)

logger.debug("Sample of retrieved results:")
for query_id, doc_scores in list(vector_search_eval_results.items())[
    :5
]:  # First 5 queries
    logger.debug(f"Query ID: {query_id}")
    logger.debug(f"Query text: {queries[query_id]}")
    logger.debug("Top 3 retrieved documents:")
    for doc_id, score in list(doc_scores.items())[:3]:
        logger.debug(f"  Doc ID: {doc_id}, Score: {score}")
    logger.debug()

ndcg, _map, recall, precision = vector_search_retriever.evaluate(
    qrels, vector_search_eval_results, vector_search_retriever.k_values
)

vector_search_metric_dicts = [ndcg, _map, recall, precision]

for name, metric_dict in zip(metric_names, vector_search_metric_dicts):
    logger.debug(f"\n{name}:")
    for k, score in metric_dict.items():
        logger.debug(f"  {k}: {score:.4f}")

"""
# **Step 8: Custom Retrieval Class For Hybrid Search**
"""
logger.info("# **Step 8: Custom Retrieval Class For Hybrid Search**")

class MongoDBHybridSearch(BaseSearch):
    def __init__(
        self,
        vector_store: MongoDBAtlasVectorSearch,
        search_index_name: str,
        batch_size=128,
    ):
        self.vector_store = vector_store
        self.search_index_name = search_index_name
        self.batch_size = batch_size

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        score_function: str = "dot",
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for query_id, query_text in queries.items():
            hybrid_search = MongoDBAtlasHybridSearchRetriever(
                vectorstore=self.vector_store,
                search_index_name=self.search_index_name,
                top_k=top_k,
            )
            documents = hybrid_search.get_relevant_documents(query_text)

            results[query_id] = {
                self._get_doc_id(doc): (len(documents) - i) / len(documents)
                for i, doc in enumerate(documents)
            }

        return results

    def _get_doc_id(self, doc: Document) -> str:
        return str(doc.metadata.get("_id", hash(doc.page_content)))

mongodb_hybrid_search = MongoDBHybridSearch(
    vector_store=vector_store, search_index_name="text_search_index"
)

hybrid_search_retriever = EvaluateRetrieval(mongodb_hybrid_search)

hybrid_search_results = hybrid_search_retriever.retrieve(corpus, queries)

logger.debug("Sample of retrieved results:")
for query_id, doc_scores in list(hybrid_search_results.items())[:5]:
    logger.debug(f"Query ID: {query_id}")
    logger.debug(f"Query text: {queries[query_id]}")
    logger.debug("Top 3 retrieved documents:")
    for doc_id, score in list(doc_scores.items())[:3]:
        logger.debug(f"  Doc ID: {doc_id}, Score: {score}")
    logger.debug()

ndcg, _map, recall, precision = hybrid_search_retriever.evaluate(
    qrels, hybrid_search_results, hybrid_search_retriever.k_values
)

hybrid_search_metric_dicts = [ndcg, _map, recall, precision]

for name, metric_dict in zip(metric_names, hybrid_search_metric_dicts):
    logger.debug(f"\n{name}:")
    for k, score in metric_dict.items():
        logger.debug(f"  {k}: {score:.4f}")

"""
# **Step 9: Evaluation Result Visualisation**
"""
logger.info("# **Step 9: Evaluation Result Visualisation**")



def plot_search_method_comparison(
    lexical_metrics, vector_metrics, hybrid_metrics, metric_names
):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Comparison of Search Methods", fontsize=16)

    search_methods = information_retrieval_search_methods
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

    for idx, (metric_name, ax) in enumerate(zip(metric_names, axes.flatten())):
        lexical_data = lexical_metrics[idx]
        vector_data = vector_metrics[idx]
        hybrid_data = hybrid_metrics[idx]

        all_keys = (
            set(lexical_data.keys()) | set(vector_data.keys()) | set(hybrid_data.keys())
        )

        x = np.arange(len(all_keys))
        width = 0.25

        for i, (method, data) in enumerate(
            zip(search_methods, [lexical_data, vector_data, hybrid_data])
        ):
            values = [data.get(k, 0) for k in all_keys]
            ax.bar(x + i * width, values, width, label=method, color=colors[i])

        ax.set_ylabel("Score")
        ax.set_title(metric_name)
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_keys, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

plot_search_method_comparison(
    lexical_search_metric_dicts,
    vector_search_metric_dicts,
    hybrid_search_metric_dicts,
    metric_names,
)

"""
# **Step 10: Storing Evaluation Results In MongoDB**
"""
logger.info("# **Step 10: Storing Evaluation Results In MongoDB**")



def store_evaluation_results(
    db: Any,
    search_method: str,
    metrics: Dict[str, Dict[str, float]],
    additional_info: Dict[str, Any] | None = None,
):
    """
    Store evaluation results in MongoDB.

    Args
    db: MongoDB database instance
    search_method: Name of the search method (e.g., 'lexical', 'vector', 'hybrid')
    metrics: Dictionary containing evaluation metrics (ndcg, map, recall, precision)
    additional_info: Optional dictionary for any additional information to store
    """
    collection = db["evaluation_results"]

    result_doc = {
        "timestamp": datetime.utcnow(),
        "search_method": search_method,
        "metrics": {},
    }

    for metric_name, metric_values in metrics.items():
        result_doc["metrics"][metric_name] = metric_values

    if additional_info:
        result_doc.update(additional_info)

    insert_result = collection.insert_one(result_doc)

    logger.debug(
        f"Evaluation results for {search_method} stored with ID: {insert_result.inserted_id}"
    )

metadata = {
    "dataset_name": DATASET,
    "corpus_size": len(corpus),
    "num_queries": len(queries),
    "num_qrels": sum(len(q) for q in qrels.values()),
}

information_retrieval_eval_metrics_list = [
    lexical_search_metric_dicts,
    vector_search_metric_dicts,
    hybrid_search_metric_dicts,
]

for search_method, metrics in zip(
    information_retrieval_search_methods, information_retrieval_eval_metrics_list
):
    store_evaluation_results(db, search_method, metrics, metadata)

"""
# **Evaluating on the Financial Opinion Mining and Question Answering (FIQA) dataset**



---
"""
logger.info("# **Evaluating on the Financial Opinion Mining and Question Answering (FIQA) dataset**")

DATASET = "fiqa"
corpus, queries, qrels = load_beir_dataset(DATASET)

logger.debug(list(corpus.items())[0])
logger.debug()
logger.debug(list(queries.items())[0])
logger.debug()
logger.debug(list(qrels.items())[0])

fiqa_corpus = "fiqa_corpus"

ingest_data(db, corpus=corpus, corpus_collection_name=fiqa_corpus)

setup_vector_search_index_with_filter(
    db[fiqa_corpus], corpus_vector_search_index_definition
)

corpus_text_index_definition = {
    "mappings": {"dynamic": True, "fields": {"text": {"type": "string"}}}
}

create_collection_search_index(
    db[fiqa_corpus], corpus_text_index_definition, TEXT_SEARCH_INDEX
)

generate_and_store_embeddings(corpus, db, fiqa_corpus)

lexical_search = MongoDBSearch(db[fiqa_corpus], "text_search_index")

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=f"{DB_NAME}.{fiqa_corpus}",
    embedding=embedding_model,
    index_name="vector_index",
    text_key="text",
)

vector_search = MongoDBVectorSearch(vector_store, embedding_model)

hybrid_search = MongoDBHybridSearch(vector_store, "text_search_index")

def evaluate_search_method(search_method, method_name):
    retriever = EvaluateRetrieval(search_method, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    metrics = retriever.evaluate(qrels, results, retriever.k_values)

    logger.debug("Sample of retrieved results:")
    for query_id, doc_scores in list(results.items())[:5]:
        logger.debug(f"Query ID: {query_id}")
        logger.debug(f"Query text: {queries[query_id]}")
        logger.debug("Top 3 retrieved documents:")
        for doc_id, score in list(doc_scores.items())[:3]:
            logger.debug(f"  Doc ID: {doc_id}, Score: {score}")
        logger.debug()

    logger.debug(f"\nResults for {method_name}:")
    ndcg, _map, recall, precision = metrics
    for metric, values in zip(
        ["NDCG", "MAP", "Recall", "Precision"], [ndcg, _map, recall, precision]
    ):
        logger.debug(f"{metric}:")
        for k, v in values.items():
            logger.debug(f"  {k}: {v:.4f}")

    store_evaluation_results(
        db,
        method_name,
        {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision},
        {"dataset": "FiQA"},
    )

    return [ndcg, _map, recall, precision]

lexical_search_metric_dicts = evaluate_search_method(lexical_search, "Lexical Search")
vector_search_metric_dicts = evaluate_search_method(vector_search, "Vector Search")
hybrid_search_metric_dicts = evaluate_search_method(hybrid_search, "Hybrid Search")

plot_search_method_comparison(
    lexical_search_metric_dicts,
    vector_search_metric_dicts,
    hybrid_search_metric_dicts,
    metric_names,
)

logger.info("\n\n[DONE]", bright=True)