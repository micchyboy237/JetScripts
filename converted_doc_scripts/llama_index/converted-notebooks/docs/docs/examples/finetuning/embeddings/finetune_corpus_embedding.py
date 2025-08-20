from datasets import load_dataset
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.experimental import Nudge
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.finetuning import generate_qa_embedding_pairs
from tqdm import tqdm
from typing import Optional, Dict
import numpy as np
import os
import shutil
import torch


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Finetuning corpus embeddings using NUDGE
[NUDGE](https://www.arxiv.org/abs/2409.02343) is a novel simple and lightweight fine-tuning method that boosts accuracy when retrieving text using semantic similarity with pre-trained embedding models. NUDGE directly modifies the embeddings of data records to maximize the similarity between training queries and their ground-truth answers. NUDGE does so non-parametrically. Non-parametric means that NUDGE does not modify model parameters to generate better embeddings, as fine-tuning the embedding model, or training adaptors would. Instead, NUDGE directly changes the embeddings themselves. Compared with fine-tuning the pre-trained model and training adaptors, NUDGE provides 3.3x and 4.3x higher increase in accuracy and runs 200x and 3x faster, respectively. [Here](https://data-people-group.github.io/blogs/2024/09/05/nudge/) is a blog post on NUDGE, and [here](https://www.arxiv.org/abs/2409.02343) is the paper with more details.

We demonstrate NUDGE's effectiveness on a commonly used Information Retrieval benchmark called Scifact.
"""
logger.info("# Finetuning corpus embeddings using NUDGE")

# %pip install llama-index-experimental llama-index-embeddings-huggingface nudge-ft torch datasets

"""
## Load the scifact benchmark
"""
logger.info("## Load the scifact benchmark")



def load_hf_dataset(dataset_name):
    hf_dataset_name = f"sepz/{dataset_name}_ft"
    corpus = load_dataset(hf_dataset_name, "data_records", split="train")

    queries_train = load_dataset(hf_dataset_name, "qs", split="train")
    queries_validation = load_dataset(hf_dataset_name, "qs", split="dev")
    queries_test = load_dataset(hf_dataset_name, "qs", split="test")

    qrels_train = load_dataset(hf_dataset_name, "qs_rel", split="train")
    qrels_validation = load_dataset(hf_dataset_name, "qs_rel", split="dev")
    qrels_test = load_dataset(hf_dataset_name, "qs_rel", split="test")

    corpus = {
        str(corpus[i]["record_id"]): corpus[i]["text"]
        for i in range(len(corpus))
    }

    queries_train = {
        str(queries_train[i]["q_id"]): queries_train[i]["input"]
        for i in range(len(queries_train))
    }
    queries_validation = {
        str(r["q_id"]): r["input"] for r in queries_validation
    }
    queries_test = {str(r["q_id"]): r["input"] for r in queries_test}

    qrels_train = (
        qrels_train.to_pandas()
        .groupby("q_id")["record_id"]
        .apply(list)
        .to_dict()
    )
    qrels_validation = (
        qrels_validation.to_pandas()
        .groupby("q_id")["record_id"]
        .apply(list)
        .to_dict()
    )
    qrels_test = (
        qrels_test.to_pandas()
        .groupby("q_id")["record_id"]
        .apply(list)
        .to_dict()
    )
    qrels_train = {str(k): [str(i) for i in v] for k, v in qrels_train.items()}
    qrels_validation = {
        str(k): [str(i) for i in v] for k, v in qrels_validation.items()
    }
    qrels_test = {str(k): [str(i) for i in v] for k, v in qrels_test.items()}

    train_dataset = EmbeddingQAFinetuneDataset(
        corpus=corpus, queries=queries_train, relevant_docs=qrels_train
    )
    validation_dataset = EmbeddingQAFinetuneDataset(
        corpus=corpus,
        queries=queries_validation,
        relevant_docs=qrels_validation,
    )
    test_dataset = EmbeddingQAFinetuneDataset(
        corpus=corpus, queries=queries_test, relevant_docs=qrels_test
    )

    return train_dataset, validation_dataset, test_dataset

"""
## Load the dataset and base embedding model
"""
logger.info("## Load the dataset and base embedding model")


train_dataset, val_dataset, test_dataset = load_hf_dataset("scifact")
base_embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

"""
If we take a peek at the dataset, we can see that its structured as
- courpus: mapping of document ID to text
- queries: mapping of query ID to query text
- relevant_docs: a mapping of query ID to list of document IDs
"""
logger.info("If we take a peek at the dataset, we can see that its structured as")

logger.debug(val_dataset.queries["2"])

logger.debug(val_dataset.relevant_docs["2"])

logger.debug(val_dataset.corpus["552"])

"""
### Using your own Datasets

As you can see, you can run this notebook on any dataset, as long as you have queries and a mapping to relevant documents! If you have documents but are missing a training set of queries checkout the our tools for generating a synthetic dataset ([1](https://docs.llamaindex.ai/en/stable/api_reference/evaluation/dataset_generation/)).

If you wanted, you could also write your own dataset, or even use llama-index to create your own.

Uncomment the code below and add your own files if you want to try it out.
"""
logger.info("### Using your own Datasets")



def load_corpus(files, verbose=False):
    if verbose:
        logger.debug(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        logger.debug(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        logger.debug(f"Parsed {len(nodes)} nodes")

    return nodes

"""
## Evaluation
A common Information Retrieval metric to report during evaluation is NDCG@k.
"""
logger.info("## Evaluation")




def build_retriever(
    corpus: Dict[str, str],
    embed_model: BaseEmbedding | str,
    corpus_embeddings: Optional[torch.Tensor] = None,
    k: int = 10,
) -> BaseRetriever:
    nodes = []
    for i, (id_, text) in enumerate(corpus.items()):
        if corpus_embeddings is not None:
            nodes.append(
                TextNode(
                    id_=id_, text=text, embedding=corpus_embeddings[i].tolist()
                )
            )
        else:
            nodes.append(TextNode(id_=id_, text=text))

    index = VectorStoreIndex(
        nodes=nodes,
        embeddings=corpus_embeddings,
        embed_model=embed_model,
        show_progress=True,
    )
    return index.as_retriever(similarity_top_k=k)


def ndcg_at_k(
    dataset: EmbeddingQAFinetuneDataset, retriever: BaseRetriever, k: int = 10
):
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs
    ndcg_scores = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_ids = relevant_docs[query_id]

        ideal_dcg = np.sum(
            [1 / np.log2(i + 2) for i in range(min(k, len(expected_ids)))]
        )
        rel_scores = np.zeros(k)
        for j in range(min(k, len(retrieved_ids))):
            if retrieved_ids[j] in expected_ids:
                rel_scores[j] = 1
        dcg = np.sum(
            [rel_scores[i] / np.log2(i + 2) for i in range(len(rel_scores))]
        )
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

        ndcg_scores.append(ndcg)

    mean_ndcg = np.mean(ndcg_scores)
    return mean_ndcg

"""
## Get the corpus embedding finetuning results
Next we use, [NUDGE](https://www.arxiv.org/abs/2409.02343), the state of the art method for finetuning corpus embeddings to maximize the accuracy of k-NN retrieval. We then take our new corpus embeddings along with the original embedding model to build a retriever. NUDGE only finetunes the corpus embeddings and does not change any of the parameters in the base embedding model.
"""
logger.info("## Get the corpus embedding finetuning results")

# %%capture

k = 10

nudge = Nudge(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    embed_model=base_embed_model,
    use_nudge_n=True,
)
nudge.finetune()
nudge_corpus_embeddings = nudge.get_finetuned_corpus_embeddings()
nudge_retriever = build_retriever(
    train_dataset.corpus, base_embed_model, nudge_corpus_embeddings, k=k
)
nudge_ndcg_test = ndcg_at_k(test_dataset, nudge_retriever, k)

"""
## Get the adapter finetuning results
"""
logger.info("## Get the adapter finetuning results")

# %%capture

embedding_adapater_finetune_engine = EmbeddingAdapterFinetuneEngine(
    train_dataset,
    base_embed_model,
    epochs=4,
    batch_size=10,
)
embedding_adapater_finetune_engine.finetune()
embedding_adapter_model = (
    embedding_adapater_finetune_engine.get_finetuned_model()
)
ft_retriever = build_retriever(
    train_dataset.corpus, embedding_adapter_model, k=k
)
ft_ndcg_test = ndcg_at_k(test_dataset, ft_retriever, k)

"""
## Get the baseline results
"""
logger.info("## Get the baseline results")

# %%capture

base_retriever = build_retriever(train_dataset.corpus, base_embed_model, k=k)
bge_ndcg_test = ndcg_at_k(test_dataset, base_retriever, k)

"""
## Display the results
"""
logger.info("## Display the results")

logger.debug(f"bge test - ndcg@10: {bge_ndcg_test:.2f}")
logger.debug(f"adaptor finetune test - ndcg@10: {ft_ndcg_test:.2f}")
logger.debug(f"NUDGE-N test - ndcg@10: {nudge_ndcg_test:.2f}")

"""
# Inserting records into the dataset
It's common to have your dataset expand over time. We will now insert and finetune the nfcorpus into the scifact example we've been working with. Usually you'd have to retrain on your entire dataset to avoid catastrophic forgetting. With NUDGE, you can easily expand your dataset iteratively by focusing only on the newest batch of data, without worrying about catastrophic forgetting. This only works when the new data being inserted does not conflict (e.g. new queries for old corpus or new corpus changes k-NN to old queries) with the existing dataset.
"""
logger.info("# Inserting records into the dataset")

# %%capture

new_train_dataset, new_val_dataset, new_test_dataset = load_hf_dataset(
    "nfcorpus"
)

new_train_dataset.queries = {
    f"nfcorpus-{k}": v for k, v in new_train_dataset.queries.items()
}
new_train_dataset.relevant_docs = {
    f"nfcorpus-{k}": [f"nfcorpus-{doc_id}" for doc_id in v]
    for k, v in new_train_dataset.relevant_docs.items()
}
new_train_dataset.corpus = {
    f"nfcorpus-{k}": v for k, v in new_train_dataset.corpus.items()
}

new_val_dataset.queries = {
    f"nfcorpus-{k}": v for k, v in new_val_dataset.queries.items()
}
new_val_dataset.relevant_docs = {
    f"nfcorpus-{k}": [f"nfcorpus-{doc_id}" for doc_id in v]
    for k, v in new_val_dataset.relevant_docs.items()
}
new_val_dataset.corpus = {
    f"nfcorpus-{k}": v for k, v in new_val_dataset.corpus.items()
}

new_test_dataset.queries = {
    f"nfcorpus-{k}": v for k, v in new_test_dataset.queries.items()
}
new_test_dataset.relevant_docs = {
    f"nfcorpus-{k}": [f"nfcorpus-{doc_id}" for doc_id in v]
    for k, v in new_test_dataset.relevant_docs.items()
}
new_test_dataset.corpus = {
    f"nfcorpus-{k}": v for k, v in new_test_dataset.corpus.items()
}

"""
## Finetune the new records
"""
logger.info("## Finetune the new records")

# %%capture

nudge.insert_data_and_finetune(
    new_train_dataset_batch=new_train_dataset,
    new_val_dataset_batch=new_val_dataset,
)
nudge_corpus_embeddings = nudge.get_finetuned_corpus_embeddings()
aggregated_corpus = {**train_dataset.corpus, **new_train_dataset.corpus}
nudge_retriever = build_retriever(
    aggregated_corpus, base_embed_model, nudge_corpus_embeddings, k=k
)
nudge_ndcg_nfcorpus_test = ndcg_at_k(new_test_dataset, nudge_retriever, k)
nudge_ndcg_scifact_test = ndcg_at_k(test_dataset, nudge_retriever, k)

"""
## Display the insertion results
Check the results on our newly inserted nfcorpus records and verify that our scifact benchmark did not regress.
"""
logger.info("## Display the insertion results")

logger.debug(
    f"NUDGE-N (aggregated) test on nfcorpus - ndcg@10: {nudge_ndcg_nfcorpus_test:.2f}"
)
logger.debug(
    f"NUDGE-N (aggregated) test on scifact - ndcg@10: {nudge_ndcg_scifact_test:.2f}"
)

logger.info("\n\n[DONE]", bright=True)