from jet.models.config import MODELS_CACHE_DIR
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.finetuning import generate_qa_embedding_pairs
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from tqdm.notebook import tqdm
import json
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/finetuning/embeddings/finetune_embedding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Finetune Embeddings

In this notebook, we show users how to finetune their own embedding models.

We go through three main sections:
1. Preparing the data (our `generate_qa_embedding_pairs` function makes this easy)
2. Finetuning the model (using our `SentenceTransformersFinetuneEngine`)
3. Evaluating the model on a validation knowledge corpus

## Generate Corpus

First, we create the corpus of text chunks by leveraging LlamaIndex to load some financial PDFs, and parsing/chunking into plain text chunks.
"""
logger.info("# Finetune Embeddings")

# %pip install datasets
# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-finetuning
# %pip install llama-index-readers-file
# %pip install llama-index-embeddings-huggingface
# %pip install "transformers[torch]"



"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

TRAIN_FILES = ["./data/10k/lyft_2021.pdf"]
VAL_FILES = ["./data/10k/uber_2021.pdf"]

TRAIN_CORPUS_FPATH = "./data/train_corpus.json"
VAL_CORPUS_FPATH = "./data/val_corpus.json"

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
We do a very naive train/val split by having the Lyft corpus as the train dataset, and the Uber corpus as the val dataset.
"""
logger.info("We do a very naive train/val split by having the Lyft corpus as the train dataset, and the Uber corpus as the val dataset.")

train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

"""
### Generate synthetic queries

Now, we use an LLM (gpt-3.5-turbo) to generate questions using each text chunk in the corpus as context.

Each pair of (generated question, text chunk used as context) becomes a datapoint in the finetuning dataset (either for training or evaluation).
"""
logger.info("### Generate synthetic queries")



# OPENAI_API_KEY = "sk-"
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



train_dataset = generate_qa_embedding_pairs(
    llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    nodes=train_nodes,
    output_path="train_dataset.json",
)
val_dataset = generate_qa_embedding_pairs(
    llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    nodes=val_nodes,
    output_path="val_dataset.json",
)

train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

"""
## Run Embedding Finetuning
"""
logger.info("## Run Embedding Finetuning")


finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="test_model",
    val_dataset=val_dataset,
)

finetune_engine.finetune()

embed_model = finetune_engine.get_finetuned_model()

embed_model

"""
## Evaluate Finetuned Model

In this section, we evaluate 3 different embedding models: 
1. proprietary OllamaFunctionCallingAdapter embedding,
2. open source `BAAI/bge-small-en`, and
3. our finetuned embedding model.

We consider 2 evaluation approaches:
1. a simple custom **hit rate** metric
2. using `InformationRetrievalEvaluator` from sentence_transformers

We show that finetuning on synthetic (LLM-generated) dataset significantly improve upon an opensource embedding model.
"""
logger.info("## Evaluate Finetuned Model")


"""
### Define eval function

**Option 1**: We use a simple **hit rate** metric for evaluation:
* for each (query, relevant_doc) pair,
* we retrieve top-k documents with the query,  and 
* it's a **hit** if the results contain the relevant_doc.

This approach is very simple and intuitive, and we can apply it to both the proprietary OllamaFunctionCallingAdapter embedding as well as our open source and fine-tuned embedding models.
"""
logger.info("### Define eval function")

def evaluate(
    dataset,
    embed_model,
    top_k=5,
    verbose=False,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, show_progress=True
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return eval_results

"""
**Option 2**: We use the `InformationRetrievalEvaluator` from sentence_transformers.

This provides a more comprehensive suite of metrics, but we can only run it against the sentencetransformers compatible models (open source and our finetuned model, *not* the OllamaFunctionCallingAdapter embedding model).
"""
logger.info("This provides a more comprehensive suite of metrics, but we can only run it against the sentencetransformers compatible models (open source and our finetuned model, *not* the OllamaFunctionCallingAdapter embedding model).")



def evaluate_st(
    dataset,
    model_id,
    name,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, name=name
    )
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)

"""
### Run Evals

#### OllamaFunctionCallingAdapter

Note: this might take a few minutes to run since we have to embed the corpus and queries
"""
logger.info("### Run Evals")

ada = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
ada_val_results = evaluate(val_dataset, ada)

df_ada = pd.DataFrame(ada_val_results)

hit_rate_ada = df_ada["is_hit"].mean()
hit_rate_ada

"""
### BAAI/bge-small-en
"""
logger.info("### BAAI/bge-small-en")

bge = "local:BAAI/bge-small-en"
bge_val_results = evaluate(val_dataset, bge)

df_bge = pd.DataFrame(bge_val_results)

hit_rate_bge = df_bge["is_hit"].mean()
hit_rate_bge

evaluate_st(val_dataset, "BAAI/bge-small-en", name="bge")

"""
### Finetuned
"""
logger.info("### Finetuned")

finetuned = "local:test_model"
val_results_finetuned = evaluate(val_dataset, finetuned)

df_finetuned = pd.DataFrame(val_results_finetuned)

hit_rate_finetuned = df_finetuned["is_hit"].mean()
hit_rate_finetuned

evaluate_st(val_dataset, "test_model", name="finetuned")

"""
### Summary of Results

#### Hit rate
"""
logger.info("### Summary of Results")

df_ada["model"] = "ada"
df_bge["model"] = "bge"
df_finetuned["model"] = "fine_tuned"

"""
We can see that fine-tuning our small open-source embedding model drastically improve its retrieval quality (even approaching the quality of the proprietary OllamaFunctionCallingAdapter embedding)!
"""
logger.info("We can see that fine-tuning our small open-source embedding model drastically improve its retrieval quality (even approaching the quality of the proprietary OllamaFunctionCallingAdapter embedding)!")

df_all = pd.concat([df_ada, df_bge, df_finetuned])
df_all.groupby("model").mean("is_hit")

"""
#### InformationRetrievalEvaluator
"""
logger.info("#### InformationRetrievalEvaluator")

df_st_bge = pd.read_csv(
    "results/Information-Retrieval_evaluation_bge_results.csv"
)
df_st_finetuned = pd.read_csv(
    "results/Information-Retrieval_evaluation_finetuned_results.csv"
)

"""
We can see that embedding finetuning improves metrics consistently across the suite of eval metrics
"""
logger.info("We can see that embedding finetuning improves metrics consistently across the suite of eval metrics")

df_st_bge["model"] = "bge"
df_st_finetuned["model"] = "fine_tuned"
df_st_all = pd.concat([df_st_bge, df_st_finetuned])
df_st_all = df_st_all.set_index("model")
df_st_all

logger.info("\n\n[DONE]", bright=True)