from jet.transformers.formatters import format_json
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import (
generate_question_context_pairs,
EmbeddingQAFinetuneDataset,
)
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.embeddings.cohere import CohereEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/cohere_retriever_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Cohere init8 and binary Embeddings Retrieval Evaluation

Cohere Embed is the first embedding model that natively supports float, int8, binary and ubinary embeddings. Refer to their [main blog post](https://txt.cohere.com/int8-binary-embeddings/) for more details on Cohere int8 & binary Embeddings.

This notebook helps you to evaluate these different embedding types and pick one for your RAG pipeline. It uses our `RetrieverEvaluator` to evaluate the quality of the embeddings using the Retriever module LlamaIndex.

Observed Metrics:

1. Hit-Rate
2. MRR (Mean-Reciprocal-Rank)

For any given question, these will compare the quality of retrieved results from the ground-truth context. The eval dataset is created using our synthetic dataset generation module. We will use GPT-4 for dataset generation to avoid bias.

# Note: The results shown at the end of the notebook are very specific to dataset, and various other parameters considered. We recommend you to use the notebook as reference to experiment on your dataset and evaluate the usage of different embedding types in your RAG pipeline.

## Installation
"""
logger.info("# Cohere init8 and binary Embeddings Retrieval Evaluation")

# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-cohere

"""
## Setup API Keys
"""
logger.info("## Setup API Keys")


# os.environ["OPENAI_API_KEY"] = "YOUR OPENAI KEY"
os.environ["COHERE_API_KEY"] = "YOUR COHEREAI API KEY"

"""
## Setup

Here we load in data (PG essay), parse into Nodes. We then index this data using our simple vector index and get a retriever for the following different embedding types.

1. `float`
2. `int8`
3. `binary`
4. `ubinary`
"""
logger.info("## Setup")

# import nest_asyncio

# nest_asyncio.apply()


"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data
"""
logger.info("## Load Data")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Create Nodes
"""
logger.info("## Create Nodes")

node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

"""
## Create retrievers for different embedding types
"""
logger.info("## Create retrievers for different embedding types")

llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)


def cohere_embedding(
    model_name: str, input_type: str, embedding_type: str
) -> CohereEmbedding:
    return CohereEmbedding(
        api_key=os.environ["COHERE_API_KEY"],
        model_name=model_name,
        input_type=input_type,
        embedding_type=embedding_type,
    )


def retriver(nodes, embedding_type="float", model_name="embed-english-v3.0"):
    vector_index = VectorStoreIndex(
        nodes,
        embed_model=cohere_embedding(
            model_name, "search_document", embedding_type
        ),
    )
    retriever = vector_index.as_retriever(
        similarity_top_k=2,
        embed_model=cohere_embedding(
            model_name, "search_query", embedding_type
        ),
    )
    return retriever

retriver_float = retriver(nodes)

retriver_int8 = retriver(nodes, "int8")

retriver_binary = retriver(nodes, "binary")

retriver_ubinary = retriver(nodes, "ubinary")

"""
### Try out Retrieval

We'll try out retrieval over a sample query with `float` retriever.
"""
logger.info("### Try out Retrieval")

retrieved_nodes = retriver_float.retrieve("What did the author do growing up?")


for node in retrieved_nodes:
    display_source_node(node, source_length=1000)

"""
## Evaluation dataset - Synthetic Dataset Generation of (query, context) pairs

Here we build a simple evaluation dataset over the existing text corpus.

We use our `generate_question_context_pairs` to generate a set of (question, context) pairs over a given unstructured text corpus. This uses the LLM to auto-generate questions from each context chunk.

We get back a `EmbeddingQAFinetuneDataset` object. At a high-level this contains a set of ids mapping to queries and relevant doc chunks, as well as the corpus itself.
"""
logger.info("## Evaluation dataset - Synthetic Dataset Generation of (query, context) pairs")


qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=2
)

queries = qa_dataset.queries.values()
logger.debug(list(queries)[0])

qa_dataset.save_json("pg_eval_dataset.json")

qa_dataset = EmbeddingQAFinetuneDataset.from_json("pg_eval_dataset.json")

"""
## Use `RetrieverEvaluator` for Retrieval Evaluation

We're now ready to run our retrieval evals. We'll run our `RetrieverEvaluator` over the eval dataset that we generated.

### Define `RetrieverEvaluator` for different embedding_types
"""
logger.info("## Use `RetrieverEvaluator` for Retrieval Evaluation")


metrics = ["mrr", "hit_rate"]

retriever_evaluator_float = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriver_float
)

retriever_evaluator_int8 = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriver_int8
)

retriever_evaluator_binary = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriver_binary
)

retriever_evaluator_ubinary = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriver_ubinary
)

sample_id, sample_query = list(qa_dataset.queries.items())[0]
sample_expected = qa_dataset.relevant_docs[sample_id]

eval_result = retriever_evaluator_float.evaluate(sample_query, sample_expected)
logger.debug(eval_result)

eval_results_float = retriever_evaluator_float.evaluate_dataset(
        qa_dataset
    )
logger.success(format_json(eval_results_float))

eval_results_int8 = retriever_evaluator_int8.evaluate_dataset(
        qa_dataset
    )
logger.success(format_json(eval_results_int8))

eval_results_binary = retriever_evaluator_binary.evaluate_dataset(
        qa_dataset
    )
logger.success(format_json(eval_results_binary))

eval_results_ubinary = retriever_evaluator_ubinary.evaluate_dataset(
        qa_dataset
    )
logger.success(format_json(eval_results_ubinary))

"""
#### Define `display_results` to get the display the results in dataframe with each retriever.
"""
logger.info("#### Define `display_results` to get the display the results in dataframe with each retriever.")



def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    columns = {"Embedding Type": [name], "hit_rate": [hit_rate], "mrr": [mrr]}

    metric_df = pd.DataFrame(columns)

    return metric_df

"""
## Evaluation Results
"""
logger.info("## Evaluation Results")

metrics_float = display_results("float", eval_results_float)

metrics_int8 = display_results("int8", eval_results_int8)

metrics_binary = display_results("binary", eval_results_binary)

metrics_ubinary = display_results("ubinary", eval_results_ubinary)

combined_metrics = pd.concat(
    [metrics_float, metrics_int8, metrics_binary, metrics_ubinary]
)
combined_metrics.set_index(["Embedding Type"], append=True, inplace=True)

combined_metrics

"""
# Note: The results shown above are very specific to dataset, and various other parameters considered. We recommend you to use the notebook as reference to experiment on your dataset and evaluate the usage of different embedding types in your RAG pipeline.
"""
logger.info("# Note: The results shown above are very specific to dataset, and various other parameters considered. We recommend you to use the notebook as reference to experiment on your dataset and evaluate the usage of different embedding types in your RAG pipeline.")

logger.info("\n\n[DONE]", bright=True)