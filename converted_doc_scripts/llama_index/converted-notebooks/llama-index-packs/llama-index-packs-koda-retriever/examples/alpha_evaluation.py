import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import generate_qa_embedding_pairs
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.packs.koda_retriever import KodaRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import pandas as pd
import shutil


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
# Evaluating RAG w/ Alpha Tuning
Evaluation is a crucial piece of development when building a RAG pipeline. Likewise, alpha tuning can be a time consuming exercise to build out, so what does the performance benefit look like for all of this extra effort? Let's dig into that.

### Fixtures
- For our dataset: Several research papers on AI (same used in the blog post referenced below) /data
- Our vector db: [Pinecone](https://www.pinecone.io/)
- For our embedding model: [ada-002](https://platform.openai.com/docs/models/embeddings) 
- For our LLM: [GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3-5-turbo)

These fixtures were chosen because they're well integrated within LlamaIndex to make this notebook very transferrable to those looking to reproduce this.
Likewise, Pinecone supports hybrid search, which is a requirement for hybrid searches to be possible. Koda Retriever will also be used!

### Testing

Koda Retriever was largely inspired from the alpha tuning [blog post written by Ravi Theja](https://blog.llamaindex.ai/llamaindex-enhancing-retrieval-performance-with-alpha-tuning-in-hybrid-search-in-rag-135d0c9b8a00) from Llama Index. For that reason, we'll follow a similar pattern and evaluate with:
- MRR (Mean Reciprocal Rank)
- Hit Rate

### Agenda:
- Fixture Setup
- Data Ingestion
- Synthetic Query Generation
- Alpha Mining & Evaluation
- Koda Retriever vs Vanilla Hybrid Retriever
"""
logger.info("# Evaluating RAG w/ Alpha Tuning")


"""
## Fixture Setup
"""
logger.info("## Fixture Setup")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("llama2-paper")  # this was previously created in my pinecone account

Settings.llm = MLXLlamaIndexLLMAdapter()
Settings.embed_model = MLXEmbedding()

vector_store = PineconeVectorStore(pinecone_index=index)
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=Settings.embed_model
)

reranker = LLMRerank(llm=Settings.llm)  # optional

koda_retriever = KodaRetriever(
    index=vector_index,
    llm=Settings.llm,
    reranker=reranker,  # optional
    verbose=True,
    similarity_top_k=10,
)

vanilla_retriever = vector_index.as_retriever()

pipeline = IngestionPipeline(
    transformations=[Settings.embed_model], vector_store=vector_store
)

"""
## Data Ingestion

Three research papers in `/data` are going to be ingested into our Pinecone instance.

Our chunking strategy will solely be [semantic](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking.html) - although this is not recommended for production. For production use cases it is recommended more analysis and other chunking strategies are considered for production use cases.
"""
logger.info("## Data Ingestion")

def load_documents(file_path, num_pages=None):
    if num_pages:
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()[
            :num_pages
        ]
    else:
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    return documents


doc1 = load_documents(
    "/workspaces/llama_index/llama-index-packs/llama-index-packs-koda-retriever/examples/data/dense_x_retrieval.pdf",
    num_pages=9,
)
doc2 = load_documents(
    "/workspaces/llama_index/llama-index-packs/llama-index-packs-koda-retriever/examples/data/llama_beyond_english.pdf",
    num_pages=7,
)
doc3 = load_documents(
    "/workspaces/llama_index/llama-index-packs/llama-index-packs-koda-retriever/examples/data/llm_compiler.pdf",
    num_pages=12,
)
docs = [doc1, doc2, doc3]
nodes = list()

node_parser = SemanticSplitterNodeParser(
    embed_model=Settings.embed_model, breakpoint_percentile_threshold=95
)
for doc in docs:
    _nodes = node_parser.build_semantic_nodes_from_documents(
        documents=doc,
    )
    nodes.extend(_nodes)

    pipeline.run(nodes=_nodes)

"""
## Synthetic Query Generation
"""
logger.info("## Synthetic Query Generation")

qa_dataset = generate_qa_embedding_pairs(nodes=nodes, llm=Settings.llm)

"""
## Alpha Mining & Evaluation

We're going to update the alpha values of a vector index retriever right before evaluation. 
We'll be evaluating alpha values in increments of .1; 0 to 1 where 0 is basic text search and 1 is a pure vector search.
"""
logger.info("## Alpha Mining & Evaluation")

def calculate_metrics(eval_results):
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    return hit_rate, mrr


async def alpha_mine(
    qa_dataset,
    vector_store_index,
    alpha_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
):
    retriever = VectorIndexRetriever(
        index=vector_store_index,
        vector_store_query_mode="hybrid",
        alpha=0.0,  # this will change
        similarity_top_k=10,
    )

    results = dict()

    for alpha in alpha_values:
        retriever._alpha = alpha
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            metric_names=["mrr", "hit_rate"], retriever=retriever
        )
        async def run_async_code_a33154eb():
            async def run_async_code_65982972():
                eval_results = retriever_evaluator.evaluate_dataset(dataset=qa_dataset)
                return eval_results
            eval_results = asyncio.run(run_async_code_65982972())
            logger.success(format_json(eval_results))
            return eval_results
        eval_results = asyncio.run(run_async_code_a33154eb())
        logger.success(format_json(eval_results))

        hit_rate, mrr = calculate_metrics(eval_results)

        results[alpha] = {"hit_rate": hit_rate, "mrr": mrr}

    return results


async def run_async_code_cdb393ae():
    async def run_async_code_f51cc4ee():
        results = await alpha_mine(qa_dataset=qa_dataset, vector_store_index=vector_index)
        return results
    results = asyncio.run(run_async_code_f51cc4ee())
    logger.success(format_json(results))
    return results
results = asyncio.run(run_async_code_cdb393ae())
logger.success(format_json(results))
results

"""
### Conclusions

As seen above, the alpha values between .1 and 1 are basically the same. This was a single dataset that was tested on and more than likely you'll want to test on multiple datasets for multiple purposes. For our purposes, the default .5 for most hybrid retrievers probably works well - although as shown in the [original blog post](https://blog.llamaindex.ai/llamaindex-enhancing-retrieval-performance-with-alpha-tuning-in-hybrid-search-in-rag-135d0c9b8a00) that started this llama pack, datasets can have wildly different results among their alpha values.

## Bonus: Koda Retriever vs Vanilla Hybrid Retriever

Finally, we'll evaluate and compare a vanilla hybrid retriever against our koda retriever.
This koda retriever will use default alpha values and categories provided in the alpha pack.
"""
logger.info("### Conclusions")

async def compare_retrievers(retrievers, qa_dataset):
    results = dict()

    for name, retriever in retrievers.items():
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            metric_names=["mrr", "hit_rate"], retriever=retriever
        )

        async def run_async_code_a33154eb():
            async def run_async_code_65982972():
                eval_results = retriever_evaluator.evaluate_dataset(dataset=qa_dataset)
                return eval_results
            eval_results = asyncio.run(run_async_code_65982972())
            logger.success(format_json(eval_results))
            return eval_results
        eval_results = asyncio.run(run_async_code_a33154eb())
        logger.success(format_json(eval_results))

        hit_rate, mrr = calculate_metrics(eval_results)

        results[name] = {"hit_rate": hit_rate, "mrr": mrr}

    return results


retrievers = {"vanilla": vanilla_retriever, "koda": koda_retriever}

async def run_async_code_0aaf936a():
    async def run_async_code_611ef2dc():
        results = await compare_retrievers(retrievers=retrievers, qa_dataset=qa_dataset)
        return results
    results = asyncio.run(run_async_code_611ef2dc())
    logger.success(format_json(results))
    return results
results = asyncio.run(run_async_code_0aaf936a())
logger.success(format_json(results))
results

logger.info("\n\n[DONE]", bright=True)