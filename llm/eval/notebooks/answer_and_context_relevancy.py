# %pip install llama-index-llms-openai
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio

nest_asyncio.apply()
def displayify_df(df):
    """For pretty displaying DataFrame in a notebook."""
    display_df = df.style.set_properties(
        **{
            "inline-size": "300px",
            "overflow-wrap": "break-word",
        }
    )
    display(display_df)
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex

import joblib
from llama_index.core.llama_dataset import BaseLlamaDataset, BaseLlamaPredictionDataset
from llama_index.core.schema import Document


data_cache_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/improved/cache/answer_and_context_relevancy/data.pkl"
def get_cached_dataset() -> tuple[BaseLlamaDataset, list[Document]]:
    return joblib.load(data_cache_path)

rag_dataset, documents = get_cached_dataset()
documents = documents[:1]
rag_dataset.examples = rag_dataset.examples[:2]

{"Documents": len(documents), "Dataset": len(rag_dataset.examples)}
documents[0].text
rag_dataset.to_pandas()[:5]
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()
prediction_dataset: BaseLlamaPredictionDataset = rag_dataset.make_predictions_with(
    predictor=query_engine, batch_size=1, show_progress=True
)
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)

from jet.llm.ollama import (
    update_llm_settings,
    create_embed_model,
    create_llm,
    small_llm_model,
    large_llm_model,
    large_embed_model,
)

settings = update_llm_settings({
    "llm_model": large_llm_model,
    "embedding_model": large_embed_model,
})

judges = {}

judges["answer_relevancy"] = AnswerRelevancyEvaluator(
    llm=create_llm(small_llm_model),
)

judges["context_relevancy"] = ContextRelevancyEvaluator(
    llm=create_llm(large_llm_model),
)
from tqdm import tqdm

batch_size = 1

eval_iterator = tqdm(zip(rag_dataset.examples, prediction_dataset.predictions),
                             total=len(prediction_dataset.predictions) * batch_size)
eval_tasks = []
for example, prediction in eval_iterator:
    eval_tasks.append(
        judges["answer_relevancy"].evaluate(
            query=example.query,
            response=prediction.response,
            sleep_time_in_seconds=1.0,
        )
    )
    eval_tasks.append(
        judges["context_relevancy"].evaluate(
            query=example.query,
            contexts=prediction.contexts,
            sleep_time_in_seconds=1.0,
        )
    )
eval_results1 = eval_tasks[0]
eval_results2 = eval_tasks[1]
eval_results = [eval_results1, eval_results2]
eval_results
evals = {
    "answer_relevancy": eval_results[::2],
    "context_relevancy": eval_results[1::2],
}
evals
from llama_index.core.evaluation.notebook_utils import get_eval_results_df
import pandas as pd
import json

deep_dfs = {}
mean_dfs = {}
for metric in evals.keys():
    deep_df, mean_df = get_eval_results_df(
        names=["baseline"] * len(evals[metric]),
        results_arr=evals[metric],
        metric=metric,
    )
    deep_dfs[metric] = deep_df
    mean_dfs[metric] = mean_df

mean_dfs
mean_scores_df = pd.concat(
    [mdf.reset_index() for _, mdf in mean_dfs.items()],
    axis=0,
    ignore_index=True,
)
mean_scores_df = mean_scores_df.set_index("index")
mean_scores_df.index = mean_scores_df.index.set_names(["metrics"])
mean_scores_df
deep_dfs["answer_relevancy"]["scores"].value_counts()
deep_dfs["context_relevancy"]["scores"].value_counts()
displayify_df(deep_dfs["context_relevancy"].head(2))
cond = deep_dfs["context_relevancy"]["scores"] < 1
displayify_df(deep_dfs["context_relevancy"][cond].head(5))
