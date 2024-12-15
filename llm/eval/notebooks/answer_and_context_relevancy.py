%pip install llama-index-llms-openaiimport nest_asyncio
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

rag_dataset, documents = download_llama_dataset(
    "EvaluatingLlmSurveyPaperDataset", "./data"
)
rag_dataset.to_pandas()[:5]
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()
prediction_dataset = await rag_dataset.amake_predictions_with(
    predictor=query_engine, batch_size=100, show_progress=True
)
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)

judges = {}

judges["answer_relevancy"] = AnswerRelevancyEvaluator(
    llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
)

judges["context_relevancy"] = ContextRelevancyEvaluator(
    llm=OpenAI(temperature=0, model="gpt-4"),
)
eval_tasks = []
for example, prediction in zip(
    rag_dataset.examples, prediction_dataset.predictions
):
    eval_tasks.append(
        judges["answer_relevancy"].aevaluate(
            query=example.query,
            response=prediction.response,
            sleep_time_in_seconds=1.0,
        )
    )
    eval_tasks.append(
        judges["context_relevancy"].aevaluate(
            query=example.query,
            contexts=prediction.contexts,
            sleep_time_in_seconds=1.0,
        )
    )
eval_results1 = await tqdm_asyncio.gather(*eval_tasks[:250])
eval_results2 = await tqdm_asyncio.gather(*eval_tasks[250:])
eval_results = eval_results1 + eval_results2
evals = {
    "answer_relevancy": eval_results[::2],
    "context_relevancy": eval_results[1::2],
}
from llama_index.core.evaluation.notebook_utils import get_eval_results_df
import pandas as pd

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
