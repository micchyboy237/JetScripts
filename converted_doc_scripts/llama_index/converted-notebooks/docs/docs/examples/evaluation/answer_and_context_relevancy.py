async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core import VectorStoreIndex
    from llama_index.core.evaluation import (
        AnswerRelevancyEvaluator,
        ContextRelevancyEvaluator,
    )
    from llama_index.core.evaluation.notebook_utils import get_eval_results_df
    from llama_index.core.llama_dataset import download_llama_dataset
    from llama_index.core.llama_pack import download_llama_pack
    from tqdm.asyncio import tqdm_asyncio
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
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/answer_and_context_relevancy.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Answer Relevancy and Context Relevancy Evaluations
    
    In this notebook, we demonstrate how to utilize the `AnswerRelevancyEvaluator` and `ContextRelevancyEvaluator` classes to get a measure on the relevancy of a generated answer and retrieved contexts, respectively, to a given user query. Both of these evaluators return a `score` that is between 0 and 1 as well as a generated `feedback` explaining the score. Note that, higher score means higher relevancy. In particular, we prompt the judge LLM to take a step-by-step approach in providing a relevancy score, asking it to answer the following two questions of a generated answer to a query for answer relevancy (for context relevancy these are slightly adjusted):
    
    1. Does the provided response match the subject matter of the user's query?
    2. Does the provided response attempt to address the focus or perspective on the subject matter taken on by the user's query?
    
    Each question is worth 1 point and so a perfect evaluation would yield a score of 2/2.
    """
    logger.info("# Answer Relevancy and Context Relevancy Evaluations")

    # %pip install llama-index-llms-ollama

    # import nest_asyncio

    # nest_asyncio.apply()

    def displayify_df(df):
        """For pretty displaying DataFrame in a notebook."""
        display_df = df.style.set_properties(
            **{
                "inline-size": "300px",
                "overflow-wrap": "break-word",
            }
        )
        display(display_df)

    """
    ### Download the dataset (`LabelledRagDataset`)
    
    For this demonstration, we will use a llama-dataset provided through our [llama-hub](https://llamahub.ai).
    """
    logger.info("### Download the dataset (`LabelledRagDataset`)")

    rag_dataset, documents = download_llama_dataset(
        "EvaluatingLlmSurveyPaperDataset", "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp"
    )

    rag_dataset.to_pandas()[:5]

    """
    Next, we build a RAG over the same source documents used to created the `rag_dataset`.
    """
    logger.info(
        "Next, we build a RAG over the same source documents used to created the `rag_dataset`.")

    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()

    """
    With our RAG (i.e `query_engine`) defined, we can make predictions (i.e., generate responses to the query) with it over the `rag_dataset`.
    """
    logger.info("With our RAG (i.e `query_engine`) defined, we can make predictions (i.e., generate responses to the query) with it over the `rag_dataset`.")

    prediction_dataset = await rag_dataset.amake_predictions_with(
        predictor=query_engine, batch_size=100, show_progress=True
    )
    logger.success(format_json(prediction_dataset))

    """
    ### Evaluating Answer and Context Relevancy Separately
    
    We first need to define our evaluators (i.e. `AnswerRelevancyEvaluator` & `ContextRelevancyEvaluator`):
    """
    logger.info("### Evaluating Answer and Context Relevancy Separately")

    judges = {}

    judges["answer_relevancy"] = AnswerRelevancyEvaluator(
        llm=OllamaFunctionCalling(
            temperature=0, model="llama3.2"),
    )

    judges["context_relevancy"] = ContextRelevancyEvaluator(
        llm=OllamaFunctionCalling(
            temperature=0, model="llama3.2"),
    )

    """
    Now, we can use our evaluator to make evaluations by looping through all of the <example, prediction> pairs.
    """
    logger.info(
        "Now, we can use our evaluator to make evaluations by looping through all of the <example, prediction> pairs.")

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
    logger.success(format_json(eval_results1))

    eval_results2 = await tqdm_asyncio.gather(*eval_tasks[250:])
    logger.success(format_json(eval_results2))

    eval_results = eval_results1 + eval_results2

    evals = {
        "answer_relevancy": eval_results[::2],
        "context_relevancy": eval_results[1::2],
    }

    """
    ### Taking a look at the evaluation results
    
    Here we use a utility function to convert the list of `EvaluationResult` objects into something more notebook friendly. This utility will provide two DataFrames, one deep one containing all of the evaluation results, and another one which aggregates via taking the mean of all the scores, per evaluation method.
    """
    logger.info("### Taking a look at the evaluation results")

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

    """
    The above utility also provides the mean score across all of the evaluations in `mean_df`.
    
    We can get a look at the raw distribution of the scores by invoking `value_counts()` on the `deep_df`.
    """
    logger.info(
        "The above utility also provides the mean score across all of the evaluations in `mean_df`.")

    deep_dfs["answer_relevancy"]["scores"].value_counts()

    deep_dfs["context_relevancy"]["scores"].value_counts()

    """
    It looks like for the most part, the default RAG does fairly well in terms of generating answers that are relevant to the query. Getting a closer look is made possible by viewing the records of any of the `deep_df`'s.
    """
    logger.info("It looks like for the most part, the default RAG does fairly well in terms of generating answers that are relevant to the query. Getting a closer look is made possible by viewing the records of any of the `deep_df`'s.")

    displayify_df(deep_dfs["context_relevancy"].head(2))

    """
    And, of course you can apply any filters as you like. For example, if you want to look at the examples that yielded less than perfect results.
    """
    logger.info("And, of course you can apply any filters as you like. For example, if you want to look at the examples that yielded less than perfect results.")

    cond = deep_dfs["context_relevancy"]["scores"] < 1
    displayify_df(deep_dfs["context_relevancy"][cond].head(5))

    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())
