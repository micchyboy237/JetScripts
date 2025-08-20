import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.evaluation import PairwiseComparisonEvaluator
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.llms.cohere import Cohere
from llama_index.llms.gemini import Gemini
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/mt_bench_human_judgement.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Benchmarking LLM Evaluators On The MT-Bench Human Judgement `LabelledPairwiseEvaluatorDataset`

In this notebook guide, we benchmark Gemini and GPT models as LLM evaluators using a slightly adapted version of the MT-Bench Human Judgements dataset. For this dataset, human evaluators compare two llm model responses to a given query and rank them according to their own preference. In the original version, there can be more than one human evaluator for a given example (query, two model responses). In the adapted version that we consider however, we aggregate these 'repeated' entries and convert the 'winner' column of the original schema to instead represent the proportion of times 'model_a' wins across all of the human evaluators. To adapt this to a llama-dataset, and to better consider ties (albeit with small samples) we set an uncertainty threshold for this proportion in that if it is between [0.4, 0.6] then we consider there to be no winner between the two models. We download this dataset from [llama-hub](https://llamahub.ai). Finally, the LLMs that we benchmark are listed below:

1. GPT-3.5 (MLX)
2. GPT-4 (MLX)
3. Gemini-Pro (Google)
"""
logger.info("# Benchmarking LLM Evaluators On The MT-Bench Human Judgement `LabelledPairwiseEvaluatorDataset`")

# %pip install llama-index-llms-ollama
# %pip install llama-index-llms-cohere
# %pip install llama-index-llms-gemini

# !pip install "google-generativeai" -q

# import nest_asyncio

# nest_asyncio.apply()

"""
### Load In Dataset

Let's load in the llama-dataset from llama-hub.
"""
logger.info("### Load In Dataset")


pairwise_evaluator_dataset, _ = download_llama_dataset(
    "MtBenchHumanJudgementDataset", "./mt_bench_data"
)

pairwise_evaluator_dataset.to_pandas()[:5]

"""
### Define Our Evaluators
"""
logger.info("### Define Our Evaluators")



llm_gpt4 = MLX(temperature=0, model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
llm_gpt35 = MLX(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
llm_gemini = Gemini(model="models/gemini-pro", temperature=0)

evaluators = {
    "gpt-4": PairwiseComparisonEvaluator(llm=llm_gpt4),
    "gpt-3.5": PairwiseComparisonEvaluator(llm=llm_gpt35),
    "gemini-pro": PairwiseComparisonEvaluator(llm=llm_gemini),
}

"""
### Benchmark With `EvaluatorBenchmarkerPack` (llama-pack)

To compare our four evaluators we will benchmark them against `MTBenchHumanJudgementDataset`, wherein references are provided by human evaluators. The benchmarks will return the following quantites:

- `number_examples`: The number of examples the dataset consists of.
- `invalid_predictions`: The number of evaluations that could not yield a final evaluation (e.g., due to inability to parse the evaluation output, or an exception thrown by the LLM evaluator)
- `inconclusives`: Since this is a pairwise comparison, to mitigate the risk for "position bias" we conduct two evaluations — one with original order of presenting the two model answers, and another with the order in which these answers are presented to the evaluator LLM is flipped. A result is inconclusive if the LLM evaluator in the second ordering flips its vote in relation to the first vote.
- `ties`: A `PairwiseComparisonEvaluator` can also return a "tie" result. This is the number of examples for which it gave a tie result.
- `agreement_rate_with_ties`: The rate at which the LLM evaluator agrees with the reference (in this case human) evaluator, when also including ties. The denominator used to compute this metric is given by: `number_examples - invalid_predictions - inconclusives`.
- `agreement_rate_without_ties`: The rate at which the LLM evaluator agress with the reference (in this case human) evaluator, when excluding and ties. The denominator used to compute this metric is given by: `number_examples - invalid_predictions - inconclusives - ties`.

To compute these metrics, we'll make use of the `EvaluatorBenchmarkerPack`.
"""
logger.info("### Benchmark With `EvaluatorBenchmarkerPack` (llama-pack)")


EvaluatorBenchmarkerPack = download_llama_pack(
    "EvaluatorBenchmarkerPack", "./pack"
)

"""
#### GPT-3.5
"""
logger.info("#### GPT-3.5")

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-3.5"],
    eval_dataset=pairwise_evaluator_dataset,
    show_progress=True,
)

async def async_func_6():
    gpt_3p5_benchmark_df = await evaluator_benchmarker.arun(
        batch_size=100, sleep_time_in_seconds=0
    )
    return gpt_3p5_benchmark_df
gpt_3p5_benchmark_df = asyncio.run(async_func_6())
logger.success(format_json(gpt_3p5_benchmark_df))

gpt_3p5_benchmark_df.index = ["gpt-3.5"]
gpt_3p5_benchmark_df

"""
#### GPT-4
"""
logger.info("#### GPT-4")

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-4"],
    eval_dataset=pairwise_evaluator_dataset,
    show_progress=True,
)

async def async_func_6():
    gpt_4_benchmark_df = await evaluator_benchmarker.arun(
        batch_size=100, sleep_time_in_seconds=0
    )
    return gpt_4_benchmark_df
gpt_4_benchmark_df = asyncio.run(async_func_6())
logger.success(format_json(gpt_4_benchmark_df))

gpt_4_benchmark_df.index = ["gpt-4"]
gpt_4_benchmark_df

"""
### Gemini Pro

NOTE: The rate limit for Gemini models is still very constraining, which is understandable given that they've just been released at the time of writing this notebook. So, we use a very small `batch_size` and moderately high `sleep_time_in_seconds` to reduce risk of getting rate-limited.
"""
logger.info("### Gemini Pro")

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gemini-pro"],
    eval_dataset=pairwise_evaluator_dataset,
    show_progress=True,
)

async def async_func_6():
    gemini_pro_benchmark_df = await evaluator_benchmarker.arun(
        batch_size=5, sleep_time_in_seconds=0.5
    )
    return gemini_pro_benchmark_df
gemini_pro_benchmark_df = asyncio.run(async_func_6())
logger.success(format_json(gemini_pro_benchmark_df))

gemini_pro_benchmark_df.index = ["gemini-pro"]
gemini_pro_benchmark_df

evaluator_benchmarker.prediction_dataset.save_json("gemini_predictions.json")

"""
### Summary

For convenience, let's put all the results in a single DataFrame.
"""
logger.info("### Summary")


final_benchmark = pd.concat(
    [
        gpt_3p5_benchmark_df,
        gpt_4_benchmark_df,
        gemini_pro_benchmark_df,
    ],
    axis=0,
)
final_benchmark

"""
From the results above, we make the following observations:
- In terms of agreement rates, all three models seem quite close, with perhaps a slight edge given to the Gemini models
- Gemini Pro and GPT-3.5 seem to be a bit more assertive than GPT-4 resulting in only 50-60 ties to GPT-4's 100 ties.
- However, perhaps related to the previous point, GPT-4 yields the least amount of inconclusives, meaning that it suffers the least from position bias.
- Overall, it seems that Gemini Pro is up to snuff with GPT models, and would say that it outperforms GPT-3.5 — looks like Gemini can be legit alternatives to GPT models for evaluation tasks.
"""
logger.info("From the results above, we make the following observations:")

logger.info("\n\n[DONE]", bright=True)