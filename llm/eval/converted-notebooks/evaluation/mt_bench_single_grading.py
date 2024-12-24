from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/mt_bench_single_grading.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Benchmarking LLM Evaluators On A Mini MT-Bench (Single Grading) `LabelledEvaluatorDataset`

# In this notebook, we'll conduct an evaluation of three different evaluators that will be judging another LLM's response for response against a user query. More specifically, we will run benchmarks using a mini version of the MT-Bench single-grading dataset. In this version, we only consider the answers on the 160 questions (i.e., 80 x 2, since there are 80 two-turn dialogues) provided by llama2-70b. The reference answers used for this benchmark are provided by GPT-4. And so, our benchmarks on these three evaluators will assess closeness to GPT-4 (actually, self-consistency for the case of GPT-4).
# 
# 1. GPT-3.5 (Ollama)
# 2. GPT-4 (Ollama)
# 3. Gemini-Pro (Google)

# %pip install llama-index-llms-ollama
# %pip install llama-index-llms-cohere
# %pip install llama-index-llms-gemini

import nest_asyncio

nest_asyncio.apply()

# !pip install "google-generativeai" -q

### Load in Evaluator Dataset

# Let's load in the llama-dataset from llama-hub.

from llama_index.core.llama_dataset import download_llama_dataset

evaluator_dataset, _ = download_llama_dataset(
    "MiniMtBenchSingleGradingDataset", "./mini_mt_bench_data"
)

evaluator_dataset.to_pandas()[:5]

### Define Our Evaluators

from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.llms.cohere import Cohere

llm_gpt4 = Ollama(temperature=0, model="llama3.1", request_timeout=300.0, context_window=4096)
llm_gpt35 = Ollama(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)
llm_gemini = Gemini(model="models/gemini-pro", temperature=0)


evaluators = {
    "gpt-4": CorrectnessEvaluator(llm=llm_gpt4),
    "gpt-3.5": CorrectnessEvaluator(llm=llm_gpt35),
    "gemini-pro": CorrectnessEvaluator(llm=llm_gemini),
}

### Benchmark With `EvaluatorBenchmarkerPack` (llama-pack)
# 
# When using the `EvaluatorBenchmarkerPack` with a `LabelledEvaluatorDataset`, the returned benchmarks will contain values for the following quantites:
# 
# - `number_examples`: The number of examples the dataset consists of.
# - `invalid_predictions`: The number of evaluations that could not yield a final evaluation (e.g., due to inability to parse the evaluation output, or an exception thrown by the LLM evaluator)
# - `correlation`: The correlation between the scores of the provided evaluator and those of the reference evaluator (in this case gpt-4).
# - `mae`: The mean absolute error between the scores of the provided evaluator and those of the reference evaluator.
# - `hamming`: The hamming distance between the scores of the provided evaluator and those of the reference evaluator.
# 
# NOTE: `correlation`, `mae`, and `hamming` are all computed without invalid predictions. So, essentially these metrics are conditional ones, conditioned on the prediction being valid.

from llama_index.core.llama_pack import download_llama_pack

EvaluatorBenchmarkerPack = download_llama_pack(
    "EvaluatorBenchmarkerPack", "./pack"
)

#### GPT 3.5

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-3.5"],
    eval_dataset=evaluator_dataset,
    show_progress=True,
)

gpt_3p5_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=100, sleep_time_in_seconds=0
)

gpt_3p5_benchmark_df.index = ["gpt-3.5"]
gpt_3p5_benchmark_df

#### GPT-4

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-4"],
    eval_dataset=evaluator_dataset,
    show_progress=True,
)

gpt_4_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=100, sleep_time_in_seconds=0
)

gpt_4_benchmark_df.index = ["gpt-4"]
gpt_4_benchmark_df

#### Gemini Pro

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gemini-pro"],
    eval_dataset=evaluator_dataset,
    show_progress=True,
)

gemini_pro_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=5, sleep_time_in_seconds=0.5
)

gemini_pro_benchmark_df.index = ["gemini-pro"]
gemini_pro_benchmark_df

evaluator_benchmarker.prediction_dataset.save_json(
    "mt_sg_gemini_predictions.json"
)

### In Summary
# 
# Putting all baselines together.

import pandas as pd

final_benchmark = pd.concat(
    [
        gpt_3p5_benchmark_df,
        gpt_4_benchmark_df,
        gemini_pro_benchmark_df,
    ],
    axis=0,
)
final_benchmark

# From the results above, we make the following observations:
# - GPT-3.5 and Gemini-Pro seem to have similar results, with perhaps the slightes edge to GPT-3.5 in terms of closeness to GPT-4.
# - Though, both don't seem to be too close to GPT-4.
# - GPT-4 seems to be pretty consistent with itself in this benchmark.

logger.info("\n\n[DONE]", bright=True)