import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/mt_bench_single_grading.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Benchmarking LLM Evaluators On A Mini MT-Bench (Single Grading) `LabelledEvaluatorDataset`

In this notebook, we'll conduct an evaluation of three different evaluators that will be judging another LLM's response for response against a user query. More specifically, we will run benchmarks using a mini version of the MT-Bench single-grading dataset. In this version, we only consider the answers on the 160 questions (i.e., 80 x 2, since there are 80 two-turn dialogues) provided by llama2-70b. The reference answers used for this benchmark are provided by GPT-4. And so, our benchmarks on these three evaluators will assess closeness to GPT-4 (actually, self-consistency for the case of GPT-4).

1. GPT-3.5 (MLX)
2. GPT-4 (MLX)
3. Gemini-Pro (Google)
"""
logger.info("# Benchmarking LLM Evaluators On A Mini MT-Bench (Single Grading) `LabelledEvaluatorDataset`")

# %pip install llama-index-llms-ollama
# %pip install llama-index-llms-cohere
# %pip install llama-index-llms-gemini

# import nest_asyncio

# nest_asyncio.apply()

# !pip install "google-generativeai" -q

"""
### Load in Evaluator Dataset

Let's load in the llama-dataset from llama-hub.
"""
logger.info("### Load in Evaluator Dataset")


evaluator_dataset, _ = download_llama_dataset(
    "MiniMtBenchSingleGradingDataset", "./mini_mt_bench_data"
)

evaluator_dataset.to_pandas()[:5]

"""
### Define Our Evaluators
"""
logger.info("### Define Our Evaluators")


llm_gpt4 = MLXLlamaIndexLLMAdapter(temperature=0, model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
llm_gpt35 = MLXLlamaIndexLLMAdapter(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
llm_gemini = Gemini(model="models/gemini-pro", temperature=0)


evaluators = {
    "gpt-4": CorrectnessEvaluator(llm=llm_gpt4),
    "gpt-3.5": CorrectnessEvaluator(llm=llm_gpt35),
    "gemini-pro": CorrectnessEvaluator(llm=llm_gemini),
}

"""
### Benchmark With `EvaluatorBenchmarkerPack` (llama-pack)

When using the `EvaluatorBenchmarkerPack` with a `LabelledEvaluatorDataset`, the returned benchmarks will contain values for the following quantites:

- `number_examples`: The number of examples the dataset consists of.
- `invalid_predictions`: The number of evaluations that could not yield a final evaluation (e.g., due to inability to parse the evaluation output, or an exception thrown by the LLM evaluator)
- `correlation`: The correlation between the scores of the provided evaluator and those of the reference evaluator (in this case gpt-4).
- `mae`: The mean absolute error between the scores of the provided evaluator and those of the reference evaluator.
- `hamming`: The hamming distance between the scores of the provided evaluator and those of the reference evaluator.

NOTE: `correlation`, `mae`, and `hamming` are all computed without invalid predictions. So, essentially these metrics are conditional ones, conditioned on the prediction being valid.
"""
logger.info("### Benchmark With `EvaluatorBenchmarkerPack` (llama-pack)")


EvaluatorBenchmarkerPack = download_llama_pack(
    "EvaluatorBenchmarkerPack", "./pack"
)

"""
#### GPT 3.5
"""
logger.info("#### GPT 3.5")

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-3.5"],
    eval_dataset=evaluator_dataset,
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
    eval_dataset=evaluator_dataset,
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
#### Gemini Pro
"""
logger.info("#### Gemini Pro")

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gemini-pro"],
    eval_dataset=evaluator_dataset,
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

evaluator_benchmarker.prediction_dataset.save_json(
    "mt_sg_gemini_predictions.json"
)

"""
### In Summary

Putting all baselines together.
"""
logger.info("### In Summary")


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
- GPT-3.5 and Gemini-Pro seem to have similar results, with perhaps the slightes edge to GPT-3.5 in terms of closeness to GPT-4.
- Though, both don't seem to be too close to GPT-4.
- GPT-4 seems to be pretty consistent with itself in this benchmark.
"""
logger.info("From the results above, we make the following observations:")

logger.info("\n\n[DONE]", bright=True)