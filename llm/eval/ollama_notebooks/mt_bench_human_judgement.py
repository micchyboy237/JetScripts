```python
# %pip install llama-index-llms-openai
# %pip install llama-index-llms-cohere
# %pip install llama-index-llms-gemini
# !pip install "google-generative-ai" -q
import nest_asyncio

nest_asyncio.apply()
from llama_index.core.llama_dataset import download_llama_dataset

pairwise_evaluator_dataset, _ = download_llama_dataset(
    "MtBenchHumanJudgementDataset", "./mt_bench_data"
)
pairwise_evaluator_dataset.to_pandas()[:5]
from llama_index.core.evaluation import PairwiseComparisonEvaluator
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

initialize_ollama_settings()

# Replace openai llm and embed models with ollama
llm_gpt4 = Ollama(temperature=0, model="llama3.1")
llm_gpt35 = Ollama(temperature=0, model="llama3.2")
llm_gemini = Ollama(model="models/gemini-pro", temperature=0)

evaluators = {
    "gpt-4": PairwiseComparisonEvaluator(llm=llm_gpt4),
    "gpt-3.5": PairwiseComparisonEvaluator(llm=llm_gpt35),
    "gemini-pro": PairwiseComparisonEvaluator(llm=llm_gemini),
}
from llama_index.core.llama_pack import download_llama_pack

EvaluatorBenchmarkerPack = download_llama_pack(
    "EvaluatorBenchmarkerPack", "./pack"
)
evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-3.5"],
    eval_dataset=pairwise_evaluator_dataset,
    show_progress=True,
)
gpt_3p5_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=100, sleep_time_in_seconds=0
)
gpt_3p5_benchmark_df.index = ["gpt-3.5"]
gpt_3p5_benchmark_df
evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-4"],
    eval_dataset=pairwise_evaluator_dataset,
    show_progress=True,
)
gpt_4_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=100, sleep_time_in_seconds=0
)
gpt_4_benchmark_df.index = ["gpt-4"]
gpt_4_benchmark_df
evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gemini-pro"],
    eval_dataset=pairwise_evaluator_dataset,
    show_progress=True,
)
gemini_pro_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=5, sleep_time_in_seconds=0.5
)
gemini_pro_benchmark_df.index = ["gemini-pro"]
gemini_pro_benchmark_df
evaluator_benchmarker.prediction_dataset.save_json("gemini_predictions.json")
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
```