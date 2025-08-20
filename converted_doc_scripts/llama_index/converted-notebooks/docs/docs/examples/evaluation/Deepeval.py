import asyncio
from jet.transformers.formatters import format_json
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.integrations.llama_index import (
instrument_llama_index,
FunctionAgent,
)
from deepeval.integrations.llama_index import FunctionAgent
from deepeval.integrations.llama_index import instrument_llama_index
from deepeval.metrics import AnswerRelevancyMetric
from dotenv import load_dotenv
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio
import deepeval
import llama_index.core.instrumentation as instrument
import os
import shutil
import time


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/Deepeval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 🚀 DeepEval - Open Source Evals with Tracing

This code tutorial shows how you can easily trace and evaluate your LlamaIndex Agents. You can read more about the DeepEval framework here: https://docs.confident-ai.com/docs/getting-started

LlamaIndex integration with DeepEval allows you to trace your LlamaIndex Agents and evaluate them using DeepEval's default metrics. Read more about the integration here: https://deepeval.com/integrations/frameworks/langchain

Feel free to check out our repository here on GitHub: https://github.com/confident-ai/deepeval

### Quickstart

Install the following packages:
"""
logger.info("# 🚀 DeepEval - Open Source Evals with Tracing")

# !pip install -q -q llama-index
# !pip install -U -q deepeval

"""
This step is optional and only if you want a server-hosted dashboard! (Psst I think you should!)
"""
logger.info("This step is optional and only if you want a server-hosted dashboard! (Psst I think you should!)")

# !deepeval login

"""
### End-to-End Evals

`deepeval` allows you to evaluate LlamaIndex applications end-to-end in under a minute.

Create a `FunctionAgent` with a list of metrics you wish to use, and pass it to your LlamaIndex application's `run` method.
"""
logger.info("### End-to-End Evals")




instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


answer_relevancy_metric = AnswerRelevancyMetric()

agent = FunctionAgent(
    tools=[multiply],
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metrics=[answer_relevancy_metric],
)


async def llm_app(input: str):
    async def run_async_code_490ef1cf():
        return await agent.run(input)
        return 
     = asyncio.run(run_async_code_490ef1cf())
    logger.success(format_json())


asyncio.run(llm_app("What is 2 * 3?"))

"""
Evaluations are supported for LlamaIndex `FunctionAgent`, `ReActAgent` and `CodeActAgent`. Only metrics with LLM parameters input and output are eligible for evaluation.

#### Synchronous

Create a `FunctionAgent` with a list of metrics you wish to use, and pass it to your LlamaIndex application's run method.
"""
logger.info("#### Synchronous")


dataset = EvaluationDataset(
    goldens=[Golden(input="What is 3 * 12?"), Golden(input="What is 4 * 13?")]
)

for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(golden.input))
    dataset.evaluate(task)

"""
### Asynchronous
"""
logger.info("### Asynchronous")


dataset = EvaluationDataset(
    goldens=[Golden(input="What's 7 * 8?"), Golden(input="What's 7 * 6?")]
)

for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(golden.input))
    dataset.evaluate(task)

"""
#### ⚠️ Warning: DeepEval runs using event loops for managing asynchronous operations.

Jupyter notebooks already maintain their own event loop, which may lead to unexpected behavior, hangs, or runtime errors when running DeepEval examples directly in a notebook cell.

Recommendation: To avoid such issues, run your DeepEval examples in a standalone Python script (.py file) instead of within Jupyter Notebook.

### Examples

Here are some examples scripts.
"""
logger.info("#### ⚠️ Warning: DeepEval runs using event loops for managing asynchronous operations.")





load_dotenv()

deepeval.login(os.getenv("CONFIDENT_API_KEY"))
instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


answer_relevancy_metric = AnswerRelevancyMetric()
agent = FunctionAgent(
    tools=[multiply],
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metrics=[answer_relevancy_metric],
)


async def llm_app(input: str):
    async def run_async_code_490ef1cf():
        return await agent.run(input)
        return 
     = asyncio.run(run_async_code_490ef1cf())
    logger.success(format_json())


dataset = EvaluationDataset(
    goldens=[Golden(input="What is 3 * 12?"), Golden(input="What is 4 * 13?")]
)
for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(golden.input))
    dataset.evaluate(task)



load_dotenv()


deepeval.login(os.getenv("CONFIDENT_API_KEY"))
instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


answer_relevancy_metric = AnswerRelevancyMetric()
agent = FunctionAgent(
    tools=[multiply],
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metrics=[answer_relevancy_metric],
)

goldens = [Golden(input="What's 7 * 8?"), Golden(input="What's 7 * 6?")]


async def llm_app(golden: Golden):
    async def run_async_code_59a63786():
        await agent.run(golden.input)
        return 
     = asyncio.run(run_async_code_59a63786())
    logger.success(format_json())


def main():
    dataset = EvaluationDataset(goldens=goldens)
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(llm_app(golden))
        dataset.evaluate(task)


if __name__ == "__main__":
    main()



load_dotenv()

deepeval.login(os.getenv("CONFIDENT_API_KEY"))
instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metric_collection="test_collection_1",
)


async def llm_app(golden: Golden):
    async def run_async_code_59a63786():
        await agent.run(golden.input)
        return 
     = asyncio.run(run_async_code_59a63786())
    logger.success(format_json())


asyncio.run(llm_app(Golden(input="What is 3 * 12?")))

logger.info("\n\n[DONE]", bright=True)