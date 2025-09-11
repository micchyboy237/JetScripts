from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_vertexai import ChatVertexAI
from langfair.auto import AutoEval
from langfair.generator import ResponseGenerator
from langfair.generator.counterfactual import CounterfactualGenerator
from langfair.metrics.counterfactual import CounterfactualMetrics
from langfair.metrics.stereotype import StereotypeMetrics
from langfair.metrics.toxicity import ToxicityMetrics
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# LangFair: Use-Case Level LLM Bias and Fairness Assessments

LangFair is a comprehensive Python library designed for conducting bias and fairness assessments of large language model (LLM) use cases. The LangFair [repository](https://github.com/cvs-health/langfair) includes a comprehensive framework for [choosing bias and fairness metrics](https://github.com/cvs-health/langfair/tree/main#-choosing-bias-and-fairness-metrics-for-an-llm-use-case), along with [demo notebooks](https://github.com/cvs-health/langfair/tree/main/examples) and a [technical playbook](https://arxiv.org/abs/2407.10853) that discusses LLM bias and fairness risks, evaluation metrics, and best practices.

Explore our [documentation site](https://cvs-health.github.io/langfair/) for detailed instructions on using LangFair.

## ⚡ Quickstart Guide
### (Optional) Create a virtual environment for using LangFair
We recommend creating a new virtual environment using venv before installing LangFair. To do so, please follow instructions [here](https://docs.python.org/3/library/venv.html).

### Installing LangFair
The latest version can be installed from PyPI:
"""
logger.info("# LangFair: Use-Case Level LLM Bias and Fairness Assessments")

pip install langfair

"""
### Usage Examples
Below are code samples illustrating how to use LangFair to assess bias and fairness risks in text generation and summarization use cases. The below examples assume the user has already defined a list of prompts from their use case, `prompts`.

##### Generate LLM responses
To generate responses, we can use LangFair's `ResponseGenerator` class. First, we must create a `langchain` LLM object. Below we use `ChatVertexAI`, but **any of [LangChain’s LLM classes](https://js.langchain.com/docs/integrations/chat/) may be used instead**. Note that `InMemoryRateLimiter` is to used to avoid rate limit errors.
"""
logger.info("### Usage Examples")

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4.5, check_every_n_seconds=0.5, max_bucket_size=280,
)
llm = ChatVertexAI(
    model_name="gemini-pro", temperature=0.3, rate_limiter=rate_limiter
)

"""
We can use `ResponseGenerator.generate_responses` to generate 25 responses for each prompt, as is convention for toxicity evaluation.
"""
logger.info("We can use `ResponseGenerator.generate_responses` to generate 25 responses for each prompt, as is convention for toxicity evaluation.")

rg = ResponseGenerator(langchain_llm=llm)
generations = await rg.generate_responses(prompts=prompts, count=25)
logger.success(format_json(generations))
responses = generations["data"]["response"]
duplicated_prompts = generations["data"]["prompt"] # so prompts correspond to responses

"""
##### Compute toxicity metrics
Toxicity metrics can be computed with `ToxicityMetrics`. Note that use of `torch.device` is optional and should be used if GPU is available to speed up toxicity computation.
"""
logger.info("##### Compute toxicity metrics")

tm = ToxicityMetrics(
)
tox_result = tm.evaluate(
    prompts=duplicated_prompts,
    responses=responses,
    return_data=True
)
tox_result['metrics']

"""
##### Compute stereotype metrics
Stereotype metrics can be computed with `StereotypeMetrics`.
"""
logger.info("##### Compute stereotype metrics")

sm = StereotypeMetrics()
stereo_result = sm.evaluate(responses=responses, categories=["gender"])
stereo_result['metrics']

"""
##### Generate counterfactual responses and compute metrics
We can generate counterfactual responses with `CounterfactualGenerator`.
"""
logger.info("##### Generate counterfactual responses and compute metrics")

cg = CounterfactualGenerator(langchain_llm=llm)
cf_generations = await cg.generate_responses(
        prompts=prompts, attribute='gender', count=25
    )
logger.success(format_json(cf_generations))
male_responses = cf_generations['data']['male_response']
female_responses = cf_generations['data']['female_response']

"""
Counterfactual metrics can be easily computed with `CounterfactualMetrics`.
"""
logger.info("Counterfactual metrics can be easily computed with `CounterfactualMetrics`.")

cm = CounterfactualMetrics()
cf_result = cm.evaluate(
    texts1=male_responses,
    texts2=female_responses,
    attribute='gender'
)
cf_result['metrics']

"""
##### Alternative approach: Semi-automated evaluation with `AutoEval`
To streamline assessments for text generation and summarization use cases, the `AutoEval` class conducts a multi-step process that completes all of the aforementioned steps with two lines of code.
"""
logger.info("##### Alternative approach: Semi-automated evaluation with `AutoEval`")

auto_object = AutoEval(
    prompts=prompts,
    langchain_llm=llm,
)
results = await auto_object.evaluate()
logger.success(format_json(results))
results['metrics']

logger.info("\n\n[DONE]", bright=True)