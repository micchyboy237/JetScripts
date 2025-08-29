from jet.models.config import MODELS_CACHE_DIR
from jet.transformers.formatters import format_json
from IPython.display import Markdown, display
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.evaluation import (
CorrectnessEvaluator,
FaithfulnessEvaluator,
RelevancyEvaluator,
)
from llama_index.core.evaluation import PairwiseComparisonEvaluator
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from typing import Tuple
from typing import Tuple, Optional
import os
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/prometheus2_cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Prometheus-2 Cookbook

In this notebook we will demonstrate usage of [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models](https://arxiv.org/abs/2405.01535).

#### Abstract from the paper:

Proprietary LMs such as GPT-4 are often employed to assess the quality of responses from various LMs. However, concerns including transparency, controllability, and affordability strongly motivate the development of open-source LMs specialized in evaluations. On the other hand, existing open evaluator LMs exhibit critical shortcomings: 1) they issue scores that significantly diverge from those assigned by humans, and 2) they lack the flexibility to perform both direct assessment and pairwise ranking, the two most prevalent forms of assessment. Additionally, they do not possess the ability to evaluate based on custom evaluation criteria, focusing instead on general attributes like helpfulness and harmlessness. To address these issues, we introduce Prometheus 2, a more powerful evaluator LM than its predecessor that closely mirrors human and GPT-4 judgements. Moreover, it is capable of processing both direct assessment and pair-wise ranking formats grouped with a user-defined evaluation criteria. On four direct assessment benchmarks and four pairwise ranking benchmarks, Prometheus 2 scores the highest correlation and agreement with humans and proprietary LM judges among all tested open evaluator LMs.

#### Note: The base models for building Prometheus-2 are Mistral-7B and Mixtral8x7B.

Here we will demonstrate the usage of Prometheus-2 as evaluator for the following evaluators available with LlamaIndex:

1. Pairwise Evaluator - Assesses whether the LLM would favor one response over another from two different query engines.
2. Faithfulness Evaluator - Determines if the answer remains faithful to the retrieved contexts, indicating the absence of hallucination.
3. Correctness Evaluator - Determines whether the generated answer matches the reference answer provided for the query, which requires labels.
4. Relevancy Evaluator - Evaluates the relevance of retrieved contexts and the response to a query.

*   If you're unfamiliar with the above evaluators, please refer to our [Evaluation Guide](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/) for more information.

*   The prompts for the demonstration are inspired/ taken from the [promethues-eval](https://github.com/prometheus-eval/prometheus-eval/blob/main/libs/prometheus-eval/prometheus_eval/prompts.py) repository.

## Installation
"""
logger.info("# Prometheus-2 Cookbook")

# !pip install llama-index
# !pip install llama-index-llms-huggingface-api

"""
### Setup API Keys
"""
logger.info("### Setup API Keys")


# os.environ["OPENAI_API_KEY"] = "sk-"  # OPENAI API KEY

# import nest_asyncio

# nest_asyncio.apply()


"""
### Download Data

For the demonstration, we will utilize the PaulGrahamEssay dataset and define a sample query along with a reference answer.
"""
logger.info("### Download Data")


paul_graham_rag_dataset, paul_graham_documents = download_llama_dataset(
    "PaulGrahamEssayDataset", "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
)

"""
Get Query and Reference(Ground truth) answer for the demonstration.
"""
logger.info("Get Query and Reference(Ground truth) answer for the demonstration.")

query = paul_graham_rag_dataset[0].query
reference = paul_graham_rag_dataset[0].reference_answer

"""
### Setup LLM and Embedding model.

You need to deploy the model on huggingface or can load it locally. Here we deployed it using HF Inference Endpoints.

We will use OllamaFunctionCallingAdapter Embedding model and LLM for building Index, prometheus LLM for evaluation.
"""
logger.info("### Setup LLM and Embedding model.")


HF_TOKEN = "YOUR HF TOKEN"
HF_ENDPOINT_URL = "YOUR HF ENDPOINT URL"

prometheus_llm = HuggingFaceInferenceAPI(
    model_name=HF_ENDPOINT_URL,
    token=HF_TOKEN,
    temperature=0.0,
    do_sample=True,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    num_output=1024,
)



Settings.llm = OllamaFunctionCallingAdapter()
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
Settings.chunk_size = 512

"""
### Pairwise Evaluation

#### Build two QueryEngines for pairwise evaluation.
"""
logger.info("### Pairwise Evaluation")



dataset_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
rag_dataset = LabelledRagDataset.from_json(f"{dataset_path}/rag_dataset.json")
documents = SimpleDirectoryReader(
    input_dir=f"{dataset_path}/source_files"
).load_data()

index = VectorStoreIndex.from_documents(documents=documents)

query_engine1 = index.as_query_engine(similarity_top_k=1)

query_engine2 = index.as_query_engine(similarity_top_k=2)

response1 = str(query_engine1.query(query))
response2 = str(query_engine2.query(query))

response1

response2

ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
REL_SYSTEM_PROMPT = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

prometheus_pairwise_eval_prompt_template = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

Your task is to compare response A and Response B and give Feedback and score [RESULT] based on Rubric for the following query.
{query}

{answer_1}

{answer_2}

A: If Response A is better than Response B.
B: If Response B is better than Response A.

def parser_function(
    outputs: str,
) -> Tuple[Optional[bool], Optional[float], Optional[str]]:
    parts = outputs.split("[RESULT]")
    if len(parts) == 2:
        feedback, result = parts[0].strip(), parts[1].strip()
        if result == "A":
            return True, 0.0, feedback
        elif result == "B":
            return True, 1.0, feedback
    return None, None, None


prometheus_pairwise_evaluator = PairwiseComparisonEvaluator(
    llm=prometheus_llm,
    parser_function=parser_function,
    enforce_consensus=False,
    eval_template=REL_SYSTEM_PROMPT
    + "\n\n"
    + prometheus_pairwise_eval_prompt_template,
)

pairwise_result = prometheus_pairwise_evaluator.evaluate(
        query,
        response=response1,
        second_response=response2,
    )
logger.success(format_json(pairwise_result))

pairwise_result

pairwise_result.score

display(Markdown(f"<b>{pairwise_result.feedback}</b>"))

"""
#### Observation:

According to the feedback, the second response is preferred over the first response, with a score of 1.0 as per our parser function.

### Correctness Evaluation
"""
logger.info("#### Observation:")

prometheus_correctness_eval_prompt_template = """###Task Description:
An instruction (might include an Input inside it), a query, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is either 1 or 2 or 3 or 4 or 5. You should refer to the score rubric.
3. The output format should only look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.
5. Only evaluate on common things between generated answer and reference answer. Don't evaluate on things which are present in reference answer but not in generated answer.

Your task is to evaluate the generated answer and reference answer for the following query:
{query}

{generated_answer}

{reference_answer}

Score 1: If the generated answer is not relevant to the user query and reference answer.
Score 2: If the generated answer is according to reference answer but not relevant to user query.
Score 3: If the generated answer is relevant to the user query and reference answer but contains mistakes.
Score 4: If the generated answer is relevant to the user query and has the exact same metrics as the reference answer, but it is not as concise.
Score 5: If the generated answer is relevant to the user query and fully correct according to the reference answer.



def parser_function(output_str: str) -> Tuple[float, str]:
    display(Markdown(f"<b>{output_str}</b>"))

    pattern = r"(.+?) \[RESULT\] (\d)"

    matches = re.findall(pattern, output_str)

    if matches:
        feedback, score = matches[0]
        score = float(score.strip()) if score is not None else score
        return score, feedback.strip()
    else:
        return None, None



prometheus_correctness_evaluator = CorrectnessEvaluator(
    llm=prometheus_llm,
    parser_function=parser_function,
    eval_template=ABS_SYSTEM_PROMPT
    + "\n\n"
    + prometheus_correctness_eval_prompt_template,
)

correctness_result = prometheus_correctness_evaluator.evaluate(
    query=query,
    response=response1,
    reference=reference,
)

display(Markdown(f"<b>{correctness_result.score}</b>"))

display(Markdown(f"<b>{correctness_result.passing}</b>"))

display(Markdown(f"<b>{correctness_result.feedback}</b>"))

"""
#### Observation:

Based on the feedback, the generated answer is relevant to the user query and matches the metrics of the reference answer precisely. However, it is not as concise, resulting in a score of 4.0. Despite this, the answer passes as True based on the threshold.

### Faithfulness Evaluator
"""
logger.info("#### Observation:")

prometheus_faithfulness_eval_prompt_template = """###Task Description:
An instruction (might include an Input inside it), an information, a context, and a score rubric representing evaluation criteria are given.
1. You are provided with evaluation task with the help of information, context information to give result based on score rubrics.
2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)”
5. Please do not generate any other opening, closing, and explanations.


{query_str}

{context_str}

Score YES: If the given piece of information is supported by context.
Score NO: If the given piece of information is not supported by context


prometheus_faithfulness_refine_prompt_template = """###Task Description:
An instruction (might include an Input inside it), a information, a context information, an existing answer, and a score rubric representing a evaluation criteria are given.
1. You are provided with evaluation task with the help of information, context information and an existing answer.
2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)"
5. Please do not generate any other opening, closing, and explanations.


{existing_answer}

{query_str}

{context_msg}

Score YES: If the existing answer is already YES or If the Information is present in the context.
Score NO: If the existing answer is NO and If the Information is not present in the context.

prometheus_faithfulness_evaluator = FaithfulnessEvaluator(
    llm=prometheus_llm,
    eval_template=ABS_SYSTEM_PROMPT
    + "\n\n"
    + prometheus_faithfulness_eval_prompt_template,
    refine_template=ABS_SYSTEM_PROMPT
    + "\n\n"
    + prometheus_faithfulness_refine_prompt_template,
)

response_vector = query_engine1.query(query)

faithfulness_result = prometheus_faithfulness_evaluator.evaluate_response(
    response=response_vector
)

faithfulness_result.score

faithfulness_result.passing

"""
#### Observation:

The score and passing denotes there is no hallucination observed.

### Relevancy Evaluator
"""
logger.info("#### Observation:")

prometheus_relevancy_eval_prompt_template = """###Task Description:
An instruction (might include an Input inside it), a query with response, context, and a score rubric representing evaluation criteria are given.
1. You are provided with evaluation task with the help of a query with response and context.
2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
3. After writing a feedback, write a score that is A or B. You should refer to the score rubric.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)”
5. Please do not generate any other opening, closing, and explanations.


{query_str}

{context_str}

Score YES: If the response for the query is in line with the context information provided.
Score NO: If the response for the query is not in line with the context information provided.


prometheus_relevancy_refine_prompt_template = """###Task Description:
An instruction (might include an Input inside it), a query with response, context, an existing answer, and a score rubric representing a evaluation criteria are given.
1. You are provided with evaluation task with the help of a query with response and context and an existing answer.
2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)"
5. Please do not generate any other opening, closing, and explanations.


{query_str}

{context_str}

Score YES: If the existing answer is already YES or If the response for the query is in line with the context information provided.
Score NO: If the existing answer is NO and If the response for the query is in line with the context information provided.

prometheus_relevancy_evaluator = RelevancyEvaluator(
    llm=prometheus_llm,
    eval_template=ABS_SYSTEM_PROMPT
    + "\n\n"
    + prometheus_relevancy_eval_prompt_template,
    refine_template=ABS_SYSTEM_PROMPT
    + "\n\n"
    + prometheus_relevancy_refine_prompt_template,
)

relevancy_result = prometheus_relevancy_evaluator.evaluate_response(
    query=query, response=response_vector
)

relevancy_result.score

relevancy_result.passing

display(Markdown(f"<b>{relevancy_result.feedback}</b>"))

"""
#### Observation:

The feedback indicates that the response to the query aligns well with the provided context information, resulting in a score of 1.0 and passing status of True.

### Conclusion:

Exploring Prometheus-2 for OSS evaluation is interesting.

The feedback is in the expected format, making parsing and decision-making easier.

It's valuable to compare with GPT-4 for evaluation purposes and consider using Prometheus-2 in evaluations.

You can refer to our [guide](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/prometheus_evaluation.ipynb) on comparing GPT-4 as an evaluator with the OSS evaluation model for experimentation.
"""
logger.info("#### Observation:")

logger.info("\n\n[DONE]", bright=True)