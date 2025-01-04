from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/prometheus_evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Evaluation using [Prometheus](https://huggingface.co/TheBloke/prometheus-13B-v1.0-GPTQ) model

# Evaluation is a crucial aspect of iterating over your RAG (Retrieval-Augmented Generation) pipeline. This process has relied heavily on GPT-4. However, a new open-source model named [Prometheus](https://arxiv.org/abs/2310.08491) has recently emerged as an alternative for evaluation purposes.
# 
# In this notebook, we will demonstrate how you can utilize the Prometheus model for evaluation, integrating it with the LlamaIndex abstractions.

# If you're unfamiliar with the Prometheus model, you might find the paper summary prepared by Andrei informative. It's important to note that this model requires rubric scores to be included in the prompt for effective evaluation. For more detailed information, you can refer to the specific prompts outlined in the notebook.

# ![Prometheus Paper Card](../data/images/prometheus_paper_card.png)

# We will demonstrate the correctness evaluation using the Prometheus model with two datasets from the Llama Datasets. If you haven't yet explored Llama Datasets, I recommend taking some time to read about them [here](https://blog.llamaindex.ai/introducing-llama-datasets-aadb9994ad9e).
# 
# 1. Paul Graham Essay
# 2. Llama2

### Note: We are showcasing original [Prometheus model](https://huggingface.co/kaist-ai/prometheus-13b-v1.0) for the analysis here. You can re-run the analysis with [quantized version of the model](https://huggingface.co/TheBloke/prometheus-13B-v1.0-GPTQ).

# %pip install llama-index-llms-ollama
# %pip install llama-index-llms-huggingface-api

import nest_asyncio

nest_asyncio.apply()

## Download Datasets

from llama_index.core.llama_dataset import download_llama_dataset

paul_graham_rag_dataset, paul_graham_documents = download_llama_dataset(
    "PaulGrahamEssayDataset", "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/retrievers/data"
)

llama2_rag_dataset, llama2_documents = download_llama_dataset(
    "Llama2PaperDataset", "./data/llama2"
)

## Define Prometheus LLM hosted on HuggingFace.
# 
# We hosted the model on HF Inference endpoint using Nvidia A10G GPU.

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

HF_TOKEN = "YOUR HF TOKEN"
HF_ENDPOINT_URL = (
    "https://q3yljc2cypyrvw3i.us-east-1.aws.endpoints.huggingface.cloud"
)

prometheus_llm = HuggingFaceInferenceAPI(
    model_name=HF_ENDPOINT_URL,
    token=HF_TOKEN,
    temperature=0.1,
    do_sample=True,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
)

## Prompt templates.
# 
# We will use same prompts for Prometheus model and GPT-4 to make consistent performance comparision.

### Correctness Evaluation Prompt

prometheus_correctness_eval_prompt_template = """###Task Description: An instruction (might include an Input inside it), a query, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given. 
			1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general. 
			2. After writing a feedback, write a score that is either 1 or 2 or 3 or 4 or 5. You should refer to the score rubric. 
			3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (1 or 2 or 3 or 4 or 5)" 
			4. Please do not generate any other opening, closing, and explanations. 
            5. Only evaluate on common things between generated answer and reference answer. Don't evaluate on things which are present in reference answer but not in generated answer.

			

            
            Score 1: If the generated answer is not relevant to the user query and reference answer.
            Score 2: If the generated answer is according to reference answer but not relevant to user query.
            Score 3: If the generated answer is relevant to the user query and reference answer but contains mistakes.
    		Score 4: If the generated answer is relevant to the user query and has the exact same metrics as the reference answer, but it is not as concise.
            Score 5: If the generated answer is relevant to the user query and fully correct according to the reference answer.

prometheus_correctness_eval_prompt_template = """###Task Description: An instruction (might include an Input inside it), a query, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given. 
			1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general. 
			2. After writing a feedback, write a score that is either 1 or 2 or 3 or 4 or 5. You should refer to the score rubric. 
			3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (1 or 2 or 3 or 4 or 5)" 
			4. Please do not generate any other opening, closing, and explanations. 
            5. Only evaluate on common things between generated answer and reference answer. Don't evaluate on things which are present in reference answer but not in generated answer.

			

            
            Score 1: If the generated answer is not relevant to the user query and reference answer.
            Score 2: If the generated answer is correct according to reference answer but not relevant to user query.
            Score 3: If the generated answer is relevant to the user query and correct according to reference answer but has some mistakes in facts.
    		Score 4: If the generated answer is relevant to the user query and has the exact same metrics and correct as the reference answer, but it is not as concise.
            Score 5: If the generated answer is relevant to the user query and fully correct according to the reference answer.

### Faithfulness Evaluation Prompt

prometheus_faithfulness_eval_prompt_template = """###Task Description: An instruction (might include an Input inside it), an information, a context, and a score rubric representing evaluation criteria are given. 
	        1. You are provided with evaluation task with the help of information, context information to give result based on score rubrics.
            2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general. 
			3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric. 
            4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)” 
            5. Please do not generate any other opening, closing, and explanations. 



            
        Score YES: If the given piece of information is supported by context.
        Score NO: If the given piece of information is not supported by context
    

prometheus_faithfulness_refine_prompt_template = """###Task Description: An instruction (might include an Input inside it), a information, a context information, an existing answer, and a score rubric representing a evaluation criteria are given. 
			1. You are provided with evaluation task with the help of information, context information and an existing answer.
            2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
			3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric. 
			4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)" 
			5. Please do not generate any other opening, closing, and explanations. 




            
            Score YES: If the existing answer is already YES or If the Information is present in the context.
            Score NO: If the existing answer is NO and If the Information is not present in the context.

### Relevancy Evaluation Prompt

prometheus_relevancy_eval_prompt_template = """###Task Description: An instruction (might include an Input inside it), a query with response, context, and a score rubric representing evaluation criteria are given. 
            1. You are provided with evaluation task with the help of a query with response and context.
            2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general. 
			3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric. 
            4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)” 
            5. Please do not generate any other opening, closing, and explanations. 



            
        Score YES: If the response for the query is in line with the context information provided.
        Score NO: If the response for the query is not in line with the context information provided.
    

prometheus_relevancy_refine_prompt_template = """###Task Description: An instruction (might include an Input inside it), a query with response, context, an existing answer, and a score rubric representing a evaluation criteria are given. 
			1. You are provided with evaluation task with the help of a query with response and context and an existing answer.
            2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general. 
			3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric. 
			4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)" 
			5. Please do not generate any other opening, closing, and explanations. 



            
            Score YES: If the existing answer is already YES or If the response for the query is in line with the context information provided.
            Score NO: If the existing answer is NO and If the response for the query is in line with the context information provided.

# Set Ollama Key for indexing

import os

# os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"

from llama_index.llms.ollama import Ollama

gpt4_llm = Ollama(model="llama3.1")

## Define parser function 
# 
# It will be used in correctness evaluator.

from typing import Tuple
import re


def parser_function(output_str: str) -> Tuple[float, str]:
    pattern = r"(.+?) \[RESULT\] (\d)"

    matches = re.findall(pattern, output_str)

    if matches:
        feedback, score = matches[0]
        score = float(score.strip()) if score is not None else score
        return score, feedback.strip()
    else:
        return None, None

## Define Correctness, FaithFulness, Relevancy Evaluators

from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken


prometheus_correctness_evaluator = CorrectnessEvaluator(
    llm=prometheus_llm,
    parser_function=parser_function,
    eval_template=prometheus_correctness_eval_prompt_template,
)

prometheus_faithfulness_evaluator = FaithfulnessEvaluator(
    llm=prometheus_llm,
    eval_template=prometheus_faithfulness_eval_prompt_template,
    refine_template=prometheus_faithfulness_refine_prompt_template,
)

prometheus_relevancy_evaluator = RelevancyEvaluator(
    llm=prometheus_llm,
    eval_template=prometheus_relevancy_eval_prompt_template,
    refine_template=prometheus_relevancy_refine_prompt_template,
)

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model(model="llama3.1").encode
)

callback_manager = CallbackManager([token_counter])
gpt4_llm.callback_manager = callback_manager

gpt4_correctness_evaluator = CorrectnessEvaluator(
    llm=gpt4_llm,
)

gpt4_faithfulness_evaluator = FaithfulnessEvaluator(
    llm=gpt4_llm,
    eval_template=prometheus_faithfulness_eval_prompt_template,
    refine_template=prometheus_faithfulness_refine_prompt_template,
)

gpt4_relevancy_evaluator = RelevancyEvaluator(
    llm=gpt4_llm,
    eval_template=prometheus_relevancy_eval_prompt_template,
    refine_template=prometheus_relevancy_refine_prompt_template,
)

prometheus_evaluators = {
    "correctness": prometheus_correctness_evaluator,
    "faithfulness": prometheus_faithfulness_evaluator,
    "relevancy": prometheus_relevancy_evaluator,
}

gpt4_evaluators = {
    "correctness": gpt4_correctness_evaluator,
    "faithfulness": gpt4_faithfulness_evaluator,
    "relevancy": gpt4_relevancy_evaluator,
}

## Let's create a function to create `query_engine` and `rag_dataset` for different datasets.

from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


def create_query_engine_rag_dataset(dataset_path):
    rag_dataset = LabelledRagDataset.from_json(
        f"{dataset_path}/rag_dataset.json"
    )
    documents = SimpleDirectoryReader(
        input_dir=f"{dataset_path}/source_files"
    ).load_data()

    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()

    return query_engine, rag_dataset

## Function to run batch evaluations on defined evaluators

from llama_index.core.evaluation import BatchEvalRunner


async def batch_eval_runner(
    evaluators, query_engine, questions, reference=None, num_workers=8
):
    batch_runner = BatchEvalRunner(
        evaluators, workers=num_workers, show_progress=True
    )

    eval_results = batch_runner.evaluate_queries(
        query_engine, queries=questions, reference=reference
    )

    return eval_results

## Function to check the distribution of scores

from collections import Counter
from typing import List, Dict


def get_scores_distribution(scores: List[float]) -> Dict[str, float]:
    score_counts = Counter(scores)

    total_scores = len(scores)

    percentage_distribution = {
        score: (count / total_scores) * 100
        for score, count in score_counts.items()
    }

    return percentage_distribution

## Function to check correctness, faithfulness and relevancy evaluation score

def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {round(score, 2)}")
    return score

## Function to compute `Hamming Distance`.

def hamming_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(list1, list2))

## Evaluation on PaulGraham Essay text

query_engine, rag_dataset = create_query_engine_rag_dataset(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/retrievers/data"
)

questions = [example.query for example in rag_dataset.examples]

reference = [[example.reference_answer] for example in rag_dataset.examples]

### Compute Correctness, Faithfulness and Relevancy Evaluation

prometheus_eval_results = await batch_eval_runner(
    prometheus_evaluators, query_engine, questions, reference
)

gpt4_eval_results = await batch_eval_runner(
    gpt4_evaluators, query_engine, questions, reference
)

### Correctness Evaluation score distribution with Prometheus Evaluator.

prometheus_scores = [
    result.score for result in prometheus_eval_results["correctness"]
]
get_scores_distribution(prometheus_scores)

### Correctness Evaluation score distribution with GPT-4 Evaluator.

gpt4_scores = [result.score for result in gpt4_eval_results["correctness"]]
get_scores_distribution(gpt4_scores)

### Feedback comparision between prometheus and gpt-4.

query = prometheus_eval_results["correctness"][0].query
response = prometheus_eval_results["correctness"][0].response
reference_answer = reference[0][0]

prometheus_feedback = prometheus_eval_results["correctness"][0].feedback
prometheus_score = prometheus_eval_results["correctness"][0].score

gpt4_feedback = gpt4_eval_results["correctness"][0].feedback
gpt4_score = gpt4_eval_results["correctness"][0].score

print(f"Query: {query} \n\n")
print(f"Generated Answer: {response} \n\n")
print(f"Reference Answer: {reference_answer} \n\n")
print(
    f"Prometheus Feedback: {prometheus_feedback} \n\n {prometheus_score} \n\n"
)
print(f"GPT-4 Feedback: {gpt4_feedback} \n\n {gpt4_score}")

#### Observation:
# 
# The feedback from Prometheus is more detailed, noting that certain specifics were omitted in the generated response, resulting in a score of `3.0`. Conversely, GPT-4's feedback is broader and less specific, awarding a score of `5.0`, despite the absence of some details.

### Prometheus Faithfulness and Relevancy Evaluation scores.

_ = get_eval_results("faithfulness", prometheus_eval_results)

_ = get_eval_results("relevancy", prometheus_eval_results)

### GPT-4 Faithfulness and Relevancy Evaluation scores.

_ = get_eval_results("faithfulness", gpt4_eval_results)

_ = get_eval_results("relevancy", gpt4_eval_results)

### Hamming Distance comparison between Prometheus and GPT-4
# 
# (Lower the better)

prometheus_faithfulness_scores = [
    result.score for result in prometheus_eval_results["faithfulness"]
]
prometheus_relevancy_scores = [
    result.score for result in prometheus_eval_results["relevancy"]
]

gpt4_faithfulness_scores = [
    result.score for result in gpt4_eval_results["faithfulness"]
]
gpt4_relevancy_scores = [
    result.score for result in gpt4_eval_results["relevancy"]
]

faithfulness_hamming_distance = hamming_distance(
    prometheus_faithfulness_scores, gpt4_faithfulness_scores
)
relevancy_hamming_distance = hamming_distance(
    prometheus_relevancy_scores, gpt4_relevancy_scores
)

print(f"Faithfulness Hamming Distance: {faithfulness_hamming_distance}")
print(f"Relevancy Hamming Distance: {relevancy_hamming_distance}")

#### Observation:
# 
# The comparison reveals that approximately `77%` and `81%` of the scores are common in case of both `Faithfulness` and `Relevancy` between Prometheus and GPT-4 evaluations respectively. This indicates a decent correlation in terms of faithfulness and relevance scoring between the Prometheus and GPT-4 models.

### GPT-4 Cost analysis

prompt_token_count = token_counter.prompt_llm_token_count
completion_token_count = token_counter.completion_llm_token_count

total_cost_paul_graham_essay = (
    prompt_token_count * 0.03 + completion_token_count * 0.06
) / 1000

token_counter.reset_counts()

## Evaluation with Llama2 paper

query_engine, rag_dataset = create_query_engine_rag_dataset("./data/llama2")

questions = [example.query for example in rag_dataset.examples]

reference = [[example.reference_answer] for example in rag_dataset.examples]

### Compute Correctness, Faithfulness and Relevancy Evaluation

prometheus_eval_results = await batch_eval_runner(
    prometheus_evaluators, query_engine, questions, reference
)

gpt4_eval_results = await batch_eval_runner(
    gpt4_evaluators, query_engine, questions, reference
)

### Correctness Evaluation score distribution with Prometheus Evaluator.

prometheus_scores = [
    result.score for result in prometheus_eval_results["correctness"]
]
get_scores_distribution(prometheus_scores)

### Correctness Evaluation score distribution with GPT-4 Evaluator.

gpt4_scores = [result.score for result in gpt4_eval_results["correctness"]]
get_scores_distribution(gpt4_scores)

### Feedback comparison between prometheus and gpt-4 for correctness.

query = prometheus_eval_results["correctness"][0].query
response = prometheus_eval_results["correctness"][0].response
reference_answer = reference[0][0]

prometheus_feedback = prometheus_eval_results["correctness"][0].feedback
prometheus_score = prometheus_eval_results["correctness"][0].score

gpt4_feedback = gpt4_eval_results["correctness"][0].feedback
gpt4_score = gpt4_eval_results["correctness"][0].score

print(f"Query: {query} \n\n")
print(f"Generated Answer: {response} \n\n")
print(f"Reference Answer: {reference_answer} \n\n")
print(
    f"Prometheus Feedback: {prometheus_feedback} \n\n {prometheus_score} \n\n"
)
print(f"GPT-4 Feedback: {gpt4_feedback} \n\n {gpt4_score}")

#### Observation:
# 
# The feedback from Prometheus is little more precise compared to GPT-4 and it penalises and gives a score of `3.0` but GPT-4 gives a score of `4.5`.

### Prometheus Faithfulness and Relevancy Evaluation scores.

_ = get_eval_results("faithfulness", prometheus_eval_results)

_ = get_eval_results("relevancy", prometheus_eval_results)

### GPT-4 Faithfulness and Relevancy Evaluation scores.

_ = get_eval_results("faithfulness", gpt4_eval_results)

_ = get_eval_results("relevancy", gpt4_eval_results)

### Hamming Distance comparison between Prometheus and GPT-4

prometheus_faithfulness_scores = [
    result.score for result in prometheus_eval_results["faithfulness"]
]
prometheus_relevancy_scores = [
    result.score for result in prometheus_eval_results["relevancy"]
]

gpt4_faithfulness_scores = [
    result.score for result in gpt4_eval_results["faithfulness"]
]
gpt4_relevancy_scores = [
    result.score for result in gpt4_eval_results["relevancy"]
]

faithfulness_hamming_distance = hamming_distance(
    prometheus_faithfulness_scores, gpt4_faithfulness_scores
)
relevancy_hamming_distance = hamming_distance(
    prometheus_relevancy_scores, gpt4_relevancy_scores
)

print(f"Faithfulness Hamming Distance: {faithfulness_hamming_distance}")
print(f"Relevancy Hamming Distance: {relevancy_hamming_distance}")

#### Observation:
# 
# The comparison reveals that approximately `44%` of the scores in case of `Faithfulness` and `63%` in case of `Relevancy` are common between Prometheus and GPT-4 evaluations. This indicates a decent amount of correlation in terms of faithfulness and relevance scoring between the Prometheus and GPT-4 models.

### Feedback comparison between prometheus and gpt-4 for faithfulness and relevancy

query = questions[0]

response = prometheus_eval_results["faithfulness"][0].response
contexts = prometheus_eval_results["faithfulness"][0].contexts

prometheus_faithfulness_feedback = prometheus_eval_results["faithfulness"][
    0
].feedback
prometheus_relevancy_feedback = prometheus_eval_results["relevancy"][
    0
].feedback

gpt4_faithfulness_feedback = gpt4_eval_results["faithfulness"][0].feedback
gpt4_relevancy_feedback = gpt4_eval_results["relevancy"][0].feedback

prometheus_faithfulness_score = prometheus_eval_results["faithfulness"][
    0
].score
prometheus_relevancy_score = prometheus_eval_results["relevancy"][0].score

gpt4_faithfulness_score = gpt4_eval_results["faithfulness"][0].score
gpt4_relevancy_score = gpt4_eval_results["relevancy"][0].score

print(f"Query: {query} \n\n")
print(f"Generated Answer: {response}")

print(f"Context-1: {contexts[0]}")

print(f"Context-2: {contexts[1]}")

print(
    f"Prometheus Faithfulness Feedback: {prometheus_faithfulness_feedback}\n\n"
)
print(f"Prometheus Faithfulness Score: {prometheus_faithfulness_score}\n\n")
print(f"Prometheus Relevancy Feedback: {prometheus_relevancy_feedback}\n\n")
print(f"Prometheus Relevancy Score: {prometheus_relevancy_score}")

#### If you compare the feedback and contexts, there is mention of range of parameters in the context and response but the feedback says the model could not find such information.

print(f"GPT-4 Faithfulness Feedback: {gpt4_faithfulness_feedback}\n\n")
print(f"GPT-4 Faithfulness Score: {gpt4_faithfulness_score}\n\n")
print(f"GPT-4 Relevancy Feedback: {gpt4_relevancy_feedback}\n\n")
print(f"GPT-4 Relevancy Score: {gpt4_relevancy_score}")

#### GPT-4 Evaluates it correctly, unlike prometheus model.

### GPT-4 Cost analysis

prompt_token_count = token_counter.prompt_llm_token_count
completion_token_count = token_counter.completion_llm_token_count

total_cost_llama2 = (
    prompt_token_count * 0.03 + completion_token_count * 0.06
) / 1000

## Total Cost Analysis

### Prometheus Model - `$2.167` for `144` queries (`44` for Paul Graham Essay and `100` for Llama2 paper) which accounts to `$0.015` per query.

### GPT4 Model - `$22` (total_cost_paul_graham_essay + total_cost_llama2) - which accounts to `$0.15` per query.

## Observation:
# 
# 1. The cost for evaluation (approx.): `$2.167` for Prometheus Model and `$22` for GPT4.
# 2. The Prometheus model, though offering more detailed feedback than GPT-4, occasionally provides incorrect feedback, necessitating cautious application.
# 3. If a generated answer lacks certain facts present in the reference answer, the Prometheus model applies stricter penalties to scores than GPT-4.
# 4. The faithfulness and relevancy feedback of Promethes shows more hallucinations/ wrong interpretations in the feedback compared to GPT-4.
# 5. The commonality between faithfulness and relevancy scores of Promethes and GPT-4 is different across two datasets and so should be used cautiously in production.
# 
# Note: The endpoint on HF is served on AWS Nvidia A100G · 1x GPU · 80 GB which costs $6.5/h. We used [Prometheus model](https://huggingface.co/kaist-ai/prometheus-13b-v1.0) for the analysis here. We also made similar analysis with [GPTQ Quantized version](https://huggingface.co/TheBloke/prometheus-13B-v1.0-GPTQ) of [Prometheus model](https://huggingface.co/kaist-ai/prometheus-13b-v1.0) and observed abit more hallucinations in feedback compared to original unquantized model. Thanks to authors of the paper and [Tom Jobbins](https://twitter.com/TheBlokeAI) for providing the quantized version of the model.

logger.info("\n\n[DONE]", bright=True)