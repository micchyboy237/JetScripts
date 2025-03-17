# %pip install llama-index-llms-openai
# %pip install llama-index-llms-huggingface-api
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from typing import List, Dict
from collections import Counter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llama_dataset import LabelledRagDataset
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import re
from typing import Tuple
from llama_index.llms.openai import OpenAI
import os
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.llama_dataset import download_llama_dataset
import nest_asyncio

nest_asyncio.apply()

paul_graham_rag_dataset, paul_graham_documents = download_llama_dataset(
    "PaulGrahamEssayDataset", "./data/paul_graham"
)

llama2_rag_dataset, llama2_documents = download_llama_dataset(
    "Llama2PaperDataset", "./data/llama2"
)

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
    
prometheus_correctness_eval_prompt_template = """

prometheus_faithfulness_eval_prompt_template = """###Task Description: An instruction (might include an Input inside it), an information, a context, and a score rubric representing evaluation criteria are given. 
	        1. You are provided with evaluation task with the help of information, context information to give result based on score rubrics.
            2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general. 
			3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric. 
            4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)” 
            5. Please do not generate any other opening, closing, and explanations. 



            
        Score YES: If the given piece of information is supported by context.
        Score NO: If the given piece of information is not supported by context
    

prometheus_faithfulness_refine_prompt_template = """

prometheus_relevancy_eval_prompt_template = """###Task Description: An instruction (might include an Input inside it), a query with response, context, and a score rubric representing evaluation criteria are given. 
            1. You are provided with evaluation task with the help of a query with response and context.
            2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general. 
			3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric. 
            4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)” 
            5. Please do not generate any other opening, closing, and explanations. 



            
        Score YES: If the response for the query is in line with the context information provided.
        Score NO: If the response for the query is not in line with the context information provided.
    

prometheus_relevancy_refine_prompt_template = """


os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"


gpt4_llm = OpenAI("gpt-4")


def parser_function(output_str: str) -> Tuple[float, str]:
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
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode
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


async def batch_eval_runner(
    evaluators, query_engine, questions, reference=None, num_workers=8
):
    batch_runner = BatchEvalRunner(
        evaluators, workers=num_workers, show_progress=True
    )

    eval_results = await batch_runner.aevaluate_queries(
        query_engine, queries=questions, reference=reference
    )

    return eval_results


def get_scores_distribution(scores: List[float]) -> Dict[str, float]:
    score_counts = Counter(scores)

    total_scores = len(scores)

    percentage_distribution = {
        score: (count / total_scores) * 100
        for score, count in score_counts.items()
    }

    return percentage_distribution


def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {round(score, 2)}")
    return score


def hamming_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(list1, list2))


query_engine, rag_dataset = create_query_engine_rag_dataset(
    "./data/paul_graham"
)
questions = [example.query for example in rag_dataset.examples]

reference = [[example.reference_answer] for example in rag_dataset.examples]
prometheus_eval_results = await batch_eval_runner(
    prometheus_evaluators, query_engine, questions, reference
)
gpt4_eval_results = await batch_eval_runner(
    gpt4_evaluators, query_engine, questions, reference
)
prometheus_scores = [
    result.score for result in prometheus_eval_results["correctness"]
]
get_scores_distribution(prometheus_scores)
gpt4_scores = [result.score for result in gpt4_eval_results["correctness"]]
get_scores_distribution(gpt4_scores)
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
_ = get_eval_results("faithfulness", prometheus_eval_results)

_ = get_eval_results("relevancy", prometheus_eval_results)
_ = get_eval_results("faithfulness", gpt4_eval_results)

_ = get_eval_results("relevancy", gpt4_eval_results)
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
prompt_token_count = token_counter.prompt_llm_token_count
completion_token_count = token_counter.completion_llm_token_count

total_cost_paul_graham_essay = (
    prompt_token_count * 0.03 + completion_token_count * 0.06
) / 1000

token_counter.reset_counts()
query_engine, rag_dataset = create_query_engine_rag_dataset("./data/llama2")
questions = [example.query for example in rag_dataset.examples]
reference = [[example.reference_answer] for example in rag_dataset.examples]
prometheus_eval_results = await batch_eval_runner(
    prometheus_evaluators, query_engine, questions, reference
)
gpt4_eval_results = await batch_eval_runner(
    gpt4_evaluators, query_engine, questions, reference
)
prometheus_scores = [
    result.score for result in prometheus_eval_results["correctness"]
]
get_scores_distribution(prometheus_scores)
gpt4_scores = [result.score for result in gpt4_eval_results["correctness"]]
get_scores_distribution(gpt4_scores)
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
_ = get_eval_results("faithfulness", prometheus_eval_results)

_ = get_eval_results("relevancy", prometheus_eval_results)
_ = get_eval_results("faithfulness", gpt4_eval_results)

_ = get_eval_results("relevancy", gpt4_eval_results)
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
print(f"GPT-4 Faithfulness Feedback: {gpt4_faithfulness_feedback}\n\n")
print(f"GPT-4 Faithfulness Score: {gpt4_faithfulness_score}\n\n")
print(f"GPT-4 Relevancy Feedback: {gpt4_relevancy_feedback}\n\n")
print(f"GPT-4 Relevancy Score: {gpt4_relevancy_score}")
prompt_token_count = token_counter.prompt_llm_token_count
completion_token_count = token_counter.completion_llm_token_count

total_cost_llama2 = (
    prompt_token_count * 0.03 + completion_token_count * 0.06
) / 1000
