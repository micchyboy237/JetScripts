from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/low_level/evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Building Evaluation from Scratch
# 
# We show how you can build evaluation modules from scratch. This includes both evaluation of the final generated response (where the output is plain text), as well as the evaluation of retrievers (where the output is a ranked list of items).
# 
# We have in-house modules in our [Evaluation](https://gpt-index.readthedocs.io/en/latest/core_modules/supporting_modules/evaluation/root.html) section.

## Setup
# 
# We load some data and define a very simple RAG query engine that we'll evaluate (uses top-k retrieval).

# %pip install llama-index-readers-file pymupdf
# %pip install llama-index-llms-ollama

# !mkdir data
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

from pathlib import Path
from llama_index.readers.file import PyMuPDFReader

loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)
node_parser = SentenceSplitter(chunk_size=1024)

nodes = node_parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)

query_engine = index.as_query_engine(llm=llm)

## Dataset Generation
# 
# We first go through an exercise of generating a synthetic evaluation dataset. We do this by synthetically generating a set of questions from existing context. We then run each question with existing context through a powerful LLM (e.g. GPT-4) to generate a "ground-truth" response.

### Define Functions
# 
# We define the functions that we will use for dataset generation:

from llama_index.core.schema import BaseNode
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, PromptTemplate
from typing import Tuple, List
import re

llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)

# We define `generate_answers_for_questions` to generate answers from questions given context.

QA_PROMPT = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)


def generate_answers_for_questions(
    questions: List[str], context: str, llm: Ollama
) -> str:
    """Generate answers for questions given context."""
    answers = []
    for question in questions:
        fmt_qa_prompt = QA_PROMPT.format(
            context_str=context, query_str=question
        )
        response_obj = llm.complete(fmt_qa_prompt)
        answers.append(str(response_obj))
    return answers

# We define `generate_qa_pairs` to generate qa pairs over an entire list of Nodes.

QUESTION_GEN_USER_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "generate the relevant questions. "
)

QUESTION_GEN_SYS_TMPL = """\
You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided.\
"""

question_gen_template = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=QUESTION_GEN_SYS_TMPL),
        ChatMessage(role=MessageRole.USER, content=QUESTION_GEN_USER_TMPL),
    ]
)


def generate_qa_pairs(
    nodes: List[BaseNode], llm: Ollama, num_questions_per_chunk: int = 10
) -> List[Tuple[str, str]]:
    """Generate questions."""
    qa_pairs = []
    for idx, node in enumerate(nodes):
        print(f"Node {idx}/{len(nodes)}")
        context_str = node.get_content(metadata_mode="all")
        fmt_messages = question_gen_template.format_messages(
            num_questions_per_chunk=10,
            context_str=context_str,
        )
        chat_response = llm.chat(fmt_messages)
        raw_output = chat_response.message.content
        result_list = str(raw_output).strip().split("\n")
        cleaned_questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip()
            for question in result_list
        ]
        answers = generate_answers_for_questions(
            cleaned_questions, context_str, llm
        )
        cur_qa_pairs = list(zip(cleaned_questions, answers))
        qa_pairs.extend(cur_qa_pairs)
    return qa_pairs

qa_pairs

### Getting Pairs over Dataset
# 
# **NOTE**: This can take a long time. For the sake of speed try inputting a subset of the nodes.

qa_pairs = generate_qa_pairs(
    nodes,
    llm,
    num_questions_per_chunk=10,
)

#### [Optional] Define save/load

import pickle

pickle.dump(qa_pairs, open("eval_dataset.pkl", "wb"))

import pickle

qa_pairs = pickle.load(open("eval_dataset.pkl", "rb"))

## Evaluating Generation
# 
# In this section we walk through a few methods for evaluating the generated results. At a high-level we use an "evaluation LLM" to measure the quality of the generated results. We do this in both the **with labels** setting and **without labels** setting. 
# 
# We go through the following evaluation algorithms:
# - **Correctness**: Compares the generated answer against the ground-truth answer.
# - **Faithfulness**: Evaluates whether a response is faithful to the contexts (label-free).

### Building a Correctness Evaluator
# 
# The correctness evaluator compares the generated answer to the reference ground-truth answer, given the query. We output a score between 1 and 5, where 1 is the worst and 5 is the best.
# 
# We do this through a system and user prompt with a chat interface.

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, PromptTemplate
from typing import Dict

CORRECTNESS_SYS_TMPL = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query, 
- a reference answer, and
- a generated answer.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, \
you should give a score of 1.
- If the generated answer is relevant but contains mistakes, \
you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, \
you should give a score between 4 and 5.
"""

CORRECTNESS_USER_TMPL = """
{query}

{reference_answer}

{generated_answer}
"""

eval_chat_template = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=CORRECTNESS_SYS_TMPL),
        ChatMessage(role=MessageRole.USER, content=CORRECTNESS_USER_TMPL),
    ]
)

# Now that we've defined the prompts template, let's define an evaluation function that feeds the prompt to the LLM and parses the output into a dict of results.

from llama_index.llms.ollama import Ollama


def run_correctness_eval(
    query_str: str,
    reference_answer: str,
    generated_answer: str,
    llm: Ollama,
    threshold: float = 4.0,
) -> Dict:
    """Run correctness eval."""
    fmt_messages = eval_chat_template.format_messages(
        llm=llm,
        query=query_str,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )
    chat_response = llm.chat(fmt_messages)
    raw_output = chat_response.message.content

    score_str, reasoning_str = raw_output.split("\n", 1)
    score = float(score_str)
    reasoning = reasoning_str.lstrip("\n")

    return {"passing": score >= threshold, "score": score, "reason": reasoning}

# Now let's try running this on some sample inputs with a chat model (GPT-4).

llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)

query_str = (
    "What is the specific name given to the fine-tuned LLMs optimized for"
    " dialogue use cases?"
)
reference_answer = (
    "The specific name given to the fine-tuned LLMs optimized for dialogue use"
    " cases is Llama 2-Chat."
)

generated_answer = str(query_engine.query(query_str))

print(str(generated_answer))

eval_results = run_correctness_eval(
    query_str, reference_answer, generated_answer, llm=llm, threshold=4.0
)
display(eval_results)

### Building a Faithfulness Evaluator
# 
# The faithfulness evaluator evaluates whether the response is faithful to any of the retrieved contexts.
# 
# This is a step up in complexity from the correctness evaluator. Since the set of contexts can be quite long, they might overflow the context window. We would need to figure out how to implement a form of **response synthesis** strategy to iterate over contexts in sequence.
# 
# We have a corresponding tutorial showing you [how to build response synthesis from scratch](https://gpt-index.readthedocs.io/en/latest/examples/low_level/response_synthesis.html). We also have [out-of-the-box response synthesis modules](https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/response_synthesizers/root.html). In this guide we'll use the out of the box modules.

EVAL_TEMPLATE = PromptTemplate(
    "Please tell if a given piece of information "
    "is supported by the context.\n"
    "You need to answer with either YES or NO.\n"
    "Answer YES if any of the context supports the information, even "
    "if most of the context is unrelated. "
    "Some examples are provided below. \n\n"
    "Information: Apple pie is generally double-crusted.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: YES\n"
    "Information: Apple pies tastes bad.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: NO\n"
    "Information: {query_str}\n"
    "Context: {context_str}\n"
    "Answer: "
)

EVAL_REFINE_TEMPLATE = PromptTemplate(
    "We want to understand if the following information is present "
    "in the context information: {query_str}\n"
    "We have provided an existing YES/NO answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "If the existing answer was already YES, still answer YES. "
    "If the information is present in the new context, answer YES. "
    "Otherwise answer NO.\n"
)

# **NOTE**: In the current response synthesizer setup we don't separate out a system and user message for chat endpoints, so we just use our standard `llm.complete` for text completion.
# 
# We now define our function below. Since we defined both a standard eval template for a given piece of context but also a refine template for subsequent contexts, we implement our "create-and-refine" response synthesis strategy to obtain the answer.

from llama_index.core.response_synthesizers import Refine
from typing import List, Dict


def run_faithfulness_eval(
    generated_answer: str,
    contexts: List[str],
    llm: Ollama,
) -> Dict:
    """Run faithfulness eval."""

    refine = Refine(
        llm=llm,
        text_qa_template=EVAL_TEMPLATE,
        refine_template=EVAL_REFINE_TEMPLATE,
    )

    response_obj = refine.get_response(generated_answer, contexts)
    response_txt = str(response_obj)

    if "yes" in response_txt.lower():
        passing = True
    else:
        passing = False

    return {"passing": passing, "reason": str(response_txt)}

# Let's try it out on some data

response = query_engine.query(query_str)
generated_answer = str(response)

context_list = [n.get_content() for n in response.source_nodes]
eval_results = run_faithfulness_eval(
    generated_answer,
    contexts=context_list,
    llm=llm,
)
display(eval_results)

## Running Evaluation over our Eval Dataset
# 
# Now let's tie the two above sections together and run our eval modules over our eval dataset!
# 
# **NOTE**: For the sake of speed/cost we extract a very limited sample.

import random

sample_size = 5
qa_pairs_sample = random.sample(qa_pairs, sample_size)

import pandas as pd


def run_evals(qa_pairs: List[Tuple[str, str]], llm: Ollama, query_engine):
    results_list = []
    for question, reference_answer in qa_pairs:
        response = query_engine.query(question)
        generated_answer = str(response)
        correctness_results = run_correctness_eval(
            query_str,
            reference_answer,
            generated_answer,
            llm=llm,
            threshold=4.0,
        )
        faithfulness_results = run_faithfulness_eval(
            generated_answer,
            contexts=context_list,
            llm=llm,
        )
        cur_result_dict = {
            "correctness": correctness_results["passing"],
            "faithfulness": faithfulness_results["passing"],
        }
        results_list.append(cur_result_dict)
    return pd.DataFrame(results_list)

evals_df = run_evals(qa_pairs_sample, llm, query_engine)

evals_df["correctness"].mean()

evals_df["faithfulness"].mean()

logger.info("\n\n[DONE]", bright=True)