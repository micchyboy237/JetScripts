from datasets import load_dataset
from haystack import Document
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
# from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator, DeepEvalMetric
from jet.adapters.haystack.ollama_chat_generator import OllamaChatGenerator
from deepeval.models import OllamaModel
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
# RAG pipeline evaluation using DeepEval

[DeepEval](https://deepeval.com/) is a framework to evaluate [Retrieval Augmented Generation](https://www.deepset.ai/blog/llms-retrieval-augmentation) (RAG) pipelines.
It supports metrics like context relevance, answer correctness, faithfulness, and more.

For more information about evaluators, supported metrics and usage, check out:

* [DeepEvalEvaluator](https://docs.haystack.deepset.ai/docs/deepevalevaluator)
* [Model based evaluation](https://docs.haystack.deepset.ai/docs/model-based-evaluation)

This notebook shows how to use [DeepEval-Haystack](https://haystack.deepset.ai/integrations/deepeval) integration to evaluate a RAG pipeline against various metrics.

## Prerequisites:

- [Ollama](https://ollama.com/) key
    - **DeepEval** uses  for computing some metrics, so we need an Ollama key.
"""
logger.info("# RAG pipeline evaluation using DeepEval")

# from getpass import getpass

# os.environ["OPENAI_API_KEY"] = getpass("Enter Ollama API key:")

"""
## Install dependencies
"""
logger.info("## Install dependencies")

# !pip install haystack-ai
# !pip install "datasets>=2.6.1"
# !pip install deepeval-haystack

"""
## Create a RAG pipeline

We'll first need to create a RAG pipeline. Refer to this [link](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline) for a detailed tutorial on how to create RAG pipelines.

In this notebook, we're using the [SQUAD V2](https://huggingface.co/datasets/rajpurkar/squad_v2) dataset for getting the context, questions and ground truth answers.

**Initialize the document store**
"""
logger.info("## Create a RAG pipeline")


document_store = InMemoryDocumentStore()

dataset = load_dataset("rajpurkar/squad_v2", split="validation")
documents = list(set(dataset["context"]))
docs = [Document(content=doc) for doc in documents]
document_store.write_documents(docs)


retriever = InMemoryBM25Retriever(document_store, top_k=3)

chat_message = ChatMessage.from_user(
    text="""Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
)
chat_prompt_builder = ChatPromptBuilder(template=[chat_message], required_variables="*")
chat_generator = OllamaChatGenerator(model="qwen3:4b-q4_K_M")

"""
**Build the RAG pipeline**
"""


rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
rag_pipeline.add_component("llm", chat_generator)
rag_pipeline.add_component("answer_builder", AnswerBuilder())

rag_pipeline.connect("retriever", "chat_prompt_builder.documents")
rag_pipeline.connect("chat_prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")

"""
**Running the pipeline**
"""

question = "In what country is Normandy located?"

response = rag_pipeline.run(
    {"retriever": {"query": question}, "chat_prompt_builder": {"question": question}, "answer_builder": {"query": question}}
)

logger.debug(response["answer_builder"]["answers"][0].data)

"""
We're done building our RAG pipeline. Let's evaluate it now!

## Get questions, contexts, responses and ground truths for evaluation

For computing most metrics, we will need to provide the following to the evaluator:
1. Questions
2. Generated responses
3. Retrieved contexts
4. Ground truth (Specifically, this is needed for `context precision`, `context recall` and `answer correctness` metrics)

We'll start with random three questions from the dataset (see below) and now we'll get the matching `contexts` and `responses` for those questions.

### Helper function to get context and responses for our questions
"""
logger.info("## Get questions, contexts, responses and ground truths for evaluation")

def get_contexts_and_responses(questions, pipeline):
    contexts = []
    responses = []
    for question in questions:
        response = pipeline.run(
            {
                "retriever": {"query": question},
                "chat_prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        responses.append(response["answer_builder"]["answers"][0].data)
    return contexts, responses

question_map = {
    "Which mountain range influenced the split of the regions?": 0,
    "What is the prize offered for finding a solution to P=NP?": 1,
    "Which Californio is located in the upper part?": 2
}
questions = list(question_map.keys())
contexts, responses = get_contexts_and_responses(questions, rag_pipeline)

"""
### Ground truths, review all fields

Now that we have questions, contexts, and responses we'll also get the matching ground truth answers.
"""
logger.info("### Ground truths, review all fields")

ground_truths = [""] * len(question_map)

for question, index in question_map.items():
    idx = dataset["question"].index(question)
    ground_truths[index] = dataset["answers"][idx]["text"][0]

logger.debug("Questions:\n")
logger.debug("\n".join(questions))

logger.debug("Contexts:\n")
for c in contexts:
  logger.debug(c[0])

logger.debug("Responses:\n")
logger.debug("\n".join(responses))

logger.debug("Ground truths:\n")
logger.debug("\n".join(ground_truths))

"""
## Evaluate the RAG pipeline

Now that we have the `questions`, `contexts`,`responses` and the `ground truths`, we can begin our pipeline evaluation and compute all the supported metrics.

## Metrics computation

In addition to evaluating the final responses of the LLM, it is important that we also evaluate the individual components of the RAG pipeline as they can significantly impact the overall performance. Therefore, there are different metrics to evaluate the retriever, the generator and the overall pipeline. For a full list of available metrics and their expected inputs, check out the [DeepEvalEvaluator Docs](https://docs.haystack.deepset.ai/docs/deepevalevaluator)

The [DeepEval documentation](https://deepeval.com/docs/metrics-introduction) provides explanation of the individual metrics with simple examples for each of them.

### Contextul Precision

The contextual precision metric measures our RAG pipeline's retriever by evaluating whether items in our contexts that are relevant to the given input are ranked higher than irrelevant ones.
"""
logger.info("## Evaluate the RAG pipeline")


context_precision_pipeline = Pipeline()
evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.CONTEXTUAL_PRECISION,
    metric_params={
        "model": OllamaModel(model="qwen3:4b-q4_K_M", temperature=0.0),
    },
)
context_precision_pipeline.add_component("evaluator", evaluator)

evaluation_results = context_precision_pipeline.run(
    {"evaluator": {"questions": questions, "contexts": contexts, "ground_truths": ground_truths, "responses": responses}}
)
logger.debug(evaluation_results["evaluator"]["results"])

"""
### Contextual Recall

Contextual recall measures the extent to which the contexts aligns with the `ground truth`.
"""
logger.info("### Contextual Recall")


context_recall_pipeline = Pipeline()
evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.CONTEXTUAL_RECALL,
    metric_params={
        "model": OllamaModel(model="qwen3:4b-q4_K_M", temperature=0.0),
    },
)
context_recall_pipeline.add_component("evaluator", evaluator)

evaluation_results = context_recall_pipeline.run(
    {"evaluator": {"questions": questions, "contexts": contexts, "ground_truths": ground_truths, "responses": responses}}
)
logger.debug(evaluation_results["evaluator"]["results"])

"""
### Contextual Relevancy

The contextual relevancy metric measures the quality of our RAG pipeline's retriever by evaluating the overall relevance of the context for a given question.
"""
logger.info("### Contextual Relevancy")


context_relevancy_pipeline = Pipeline()
evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.CONTEXTUAL_RELEVANCE,
    metric_params={
        "model": OllamaModel(model="qwen3:4b-q4_K_M", temperature=0.0),
    },
)
context_relevancy_pipeline.add_component("evaluator", evaluator)

evaluation_results = context_relevancy_pipeline.run(
    {"evaluator": {"questions": questions, "contexts": contexts, "responses": responses}}
)
logger.debug(evaluation_results["evaluator"]["results"])

"""
### Answer relevancy

The answer relevancy metric measures the quality of our RAG pipeline's response by evaluating how relevant the response is compared to the provided question.
"""
logger.info("### Answer relevancy")


answer_relevancy_pipeline = Pipeline()
evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.ANSWER_RELEVANCY,
    metric_params={
        "model": OllamaModel(model="qwen3:4b-q4_K_M", temperature=0.0),
    },
)
answer_relevancy_pipeline.add_component("evaluator", evaluator)

evaluation_results = answer_relevancy_pipeline.run(
    {"evaluator": {"questions": questions, "responses": responses, "contexts": contexts}}
)
logger.debug(evaluation_results["evaluator"]["results"])

"""
### Faithfulness

The faithfulness metric measures the quality of our RAG pipeline's responses by evaluating whether the response factually aligns with the contents of context we provided.
"""
logger.info("### Faithfulness")


faithfulness_pipeline = Pipeline()
evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.FAITHFULNESS,
    metric_params={
        "model": OllamaModel(model="qwen3:4b-q4_K_M", temperature=0.0),
    },
 )
faithfulness_pipeline.add_component("evaluator", evaluator)

evaluation_results = faithfulness_pipeline.run(
    {"evaluator": {"questions": questions, "contexts": contexts, "responses": responses}}
)
logger.debug(evaluation_results["evaluator"]["results"])

"""
**Our pipeline evaluation using DeepEval is now complete!**

**Haystack Useful Sources**

* [Docs](https://docs.haystack.deepset.ai/docs/intro)
* [Tutorials](https://haystack.deepset.ai/tutorials)
* [Other Cookbooks](https://github.com/deepset-ai/haystack-cookbook)
"""

logger.info("\n\n[DONE]", bright=True)