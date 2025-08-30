from datasets import load_dataset
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.prediction import Prediction
from dspy.teleprompt import BootstrapFewShot
from haystack import Document
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.evaluators import SASEvaluator
from haystack.components.generators import OllamaFunctionCallingAdapterGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from jet.logger import CustomLogger
from rich import print
import dspy
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Prompt Optimization with DSPy

<img src="https://raw.githubusercontent.com/stanfordnlp/dspy/main/docs/images/DSPy8.png" width="400" style="display:inline;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://haystack.deepset.ai/images/haystack-ogimage.png" width="430" style="display:inline;">

When building applications with LLMs, writing effective prompts is a long process of trial and error.
Often, if you switch models, you also have to change the prompt.
What if you could automate this process?

That's where **DSPy** comes in - a framework designed to algorithmically optimize prompts for Language Models.
By applying classical machine learning concepts (training and evaluation data, metrics, optimization), DSPy generates better prompts for a given model and task.

In this notebook, we will see **how to combine DSPy with the robustness of Haystack Pipelines**.
- ▶️ Start from a Haystack RAG pipeline with a basic prompt
- 🎯 Define a goal (in this case, get correct and concise answers)
- 📊 Create a DSPy program, define data and metrics
- ✨ Optimize and evaluate -> improved prompt
- 🚀 Build a refined Haystack RAG pipeline using the optimized prompt

## Setup
"""
logger.info("# Prompt Optimization with DSPy")

# ! pip install haystack-ai datasets dspy-ai sentence-transformers

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter OllamaFunctionCallingAdapter API key:")

"""
## Load data

We will use the first 1000 rows of a [labeled PubMed dataset](https://huggingface.co/datasets/vblagoje/PubMedQA_instruction/viewer/default/train?row=0) with questions, contexts and answers.

Initially, we will use only the contexts as documents and write them to a Document Store.

(Later, we will also use the questions and answers from a small subset of the dataset to create training and dev sets for optimization.)
"""
logger.info("## Load data")


dataset = load_dataset("vblagoje/PubMedQA_instruction", split="train")
dataset = dataset.select(range(1000))
docs = [Document(content=doc["context"]) for doc in dataset]


document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

document_store.filter_documents()[:5]

"""
## Initial Haystack pipeline

Let's create a simple RAG Pipeline in Haystack. For more information, see [the documentation](https://docs.haystack.deepset.ai/docs/get_started).

Next, we will see how to improve the prompt.
"""
logger.info("## Initial Haystack pipeline")



retriever = InMemoryBM25Retriever(document_store, top_k=3)
generator = OllamaFunctionCallingAdapterGenerator(model="llama3.2")

template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)


rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)

rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

"""
Let's ask some questions...
"""
logger.info("Let's ask some questions...")

question = "What effects does ketamine have on rat neural stem cells?"

response = rag_pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})

logger.debug(response["llm"]["replies"][0])

question = "Is the anterior cingulate cortex linked to pain-induced depression?"

response = rag_pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})

logger.debug(response["llm"]["replies"][0])

"""
The answers seems correct, but suppose that **our use case requires shorter answers**. How can we adjust the prompt to achieve this effect while maintaining correctness?

## DSPy

We will use DSPy to automatically improve the prompt for our goal: getting correct and short answers.

We will perform several steps:
- define a DSPy module for RAG
- create training and dev sets
- define a metric
- evaluate the unoptimized RAG module
- optimize the module
- evaluate the optimized RAG

Broadly speaking, these steps follow those listed in the [DSPy guide](https://dspy-docs.vercel.app/docs/building-blocks/solving_your_task).
"""
logger.info("## DSPy")



lm = dspy.OllamaFunctionCallingAdapter(model='llama3.2')
dspy.settings.configure(lm=lm)

"""
### DSPy Signature

The RAG module involves two main tasks (smaller modules): retrieval and generation.

For generation, we need to define a signature: a declarative specification of input/output behavior of a DSPy module.
In particular, the generation module receives the `context` and a `question` as input and returns an `answer`.

In DSPy, the docstring and the field description are used to create the prompt.
"""
logger.info("### DSPy Signature")

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="short and precise answer")

"""
### DSPy RAG module

- the `__init__` method can be used to declare sub-modules.
- the logic of the module is contained in the `forward` method.
---
- `ChainOfThought` module encourages Language Model reasoning with a specific prompt ("Let's think step by step") and examples. [Paper](https://arxiv.org/abs/2201.11903)
- we want to reuse our Haystack retriever and the already indexed data, so we also define a `retrieve` method.
"""
logger.info("### DSPy RAG module")

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def retrieve(self, question):
        results = retriever.run(query=question)
        passages = [res.content for res in results['documents']]
        return Prediction(passages=passages)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

"""
### Create training and dev sets

In general, to use DSPy for prompt optimization, you have to prepare some examples for your task (or use a similar dataset).

The training set is used for optimization, while the dev set is used for evaluation.

We create them using respectively 20 and 50 examples (question and answer) from our original labeled PubMed dataset.
"""
logger.info("### Create training and dev sets")

trainset, devset=[],[]

for i,ex in enumerate(dataset):
  example = dspy.Example(question = ex["instruction"], answer=ex["response"]).with_inputs('question')

  if i<20:
    trainset.append(example)
  elif i<70:
    devset.append(example)
  else:
    break

"""
### Define a metric

Defining a metric is a crucial step for evaluating and optimizing our prompt.

As we show in this example, metrics can be defined in a very customized way.

In our case, we want to focus on two aspects: correctness and brevity of the answers.
- for correctness, we use semantic similarity between the predicted answer and the ground truth answer ([Haystack SASEvaluator](https://docs.haystack.deepset.ai/docs/sasevaluator)). SAS score varies between 0 and 1.
- to encourage short answers, we add a penalty for long answers based on a simple mathematical formulation. The penalty varies between 0 (for answers of 20 words or less) and 0.5 (for answers of 40 words or more).
"""
logger.info("### Define a metric")

sas_evaluator = SASEvaluator()
sas_evaluator.warm_up()

def mixed_metric(example, pred, trace=None):
    semantic_similarity = sas_evaluator.run(ground_truth_answers=[example.answer], predicted_answers=[pred.answer])["score"]

    n_words=len(pred.answer.split())
    long_answer_penalty=0
    if 20<n_words<40:
      long_answer_penalty = 0.025 * (n_words - 20)
    elif n_words>=40:
      long_answer_penalty = 0.5

    return semantic_similarity - long_answer_penalty

"""
### Evaluate unoptimized RAG module

Let's first check how the unoptimized RAG module performs on the dev set.
Then we will optimize it.
"""
logger.info("### Evaluate unoptimized RAG module")

uncompiled_rag = RAG()


evaluate = Evaluate(
    metric=mixed_metric, devset=devset, num_threads=1, display_progress=True, display_table=5
)
evaluate(uncompiled_rag)

"""
### Optimization

We can now compile/optimized the DSPy program we created.

This can be done using a teleprompter/optimizer, based on our metric and training set.

In particular, `BootstrapFewShot` tries to improve the metric in the training set by adding few shot examples to the prompt.
"""
logger.info("### Optimization")


optimizer = BootstrapFewShot(metric=mixed_metric)

compiled_rag = optimizer.compile(RAG(), trainset=trainset)

"""
### Evaluate optimized RAG module

Let's now see if the training has been successful, evaluating the compiled RAG module on the dev set.
"""
logger.info("### Evaluate optimized RAG module")

evaluate = Evaluate(
    metric=mixed_metric, devset=devset, num_threads=1, display_progress=True, display_table=5
)
evaluate(compiled_rag)

"""
Based on our simple metric, we got a significant improvement!

### Inspect the optimized prompt

Let's take a look at the few shot examples that made our results improve...
"""
logger.info("### Inspect the optimized prompt")

lm.inspect_history(n=1)

"""
## Optimized Haystack Pipeline

We can now use the static part of the optimized prompt (including examples) and create a better Haystack RAG Pipeline.

We include an `AnswerBuilder`, to capture only the relevant part of the generation (all text after `Answer: `).
"""
logger.info("## Optimized Haystack Pipeline")

# %%capture

static_prompt = lm.inspect_history(n=1).rpartition("---\n")[0]



template = static_prompt+"""
---

Context:
{% for document in documents %}
    «{{ document.content }}»
{% endfor %}

Question: {{question}}
Reasoning: Let's think step by step in order to
"""

new_prompt_builder = PromptBuilder(template=template)

new_retriever = InMemoryBM25Retriever(document_store, top_k=3)
new_generator = OllamaFunctionCallingAdapterGenerator(model="llama3.2")

answer_builder = AnswerBuilder(pattern="Answer: (.*)")


optimized_rag_pipeline = Pipeline()
optimized_rag_pipeline.add_component("retriever", new_retriever)
optimized_rag_pipeline.add_component("prompt_builder", new_prompt_builder)
optimized_rag_pipeline.add_component("llm", new_generator)
optimized_rag_pipeline.add_component("answer_builder", answer_builder)

optimized_rag_pipeline.connect("retriever", "prompt_builder.documents")
optimized_rag_pipeline.connect("prompt_builder", "llm")
optimized_rag_pipeline.connect("llm.replies", "answer_builder.replies")

"""
Let's ask the same questions as before...
"""
logger.info("Let's ask the same questions as before...")

question = "What effects does ketamine have on rat neural stem cells?"

response = optimized_rag_pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}, "answer_builder": {"query": question}})

logger.debug(response["answer_builder"]["answers"][0].data)

question = "Is the anterior cingulate cortex linked to pain-induced depression?"
response = optimized_rag_pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}, "answer_builder": {"query": question}})

logger.debug(response["answer_builder"]["answers"][0].data)

"""
The answer are correct and shorter than before!

*(Notebook by [Stefano Fiorucci](https://github.com/anakin87))*
"""
logger.info("The answer are correct and shorter than before!")

logger.info("\n\n[DONE]", bright=True)