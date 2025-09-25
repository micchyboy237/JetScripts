from flow_judge import Hf
from flow_judge.integrations.haystack import HaystackFlowJudge
from flow_judge.metrics.presets import RESPONSE_FAITHFULNESS_5POINT
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.evaluators import ContextRelevanceEvaluator, FaithfulnessEvaluator, SASEvaluator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation import EvaluationRunResult
from jet.logger import logger
from typing import List, Tuple
import json
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
# Evaluating AI with Haystack

by Bilge Yucel ([X](https://x.com/bilgeycl), [Linkedin](https://www.linkedin.com/in/bilge-yucel/))

In this cookbook, we walk through the [Evaluators](https://docs.haystack.deepset.ai/docs/evaluators) in Haystack, create an evaluation pipeline and try different Evaluation Frameworks like [FlowJudge](https://haystack.deepset.ai/integrations/flow-judge). 

📚 **Useful Resources:**
* [Article: Benchmarking Haystack Pipelines for Optimal Performance](https://haystack.deepset.ai/blog/benchmarking-haystack-pipelines)
* [Evaluation Walkthrough](https://haystack.deepset.ai/tutorials/guide_evaluation)
* [Evaluation tutorial](https://haystack.deepset.ai/tutorials/35_evaluating_rag_pipelines)
* [Evaluation Docs](https://docs.haystack.deepset.ai/docs/evaluation)
* [haystack-evaluation repo](https://github.com/deepset-ai/haystack-evaluation/tree/main)

## 📺 Watch Along

<iframe width="560" height="315" src="https://www.youtube.com/embed/Dy-n_yC3Cto?si=LB0GdFP0VO-nJT-n" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
"""
logger.info("# Evaluating AI with Haystack")

# !pip install haystack-ai "sentence-transformers>=3.0.0" pypdf "flow-judge[hf]"

"""
## 1. Building your pipeline

### ARAGOG

This dataset is based on the paper [Advanced Retrieval Augmented Generation Output Grading (ARAGOG)](https://arxiv.org/pdf/2404.01037). It's a
collection of papers from ArXiv covering topics around Transformers and Large Language Models, all in PDF format.

The dataset contains:
- 13 PDF papers.
- 107 questions and answers generated with the assistance of GPT-4, and validated/corrected by humans.

We have:
- ground-truth answers
- questions

Get the dataset [here](https://github.com/deepset-ai/haystack-evaluation/blob/main/datasets/README.md)
"""
logger.info("## 1. Building your pipeline")

# !mkdir -p ARAGOG/papers_for_questions

# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/DetectGPT.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/MMLU_measure.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/PAL.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/bert.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/codenet.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/distilbert.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/glm_130b.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/hellaswag.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/llama.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/llm_long_tail.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/meaning_of_prompt.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/megatron.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/red_teaming.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/roberta.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/superglue.pdf -P ARAGOG/papers_for_questions
# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/papers_for_questions/task2vec.pdf -P ARAGOG/papers_for_questions

"""
### Indexing Pipeline
"""
logger.info("### Indexing Pipeline")



embedding_model="sentence-transformers/all-MiniLM-L6-v2"
document_store = InMemoryDocumentStore()

files_path = "./ARAGOG/papers_for_questions"
pipeline = Pipeline()
pipeline.add_component("converter", PyPDFToDocument())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_length=250, split_by="word"))  # default splitting by word
pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "embedder")
pipeline.connect("embedder", "writer")
pdf_files = [files_path + "/" + f_name for f_name in os.listdir(files_path)]

pipeline.run({"converter": {"sources": pdf_files}})

document_store.count_documents()

"""
### RAG
"""
logger.info("### RAG")

# from getpass import getpass

# if not os.getenv("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass('OPENAI_API_KEY: ')


chat_message = ChatMessage.from_user(
    text="""You have to answer the following question based on the given context information only.
If the context is empty or just a '\\n' answer with None, example: "None".

Context:
{% for document in documents %}
  {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
)

basic_rag = Pipeline()
basic_rag.add_component("query_embedder", SentenceTransformersTextEmbedder(
    model=embedding_model, progress_bar=False
))
basic_rag.add_component("retriever", InMemoryEmbeddingRetriever(document_store))
basic_rag.add_component("chat_prompt_builder", ChatPromptBuilder(template=[chat_message], required_variables="*"))
basic_rag.add_component("chat_generator", OpenAIChatGenerator(model="llama3.2"))

basic_rag.connect("query_embedder", "retriever.query_embedding")
basic_rag.connect("retriever", "chat_prompt_builder.documents")
basic_rag.connect("chat_prompt_builder", "chat_generator")

"""
## 2. Human Evaluation
"""
logger.info("## 2. Human Evaluation")

# !wget https://raw.githubusercontent.com/deepset-ai/haystack-evaluation/main/datasets/ARAGOG/eval_questions.json -P ARAGOG


def read_question_answers() -> Tuple[List[str], List[str]]:
    with open("./ARAGOG/eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
    return questions, answers

all_questions, all_answers = read_question_answers()

logger.debug(len(all_questions))
logger.debug(len(all_answers))

questions = all_questions[:15]
answers = all_answers[:15]

index = 5
logger.debug(questions[index])
logger.debug(answers[index])
question = questions[index]

basic_rag.run({"query_embedder":{"text":question}, "chat_prompt_builder":{"question": question}})

"""
## 3. Deciding on Metrics

* **Semantic Answer Similarity**: SASEvaluator compares the embedding of a generated answer against a ground-truth answer based on a common embedding model.
* **ContextRelevanceEvaluator** will assess the relevancy of the retrieved context to answer the query question
* **FaithfulnessEvaluator** evaluates whether the generated answer can be derived from the context

## 4. Building an Evaluation Pipeline
"""
logger.info("## 3. Deciding on Metrics")


eval_pipeline = Pipeline()
eval_pipeline.add_component("context_relevance", ContextRelevanceEvaluator(raise_on_failure=False))
eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator(raise_on_failure=False))
eval_pipeline.add_component("sas", SASEvaluator(model=embedding_model))

"""
## 5. Running Evaluation

### Run the RAG Pipeline
"""
logger.info("## 5. Running Evaluation")

predicted_answers = []
retrieved_context = []

for question in questions: # loops over 15 questions
    result = basic_rag.run(
        {"query_embedder":{"text":question}, "chat_prompt_builder":{"question": question}}, include_outputs_from={"retriever"}
    )
    predicted_answers.append(result["chat_generator"]["replies"][0].text)
    retrieved_context.append(result["retriever"]["documents"])

"""
### Run the Evaluation
"""
logger.info("### Run the Evaluation")

eval_pipeline_results = eval_pipeline.run(
    {
        "context_relevance": {"questions": questions, "contexts": retrieved_context},
        "faithfulness": {"questions": questions, "contexts": retrieved_context, "predicted_answers": predicted_answers},
        "sas": {"predicted_answers": predicted_answers, "ground_truth_answers": answers},
    }
)

results = {
    "context_relevance": eval_pipeline_results['context_relevance'],
    "faithfulness": eval_pipeline_results['faithfulness'],
    "sas": eval_pipeline_results['sas']
}

"""
## 6. Analyzing Results

[EvaluationRunResult](https://docs.haystack.deepset.ai/reference/evaluation-api#evaluationrunresult)
"""
logger.info("## 6. Analyzing Results")


inputs = {
    'questions': questions,
    'contexts': retrieved_context,
    'true_answers': answers,
    'predicted_answers': predicted_answers
}
run_name="rag_eval"
eval_results = EvaluationRunResult(run_name=run_name, inputs=inputs, results=results)
eval_results.aggregated_report()

index = 2
logger.debug(eval_pipeline_results['context_relevance']["individual_scores"][index], "\nQuestion:", questions[index],"\nTrue Answer:", answers[index], "\nAnswer:", predicted_answers[index])
logger.debug("".join([doc.content for doc in retrieved_context[index]]))

"""
## Evaluation Frameworks

* For [RagasEvaluator](https://docs.haystack.deepset.ai/docs/ragasevaluator) check out our cookbook [RAG Pipeline Evaluation Using RAGAS](https://haystack.deepset.ai/cookbook/rag_eval_ragas)
* Here we will show how to use [FlowJudge](https://haystack.deepset.ai/integrations/flow-judge)
"""
logger.info("## Evaluation Frameworks")


model = Hf(flash_attn=False)

flow_judge_evaluator = HaystackFlowJudge(
    metric=RESPONSE_FAITHFULNESS_5POINT,
    model=model,
    progress_bar=True,
    raise_on_failure=True,
    save_results=True,
    fail_on_parse_error=False
)

# from getpass import getpass

# if not os.getenv("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass('OPENAI_API_KEY: ')

str_fj_retrieved_context = []
for context in retrieved_context:
    str_context = [doc.content for doc in context]
    str_fj_retrieved_context.append(" ".join(str_context)) # ["", "", ...]


integration_eval_pipeline = Pipeline()
integration_eval_pipeline.add_component("flow_judge_evaluator", flow_judge_evaluator)

eval_framework_pipeline_results = integration_eval_pipeline.run(
    {
        "flow_judge_evaluator": {"query": questions, "context": str_fj_retrieved_context, "response": predicted_answers},
    }
)

eval_framework_pipeline_results

logger.info("\n\n[DONE]", bright=True)