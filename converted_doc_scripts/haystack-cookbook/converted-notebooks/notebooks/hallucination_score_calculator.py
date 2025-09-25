from haystack import Document
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_experimental.components.generators.chat.openai import OpenAIChatGenerator
from haystack_experimental.utils.hallucination_risk_calculator.dataclasses import HallucinationScoreConfig
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

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Calculating a Hallucination Score with the OpenAIChatGenerator

In this cookbook we will show how to calculate a hallucination risk based on the research paper [LLMs are Bayesian, in Expectation, not in Realization](https://arxiv.org/abs/2507.11768) and this GitHub repo, https://github.com/leochlon/hallbayes.

In this notebook, we'll use the OpenAIChatGenerator from haystack-experimental.

## Setup Environment
"""
logger.info("# Calculating a Hallucination Score with the OpenAIChatGenerator")

# %pip install haystack-experimental -q

"""
Set up Ollama API Key
"""
logger.info("Set up Ollama API Key")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter Ollama API key:")

"""
## Closed Book Example

Based on the example from the original GitHub repo [here](https://github.com/leochlon/hallbayes?tab=readme-ov-file#2-closed-book-no-evidence)
"""
logger.info("## Closed Book Example")



llm = OpenAIChatGenerator(model="llama3.2")

closed_book_result = llm.run(
    messages=[ChatMessage.from_user(text="Who won the 2019 Nobel Prize in Physics?")],
    hallucination_score_config=HallucinationScoreConfig(
        skeleton_policy="closed_book" # NOTE: We set "closed_book" here for closed-book hallucination risk calculation
    ),
)
logger.debug(f"Decision: {closed_book_result['replies'][0].meta['hallucination_decision']}")
logger.debug(f"Risk bound: {closed_book_result['replies'][0].meta['hallucination_risk']:.3f}")
logger.debug(f"Rationale: {closed_book_result['replies'][0].meta['hallucination_rationale']}")
logger.debug(f"Answer:\n{closed_book_result['replies'][0].text}")

"""
## Evidence-based Example
Based on the example from the original GitHub repo [here](https://github.com/leochlon/hallbayes?tab=readme-ov-file#1-evidence-based-when-you-have-context)
"""
logger.info("## Evidence-based Example")



llm = OpenAIChatGenerator(model="llama3.2")

rag_result = llm.run(
    messages=[
        ChatMessage.from_user(
            text="Task: Answer strictly based on the evidence provided below.\n"
            "Question: Who won the Nobel Prize in Physics in 2019?\n"
            "Evidence:\n"
            "- Nobel Prize press release (2019): James Peebles (1/2); Michel Mayor & Didier Queloz (1/2).\n"
            "Constraints: If evidence is insufficient or conflicting, refuse."
        )
    ],
    hallucination_score_config=HallucinationScoreConfig(
        skeleton_policy="evidence_erase"  # NOTE: We set "evidence_erase" here for evidence-based hallucination risk calculation
    ),
)
logger.debug(f"Decision: {rag_result['replies'][0].meta['hallucination_decision']}")
logger.debug(f"Risk bound: {rag_result['replies'][0].meta['hallucination_risk']:.3f}")
logger.debug(f"Rationale: {rag_result['replies'][0].meta['hallucination_rationale']}")
logger.debug(f"Answer:\n{rag_result['replies'][0].text}")

"""
## RAG-based Example

Create a Document Store and index some documents
"""
logger.info("## RAG-based Example")


document_store = InMemoryDocumentStore()

docs = [
    Document(content="Nobel Prize press release (2019): James Peebles (1/2); Michel Mayor & Didier Queloz (1/2)"),
    Document(content="Nikola Tesla was a Serbian-American engineer, futurist, and inventor. He is known for his contributions to the design of the modern alternating current (AC) electricity supply system.")
]
document_store.write_documents(docs)

"""
Create a RAG Question Answering pipeline
"""
logger.info("Create a RAG Question Answering pipeline")



pipe = Pipeline()

user_template = """Task: Answer strictly based on the evidence provided below.
Question: {{query}}
Evidence:
{%- for document in documents %}
- {{document.content}}
{%- endfor -%}
Constraints: If evidence is insufficient or conflicting, refuse.
"""
pipe.add_component("retriever", InMemoryBM25Retriever(document_store))
pipe.add_component(
    "prompt_builder",
    ChatPromptBuilder(template=[ChatMessage.from_user(user_template)], required_variables="*")
)
pipe.add_component("llm", OpenAIChatGenerator(model="llama3.2"))

pipe.connect("retriever.documents", "prompt_builder.documents")
pipe.connect("prompt_builder.prompt", "llm.messages")

"""
Run a query that is answerable based on the evidence
"""
logger.info("Run a query that is answerable based on the evidence")

query = "Who won the Nobel Prize in Physics in 2019?"

result = pipe.run(
    data={
        "retriever": {"query": query},
        "prompt_builder": {"query": query},
        "llm": {
            "hallucination_score_config": HallucinationScoreConfig(skeleton_policy="evidence_erase")
        }
    }
)

logger.debug(f"Decision: {result['llm']['replies'][0].meta['hallucination_decision']}")
logger.debug(f"Risk bound: {result['llm']['replies'][0].meta['hallucination_risk']:.3f}")
logger.debug(f"Rationale: {result['llm']['replies'][0].meta['hallucination_rationale']}")
logger.debug(f"Answer:\n{result['llm']['replies'][0].text}")

"""
Run a query that should not be answered
"""
logger.info("Run a query that should not be answered")

query = "Who won the Nobel Prize in Physics in 2022?"

result = pipe.run(
    data={
        "retriever": {"query": query},
        "prompt_builder": {"query": query},
        "llm": {
            "hallucination_score_config": HallucinationScoreConfig(skeleton_policy="evidence_erase")
        }
    }
)

logger.debug(f"Decision: {result['llm']['replies'][0].meta['hallucination_decision']}")
logger.debug(f"Risk bound: {result['llm']['replies'][0].meta['hallucination_risk']:.3f}")
logger.debug(f"Rationale: {result['llm']['replies'][0].meta['hallucination_rationale']}")
logger.debug(f"Answer:\n{result['llm']['replies'][0].text}")

logger.info("\n\n[DONE]", bright=True)