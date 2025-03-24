from jet.llm.ollama.constants import OLLAMA_LARGE_CHUNK_OVERLAP, OLLAMA_LARGE_CHUNK_SIZE, OLLAMA_LARGE_EMBED_MODEL, OLLAMA_LARGE_LLM_MODEL, OLLAMA_SMALL_EMBED_MODEL
from jet.llm.query.retrievers import setup_index
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.transformers.formatters import format_json
from llama_index.core import PromptTemplate
from typing import Optional
from typing import List
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import Document
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import SimpleDirectoryReader
from llama_index.core.types import PydanticProgramMode
from jet.vectors import SettingsManager
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from pydantic import BaseModel, Field

chunk_size = OLLAMA_LARGE_CHUNK_SIZE
chunk_overlap = OLLAMA_LARGE_CHUNK_OVERLAP
llm_settings = initialize_ollama_settings({
    "llm_model": OLLAMA_LARGE_LLM_MODEL,
    "embedding_model": OLLAMA_LARGE_EMBED_MODEL,
    "chunk_size": chunk_size,
    "chunk_overlap": chunk_overlap,
})

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/response_synthesizers/pydantic_tree_summarize.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pydantic Tree Summarize
#
# In this notebook, we demonstrate how to use tree summarize with structured outputs. Specifically, tree summarize is used to output pydantic objects.


llm_settings.pydantic_program_mode = PydanticProgramMode.LLM

# Download Data


# Load Data

DATA_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
question = "Tell me about yourself."

documents = SimpleDirectoryReader(DATA_PATH).load_data()
texts = [doc.text for doc in documents]

# query_nodes = setup_index(
#     documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
# result = query_nodes(
#     question, FUSION_MODES.RELATIVE_SCORE, score_threshold=0.2)
# result_nodes = result["nodes"]
# result_texts = result["texts"]
# display_jet_source_nodes(question, result_nodes)
# texts = result_texts
logger.log("Result texts:", len(texts),
           colors=["DEBUG", "SUCCESS"])


question = """
Provide the data samples given the context.
Example response format:
{
    "data": [
        {
            "question": "Question 1",
            "reference": "Context 1",
            "answer": "Answer 1"
        }
    ]
}
""".strip()


# Create pydantic model to structure response


class Data(BaseModel):
    question: str = Field(
        ..., description="Short question text answering partial context information provided.")
    reference: str = Field(...,
                           description="The partial sentences or paragraph from context used by the question as reference.")
    answer: str = Field(
        ..., description="The concise answer to the question given the relevant partial context.")


class QuestionReferenceAnswer(BaseModel):
    data: list[Data]


qa_prompt_tmpl = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Please also write the answer as JSON that adheres to the schema.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(qa_prompt_tmpl)

refine_prompt_tmpl = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "Please also write the answer as JSON that adheres to the schema.\n"
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
refine_prompt = PromptTemplate(refine_prompt_tmpl)

summarizer = TreeSummarize(
    llm=llm_settings.llm,
    verbose=True,
    streaming=False,
    output_cls=QuestionReferenceAnswer,
    summary_template=qa_prompt
)


response = summarizer.get_response(question, texts)

# Inspect the response
#
# Here, we see the response is in an instance of our `QuestionReferenceAnswer` class.

logger.newline()
logger.info("RESPONSE:")
logger.success(format_json(response))


logger.info("\n\n[DONE]", bright=True)
