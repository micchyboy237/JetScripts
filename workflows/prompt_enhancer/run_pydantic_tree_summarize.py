from jet.llm.ollama.constants import OLLAMA_LARGE_CHUNK_OVERLAP, OLLAMA_LARGE_CHUNK_SIZE, OLLAMA_LARGE_EMBED_MODEL, OLLAMA_LARGE_LLM_MODEL, OLLAMA_SMALL_EMBED_MODEL
from jet.llm.query.retrievers import setup_index
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.transformers.formatters import format_json
from jet.utils.class_utils import class_to_string
from jet.vectors.metadata import parse_nodes
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
from jet.llm.ollama import initialize_ollama_settings
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


# Download Data


# Load Data

DATA_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
query = """Generate real world diverse questions and answers that an employer can have for a job interview based on provided context and schema.
Example response format:
{
    "data": [
        {
            "question": "Question 1",
            "answer": "Answer 1"
        }
    ]
}
""".strip()

chunk_size = 1024
chunk_overlap = 200

data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
logger.newline()
logger.info("Loading data...")
docs = SimpleDirectoryReader(data_dir).load_data(show_progress=True)
logger.log("All docs:", len(docs), colors=["DEBUG", "SUCCESS"])
base_nodes = parse_nodes(docs, chunk_size=chunk_size,
                         chunk_overlap=chunk_overlap)
texts = "\n\n".join([doc.text for doc in base_nodes])
logger.log("Parsed nodes:", len(base_nodes),
           colors=["DEBUG", "SUCCESS"])


# Create pydantic model to structure response


class Data(BaseModel):
    question: str = Field(
        description="Short question text answering context information provided.")
    answer: str | list[str] = Field(
        description="The concise answer or list of answers to the question given the relevant context.")


class QuestionAnswer(BaseModel):
    data: list[Data]


INSTRUCTIONS_PROMPT = f"""
Return only the generated JSON value without any explanations surrounded by ```json that adheres to the model below:
{class_to_string(QuestionAnswer)}
""".strip()


QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    # "Please also write the answer as JSON that adheres to the schema.\n"
    "{instructions_str}\n"
    "Query: {query_str}\n"
    "Response:\n"
)
qa_prompt = PromptTemplate(QA_PROMPT_TMPL)

response = llm_settings.llm.structured_predict(
    QuestionAnswer,
    PromptTemplate(QA_PROMPT_TMPL),
    context_str=texts,
    instructions_str=INSTRUCTIONS_PROMPT,
    query_str=query,
    llm_kwargs={
        "options": {
            "temperature": 0,
        },
    },

)


# Inspect the response
#
# Here, we see the response is in an instance of our `QuestionAnswer` class.

logger.newline()
logger.info("RESPONSE:")
logger.success(format_json(response))


logger.info("\n\n[DONE]", bright=True)
