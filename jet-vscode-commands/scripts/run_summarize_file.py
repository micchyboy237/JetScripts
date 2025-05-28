import sys
import json
from typing import Optional
from pydantic import ValidationError

from llama_index.core import PromptTemplate
from llama_index.core.types import BaseModel
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.types import PydanticProgramMode
from jet.llm.ollama.base import Ollama

from jet.vectors.rag import SettingsManager
from jet.validation import validate_json
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from jet.transformers.object import make_serializable
initialize_ollama_settings()
llm = Ollama(
    temperature=0,
    context_window=4096,
    request_timeout=300.0,
    model="mistral",
)


# Load Data
target_file_path = sys.argv[1]

# documents = SimpleDirectoryReader(
#     input_files=[target_file_path]
# ).load_data()
# texts = [doc.text for doc in docs]

with open(target_file_path, "r") as f:
    text = f.read()
texts = [text]


settings_manager = SettingsManager.create()
settings_manager.llm = llm
settings_manager.pydantic_program_mode = PydanticProgramMode.LLM


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


try:
    # Create pydantic data model to structure response
    class FileSummary(BaseModel):
        features: list[str]
        use_cases: list[str]
        additional_info: Optional[str] = None

    summarizer = TreeSummarize(
        llm=settings_manager.llm,
        verbose=True,
        streaming=False,
        output_cls=FileSummary,
        summary_template=qa_prompt
    )

    question = 'Summarize the features and use cases of this code.'
    result = summarizer.get_response(question, texts)
except ValidationError as e:
    logger.error(json.dumps(make_serializable(e.errors()), indent=2))
    if e.errors() and type(e.errors()[0]["input"]) == dict:
        current_result_dict = e.errors()[0]["input"]
        if FileSummary.__name__ in current_result_dict:
            current_result_dict = current_result_dict[FileSummary.__name__]

        result = validate_json(current_result_dict,
                               FileSummary.model_json_schema())

# Inspect the response
#
# Here, we see the response is in an instance of our `FileSummary` class.

logger.success(json.dumps(make_serializable(result), indent=2))

logger.info("\n\n[DONE]")
