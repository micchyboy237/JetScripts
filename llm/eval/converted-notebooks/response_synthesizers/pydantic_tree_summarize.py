from llama_index.core import PromptTemplate
from typing import Optional
from typing import List
from llama_index.core.types import BaseModel
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import SimpleDirectoryReader
from llama_index.core.types import PydanticProgramMode
from jet.vectors.rag import SettingsManager
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/response_synthesizers/pydantic_tree_summarize.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pydantic Tree Summarize
#
# In this notebook, we demonstrate how to use tree summarize with structured outputs. Specifically, tree summarize is used to output pydantic objects.


settings_manager = SettingsManager.create()
settings_manager.pydantic_program_mode = PydanticProgramMode.LLM

# Download Data


# Load Data


reader = SimpleDirectoryReader(
    input_files=[
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/scraped_urls/www_imdb_com_title_tt32812118.md"]
)

docs = reader.load_data()

texts = [doc.text for doc in docs]
texts[0]

# Summarize


# Create pydantic model to structure response


class AnimeDetails(BaseModel):
    seasons: int
    episodes: int
    additional_info: Optional[str] = None


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
    llm=settings_manager.llm,
    verbose=True,
    streaming=False,
    output_cls=AnimeDetails,
    summary_template=qa_prompt
)

question = 'How many seasons and episodes does "I’ll Become a Villainess Who Goes Down in History" anime have?'
response = summarizer.get_response(question, texts)

# Inspect the response
#
# Here, we see the response is in an instance of our `AnimeDetails` class.

logger.success(response)

logger.success(response.seasons)

logger.success(response.episodes)

logger.success(response.additional_info)

logger.info("\n\n[DONE]", bright=True)
