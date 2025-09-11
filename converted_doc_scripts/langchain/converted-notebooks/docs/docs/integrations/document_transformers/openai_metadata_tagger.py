from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.document_transformers.ollama_functions import (
create_metadata_tagger,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
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
# Ollama metadata tagger

It can often be useful to tag ingested documents with structured metadata, such as the title, tone, or length of a document, to allow for a more targeted similarity search later. However, for large numbers of documents, performing this labelling process manually can be tedious.

The `OllamaMetadataTagger` document transformer automates this process by extracting metadata from each provided document according to a provided schema. It uses a configurable `Ollama Functions`-powered chain under the hood, so if you pass a custom LLM instance, it must be an `Ollama` model with functions support. 

**Note:** This document transformer works best with complete documents, so it's best to run it first with whole documents before doing any other splitting or processing!

For example, let's say you wanted to index a set of movie reviews. You could initialize the document transformer with a valid `JSON Schema` object as follows:
"""
logger.info("# Ollama metadata tagger")


schema = {
    "properties": {
        "movie_title": {"type": "string"},
        "critic": {"type": "string"},
        "tone": {"type": "string", "enum": ["positive", "negative"]},
        "rating": {
            "type": "integer",
            "description": "The number of stars the critic rated the movie",
        },
    },
    "required": ["movie_title", "critic", "tone"],
}

llm = ChatOllama(model="llama3.2")

document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)

"""
You can then simply pass the document transformer a list of documents, and it will extract metadata from the contents:
"""
logger.info("You can then simply pass the document transformer a list of documents, and it will extract metadata from the contents:")

original_documents = [
    Document(
        page_content="Review of The Bee Movie\nBy Roger Ebert\n\nThis is the greatest movie ever made. 4 out of 5 stars."
    ),
    Document(
        page_content="Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.",
        metadata={"reliable": False},
    ),
]

enhanced_documents = document_transformer.transform_documents(original_documents)


logger.debug(
    *[d.page_content + "\n\n" + json.dumps(d.metadata) for d in enhanced_documents],
    sep="\n\n---------------\n\n",
)

"""
The new documents can then be further processed by a text splitter before being loaded into a vector store. Extracted fields will not overwrite existing metadata.

You can also initialize the document transformer with a Pydantic schema:
"""
logger.info("The new documents can then be further processed by a text splitter before being loaded into a vector store. Extracted fields will not overwrite existing metadata.")




class Properties(BaseModel):
    movie_title: str
    critic: str
    tone: Literal["positive", "negative"]
    rating: int = Field(description="Rating out of 5 stars")


document_transformer = create_metadata_tagger(Properties, llm)
enhanced_documents = document_transformer.transform_documents(original_documents)

logger.debug(
    *[d.page_content + "\n\n" + json.dumps(d.metadata) for d in enhanced_documents],
    sep="\n\n---------------\n\n",
)

"""
## Customization

You can pass the underlying tagging chain the standard LLMChain arguments in the document transformer constructor. For example, if you wanted to ask the LLM to focus specific details in the input documents, or extract metadata in a certain style, you could pass in a custom prompt:
"""
logger.info("## Customization")


prompt = ChatPromptTemplate.from_template(
    """Extract relevant information from the following text.
Anonymous critics are actually Roger Ebert.

{input}
"""
)

document_transformer = create_metadata_tagger(schema, llm, prompt=prompt)
enhanced_documents = document_transformer.transform_documents(original_documents)

logger.debug(
    *[d.page_content + "\n\n" + json.dumps(d.metadata) for d in enhanced_documents],
    sep="\n\n---------------\n\n",
)

logger.info("\n\n[DONE]", bright=True)