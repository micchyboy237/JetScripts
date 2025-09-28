from haystack import Document
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.extractors.llm_metadata_extractor import LLMMetadataExtractor
# from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from jet.adapters.haystack.ollama_chat_generator import OllamaChatGenerator
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
# LLMMetaDataExtractor: seamless metadata extraction from documents with just a prompt

Notebook by [David S. Batista](https://www.davidsbatista.net/)

This notebook shows how to use [`LLMMetadataExtractor`](https://docs.haystack.deepset.ai/docs/llmmetadataextractor), we will use a arge Language Model to perform metadata extraction from a Document.

## Setting Up
"""
logger.info("# LLMMetaDataExtractor: seamless metadata extraction from documents with just a prompt")

# !uv pip install haystack-ai
# !uv pip install "sentence-transformers>=3.0.0"

"""
## Initialize LLMMetadataExtractor

Let's define what kind of metadata we want to extract from our documents, we wil do it through a LLM prompt, which will then be used by the `LLMMetadataExtractor` component. In this case we want to extract named-entities from our documents.
"""
logger.info("## Initialize LLMMetadataExtractor")

NER_PROMPT = """
    -Goal-
    Given text and a list of entity types, identify all entities of those types from the text.

    -Steps-
    1. Identify all entities. For each identified entity, extract the following information:
    - entity: Name of the entity
    - entity_type: One of the following types: [organization, product, service, industry]
    Format each entity as a JSON like: {"entity": <entity_name>, "entity_type": <entity_type>}

    2. Return output in a single list with all the entities identified in steps 1.

    -Examples-
    Example 1:
    entity_types: [organization, person, partnership, financial metric, product, service, industry, investment strategy, market trend]
    text: Another area of strength is our co-brand issuance. Visa is the primary network partner for eight of the top
    10 co-brand partnerships in the US today and we are pleased that Visa has finalized a multi-year extension of
    our successful credit co-branded partnership with Alaska Airlines, a portfolio that benefits from a loyal customer
    base and high cross-border usage.
    We have also had significant co-brand momentum in CEMEA. First, we launched a new co-brand card in partnership
    with Qatar Airways, British Airways and the National Bank of Kuwait. Second, we expanded our strong global
    Marriott relationship to launch Qatar's first hospitality co-branded card with Qatar Islamic Bank. Across the
    United Arab Emirates, we now have exclusive agreements with all the leading airlines marked by a recent
    agreement with Emirates Skywards.
    And we also signed an inaugural Airline co-brand agreement in Morocco with Royal Air Maroc. Now newer digital
    issuers are equally
    ------------------------
    output:
    {"entities": [{"entity": "Visa", "entity_type": "company"}, {"entity": "Alaska Airlines", "entity_type": "company"}, {"entity": "Qatar Airways", "entity_type": "company"}, {"entity": "British Airways", "entity_type": "company"}, {"entity": "National Bank of Kuwait", "entity_type": "company"}, {"entity": "Marriott", "entity_type": "company"}, {"entity": "Qatar Islamic Bank", "entity_type": "company"}, {"entity": "Emirates Skywards", "entity_type": "company"}, {"entity": "Royal Air Maroc", "entity_type": "company"}]}
    -Real Data-
    entity_types: [company, organization, person, country, product, service]
    text: {{ document.content }}
    output:
    """

"""
Let's initialise an instance of the `LLMMetadataExtractor` using Ollama as the LLM provider and the prompt defined above to perform metadata extraction
"""
logger.info("Let's initialise an instance of the `LLMMetadataExtractor` using Ollama as the LLM provider and the prompt defined above to perform metadata extraction")


"""
# We will also need to set the OPENAI_API_KEY
"""
# logger.info("We will also need to set the OPENAI_API_KEY")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#   os.environ["OPENAI_API_KEY"] = getpass("Enter Ollama API key:")

"""
We will instatiate a `LLMMetadataExtractor` instance using the Ollama as LLM provider. Notice that the parameter `prompt` is set to the prompt we defined above, and that we also need to set which keys should be present in the JSON ouput, in this case "entities".

Another important aspect is the `raise_on_failure=False`, if for some document the LLM fails (e.g.: network error, or doesn't return a valid JSON object) we continue the processing of all the documents in the input.
"""
logger.info("We will instatiate a `LLMMetadataExtractor` instance using the Ollama as LLM provider. Notice that the parameter `prompt` is set to the prompt we defined above, and that we also need to set which keys should be present in the JSON ouput, in this case \"entities\".")


chat_generator = OllamaChatGenerator(
    model="qwen3:4b-q4_K_M",
    generation_kwargs={
        "max_tokens": 500,
        "temperature": 0.0,
        "seed": 0,
        "response_format": {"type": "json_object"},
    },
    # max_retries=1,
    timeout=60.0,
)

metadata_extractor = LLMMetadataExtractor(
    prompt=NER_PROMPT,
    chat_generator=chat_generator,
    expected_keys=["entities"],
    raise_on_failure=False,
)

"""
### Let's define documents from which the component will extract metadata, i.e.: named-entities
"""
logger.info("### Let's define documents from which the component will extract metadata, i.e.: named-entities")


docs = [
    Document(content="deepset was founded in 2018 in Berlin, and is known for its Haystack framework"),
    Document(content="Hugging Face is a company that was founded in New York, USA and is known for its Transformers library"),
    Document(content="Google was founded in 1998 by Larry Page and Sergey Brin"),
    Document(content="Peugeot is a French automotive manufacturer that was founded in 1810 by Jean-Pierre Peugeot"),
    Document(content="Siemens is a German multinational conglomerate company headquartered in Munich and Berlin, founded in 1847 by Werner von Siemens")
]

"""
and let's extract :)
"""
logger.info("and let's extract :)")

result = metadata_extractor.run(documents=docs)

result

"""
## Indexing Pipeline with Extraction

Let's now build an indexing pipeline, where we simply give the Documents as input and get a Document Store with the documents indexed with metadata
"""
logger.info("## Indexing Pipeline with Extraction")


doc_store = InMemoryDocumentStore()

p = Pipeline()
p.add_component(instance=metadata_extractor, name="metadata_extractor")
p.add_component(instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="embedder")
p.add_component(instance=DocumentWriter(document_store=doc_store), name="writer")
p.connect("metadata_extractor.documents", "embedder.documents")
p.connect("embedder.documents", "writer.documents")

"""
## Try it Out!
"""
logger.info("## Try it Out!")

p.run(data={"metadata_extractor": {"documents": docs}})

"""
Let's inspect the documents metadata in the document store
"""
logger.info("Let's inspect the documents metadata in the document store")

for doc in doc_store.storage.values():
    logger.debug(doc.content)
    logger.debug(doc.meta)
    logger.debug("\n---------")

logger.info("\n\n[DONE]", bright=True)