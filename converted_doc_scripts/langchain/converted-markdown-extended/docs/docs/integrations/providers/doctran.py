from jet.logger import logger
from langchain_community.document_transformers import DoctranPropertyExtractor
from langchain_community.document_transformers import DoctranQATransformer
from langchain_community.document_transformers import DoctranTextTranslator
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
# Doctran

>[Doctran](https://github.com/psychic-api/doctran) is a python package. It uses LLMs and open-source
> NLP libraries to transform raw text into clean, structured, information-dense documents
> that are optimized for vector space retrieval. You can think of `Doctran` as a black box where
> messy strings go in and nice, clean, labelled strings come out.


## Installation and Setup
"""
logger.info("# Doctran")

pip install doctran

"""
## Document Transformers

### Document Interrogator

See a [usage example for DoctranQATransformer](/docs/integrations/document_transformers/doctran_interrogate_document).
"""
logger.info("## Document Transformers")


"""
### Property Extractor

See a [usage example for DoctranPropertyExtractor](/docs/integrations/document_transformers/doctran_extract_properties).
"""
logger.info("### Property Extractor")


"""
### Document Translator

See a [usage example for DoctranTextTranslator](/docs/integrations/document_transformers/doctran_translate_document).
"""
logger.info("### Document Translator")


logger.info("\n\n[DONE]", bright=True)