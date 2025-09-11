from jet.logger import logger
from langchain_community.embeddings.laser import LaserEmbeddings
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
# LASER Language-Agnostic SEntence Representations Embeddings by Meta AI

>[LASER](https://github.com/facebookresearch/LASER/) is a Python library developed by the Meta AI Research team and used for creating multilingual sentence embeddings for over 147 languages as of 2/25/2024 
>- List of supported languages at https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

## Dependencies

To use LaserEmbed with LangChain, install the `laser_encoders` Python package.
"""
logger.info("# LASER Language-Agnostic SEntence Representations Embeddings by Meta AI")

# %pip install laser_encoders

"""
## Imports
"""
logger.info("## Imports")


"""
## Instantiating Laser
   
### Parameters
- `lang: Optional[str]`
    >If empty will default
    to using a multilingual LASER encoder model (called "laser2").
    You can find the list of supported languages and lang_codes [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)
    and [here](https://github.com/facebookresearch/LASER/blob/main/laser_encoders/language_list.py)
.
"""
logger.info("## Instantiating Laser")

embeddings = LaserEmbeddings(lang="eng_Latn")

"""
## Usage

### Generating document embeddings
"""
logger.info("## Usage")

document_embeddings = embeddings.embed_documents(
    ["This is a sentence", "This is some other sentence"]
)

"""
### Generating query embeddings
"""
logger.info("### Generating query embeddings")

query_embeddings = embeddings.embed_query("This is a query")

logger.info("\n\n[DONE]", bright=True)