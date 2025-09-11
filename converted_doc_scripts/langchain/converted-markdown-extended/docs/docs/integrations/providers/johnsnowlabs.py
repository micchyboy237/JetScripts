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
# Johnsnowlabs

Gain access to the [johnsnowlabs](https://www.johnsnowlabs.com/) ecosystem of enterprise NLP libraries
with over 21.000 enterprise NLP models in over 200 languages with the open source `johnsnowlabs` library.
For all 24.000+ models, see the [John Snow Labs Model Models Hub](https://nlp.johnsnowlabs.com/models)

## Installation and Setup
"""
logger.info("# Johnsnowlabs")

pip install johnsnowlabs

"""
To [install enterprise features](https://nlp.johnsnowlabs.com/docs/en/jsl/install_licensed_quick, run:
"""
logger.info("To [install enterprise features](https://nlp.johnsnowlabs.com/docs/en/jsl/install_licensed_quick, run:")

nlp.install()

"""
You can embed your queries and documents with either `gpu`,`cpu`,`apple_silicon`,`aarch` based optimized binaries.
By default cpu binaries are used.
Once a session is started, you must restart your notebook to switch between GPU or CPU, or changes will not take effect.

## Embed Query with CPU:
"""
logger.info("## Embed Query with CPU:")

document = "foo bar"
embedding = JohnSnowLabsEmbeddings('embed_sentence.bert')
output = embedding.embed_query(document)

"""
## Embed Query with GPU:
"""
logger.info("## Embed Query with GPU:")

document = "foo bar"
embedding = JohnSnowLabsEmbeddings('embed_sentence.bert','gpu')
output = embedding.embed_query(document)

"""
## Embed Query with Apple Silicon (M1,M2,etc..):
"""
logger.info("## Embed Query with Apple Silicon (M1,M2,etc..):")

documents = ["foo bar", 'bar foo']
embedding = JohnSnowLabsEmbeddings('embed_sentence.bert','apple_silicon')
output = embedding.embed_query(document)

"""
## Embed Query with AARCH:
"""
logger.info("## Embed Query with AARCH:")

documents = ["foo bar", 'bar foo']
embedding = JohnSnowLabsEmbeddings('embed_sentence.bert','aarch')
output = embedding.embed_query(document)

"""
## Embed Document with CPU:
"""
logger.info("## Embed Document with CPU:")

documents = ["foo bar", 'bar foo']
embedding = JohnSnowLabsEmbeddings('embed_sentence.bert','gpu')
output = embedding.embed_documents(documents)

"""
## Embed Document with GPU:
"""
logger.info("## Embed Document with GPU:")

documents = ["foo bar", 'bar foo']
embedding = JohnSnowLabsEmbeddings('embed_sentence.bert','gpu')
output = embedding.embed_documents(documents)

"""
## Embed Document with Apple Silicon (M1,M2,etc..):

documents = ["foo bar", 'bar foo']
embedding = JohnSnowLabsEmbeddings('embed_sentence.bert','apple_silicon')
output = embedding.embed_documents(documents)

## Embed Document with AARCH:


"""
logger.info("## Embed Document with Apple Silicon (M1,M2,etc..):")

documents = ["foo bar", 'bar foo']
embedding = JohnSnowLabsEmbeddings('embed_sentence.bert','aarch')
output = embedding.embed_documents(documents)

"""
Models are loaded with [nlp.load](https://nlp.johnsnowlabs.com/docs/en/jsl/load_api) and spark session is started with [nlp.start()](https://nlp.johnsnowlabs.com/docs/en/jsl/start-a-sparksession) under the hood.
"""
logger.info("Models are loaded with [nlp.load](https://nlp.johnsnowlabs.com/docs/en/jsl/load_api) and spark session is started with [nlp.start()](https://nlp.johnsnowlabs.com/docs/en/jsl/start-a-sparksession) under the hood.")

logger.info("\n\n[DONE]", bright=True)