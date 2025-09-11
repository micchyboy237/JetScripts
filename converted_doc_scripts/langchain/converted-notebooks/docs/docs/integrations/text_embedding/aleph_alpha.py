from jet.logger import logger
from langchain_community.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain_community.embeddings import AlephAlphaSymmetricSemanticEmbedding
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
# Aleph Alpha

There are two possible ways to use Aleph Alpha's semantic embeddings. If you have texts with a dissimilar structure (e.g. a Document and a Query) you would want to use asymmetric embeddings. Conversely, for texts with comparable structures, symmetric embeddings are the suggested approach.

## Asymmetric
"""
logger.info("# Aleph Alpha")


document = "This is a content of the document"
query = "What is the content of the document?"

embeddings = AlephAlphaAsymmetricSemanticEmbedding(normalize=True, compress_to_size=128)

doc_result = embeddings.embed_documents([document])

query_result = embeddings.embed_query(query)

"""
## Symmetric
"""
logger.info("## Symmetric")


text = "This is a test text"

embeddings = AlephAlphaSymmetricSemanticEmbedding(normalize=True, compress_to_size=128)

doc_result = embeddings.embed_documents([text])

query_result = embeddings.embed_query(text)

logger.info("\n\n[DONE]", bright=True)