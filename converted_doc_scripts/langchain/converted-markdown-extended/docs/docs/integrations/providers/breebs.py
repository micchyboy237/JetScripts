from jet.logger import logger
from langchain.retrievers import BreebsRetriever
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
# Breebs (Open Knowledge)

>[Breebs](https://www.breebs.com/) is an open collaborative knowledge platform.
>Anybody can create a `Breeb`, a knowledge capsule based on PDFs stored on a Google Drive folder.
>A `Breeb` can be used by any LLM/chatbot to improve its expertise, reduce hallucinations and give access to sources.
>Behind the scenes, `Breebs` implements several `Retrieval Augmented Generation (RAG)` models
> to seamlessly provide useful context at each iteration.

## Retriever
"""
logger.info("# Breebs (Open Knowledge)")


"""
[See a usage example (Retrieval & ConversationalRetrievalChain)](/docs/integrations/retrievers/breebs)
"""

logger.info("\n\n[DONE]", bright=True)