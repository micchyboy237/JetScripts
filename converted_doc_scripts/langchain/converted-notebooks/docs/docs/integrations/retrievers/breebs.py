from jet.logger import logger
from langchain_community.retrievers import BreebsRetriever
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
# BREEBS (Open Knowledge)

>[BREEBS](https://www.breebs.com/) is an open collaborative knowledge platform. 
Anybody can create a Breeb, a knowledge capsule, based on PDFs stored on a Google Drive folder.
A breeb can be used by any LLM/chatbot to improve its expertise, reduce hallucinations and give access to sources.
Behind the scenes, Breebs implements several Retrieval Augmented Generation (RAG) models to seamlessly provide useful context at each iteration.  

## List of available Breebs

To get the full list of Breebs, including their key (breeb_key) and description : 
https://breebs.promptbreeders.com/web/listbreebs.  
Dozens of Breebs have already been created by the community and are freely available for use. They cover a wide range of expertise, from organic chemistry to mythology, as well as tips on seduction and decentralized finance.


## Creating a new Breeb

To generate a new Breeb, simply compile PDF files in a publicly shared Google Drive folder and initiate the creation process on the [BREEBS website](https://www.breebs.com/) by clicking the "Create Breeb" button. You can currently include up to 120 files, with a total character limit of 15 million.  

## Retriever example
"""
logger.info("# BREEBS (Open Knowledge)")


breeb_key = "Parivoyage"
retriever = BreebsRetriever(breeb_key)
documents = retriever.invoke(
    "What are some unique, lesser-known spots to explore in Paris?"
)
logger.debug(documents)

logger.info("\n\n[DONE]", bright=True)