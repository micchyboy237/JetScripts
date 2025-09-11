from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
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
# Analyze a single long document

The AnalyzeDocumentChain takes in a single document, splits it up, and then runs it through a CombineDocumentsChain.
"""
logger.info("# Analyze a single long document")

with open("../docs/docs/modules/state_of_the_union.txt") as f:
    state_of_the_union = f.read()


llm = ChatOllama(model="llama3.2")


qa_chain = load_qa_chain(llm, chain_type="map_reduce")

qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

qa_document_chain.run(
    input_document=state_of_the_union,
    question="what did the president say about justice breyer?",
)

logger.info("\n\n[DONE]", bright=True)