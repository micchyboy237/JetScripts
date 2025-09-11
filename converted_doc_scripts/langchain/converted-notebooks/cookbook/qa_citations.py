from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import create_citation_fuzzy_match_chain
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
# Citing retrieval sources

This notebook shows how to use Ollama functions ability to extract citations from text.
"""
logger.info("# Citing retrieval sources")


question = "What did the author do during college?"
context = """
My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.
I went to an arts highschool but in university I studied Computational Mathematics and physics.
As part of coop I worked at many companies including Stitchfix, Facebook.
I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.
"""

llm = ChatOllama(model="llama3.2")

chain = create_citation_fuzzy_match_chain(llm)

result = chain.run(question=question, context=context)

logger.debug(result)

def highlight(text, span):
    return (
        "..."
        + text[span[0] - 20 : span[0]]
        + "*"
        + "\033[91m"
        + text[span[0] : span[1]]
        + "\033[0m"
        + "*"
        + text[span[1] : span[1] + 20]
        + "..."
    )

for fact in result.answer:
    logger.debug("Statement:", fact.fact)
    for span in fact.get_spans(context):
        logger.debug("Citation:", highlight(context, span))
    logger.debug()

logger.info("\n\n[DONE]", bright=True)