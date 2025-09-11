from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain_experimental.pal_chain import PALChain
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
# Program-aided language model (PAL) chain

Implements Program-Aided Language Models, as in https://arxiv.org/pdf/2211.10435.pdf.
"""
logger.info("# Program-aided language model (PAL) chain")


llm = Ollama(temperature=0, max_tokens=512)

"""
## Math Prompt
"""
logger.info("## Math Prompt")

pal_chain = PALChain.from_math_prompt(llm, verbose=True)

question = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"

pal_chain.run(question)

"""
## Colored Objects
"""
logger.info("## Colored Objects")

pal_chain = PALChain.from_colored_object_prompt(llm, verbose=True)

question = "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses. If I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"

pal_chain.run(question)

"""
## Intermediate Steps
You can also use the intermediate steps flag to return the code executed that generates the answer.
"""
logger.info("## Intermediate Steps")

pal_chain = PALChain.from_colored_object_prompt(
    llm, verbose=True, return_intermediate_steps=True
)

question = "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses. If I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"

result = pal_chain({"question": question})

result["intermediate_steps"]

logger.info("\n\n[DONE]", bright=True)