from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import LLMSummarizationCheckerChain
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
# Summarization checker chain
This notebook shows some examples of LLMSummarizationCheckerChain in use with different types of texts.  It has a few distinct differences from the `LLMCheckerChain`, in that it doesn't have any assumptions to the format of the input text (or summary).
Additionally, as the LLMs like to hallucinate when fact checking or get confused by context, it is sometimes beneficial to run the checker multiple times.  It does this by feeding the rewritten "True" result back on itself, and checking the "facts" for truth.  As you can see from the examples below, this can be very effective in arriving at a generally true body of text.

You can control the number of times the checker runs by setting the `max_checks` parameter.  The default is 2, but you can set it to 1 if you don't want any double-checking.
"""
logger.info("# Summarization checker chain")


llm = ChatOllama(temperature=0)
checker_chain = LLMSummarizationCheckerChain.from_llm(
    llm, verbose=True, max_checks=2)
text = """
Your 9-year old might like these recent discoveries made by The James Webb Space Telescope (JWST):
• In 2023, The JWST spotted a number of galaxies nicknamed "green peas." They were given this name because they are small, round, and green, like peas.
• The telescope captured images of galaxies that are over 13 billion years old. This means that the light from these galaxies has been traveling for over 13 billion years to reach us.
• JWST took the very first pictures of a planet outside of our own solar system. These distant worlds are called "exoplanets." Exo means "from outside."
These discoveries can spark a child's imagination about the infinite wonders of the universe."""
checker_chain.run(text)


llm = ChatOllama(temperature=0)
checker_chain = LLMSummarizationCheckerChain.from_llm(
    llm, verbose=True, max_checks=3)
text = "The Greenland Sea is an outlying portion of the Arctic Ocean located between Iceland, Norway, the Svalbard archipelago and Greenland. It has an area of 465,000 square miles and is one of five oceans in the world, alongside the Pacific Ocean, Atlantic Ocean, Indian Ocean, and the Southern Ocean. It is the smallest of the five oceans and is covered almost entirely by water, some of which is frozen in the form of glaciers and icebergs. The sea is named after the island of Greenland, and is the Arctic Ocean's main outlet to the Atlantic. It is often frozen over so navigation is limited, and is considered the northern branch of the Norwegian Sea."
checker_chain.run(text)


llm = ChatOllama(temperature=0)
checker_chain = LLMSummarizationCheckerChain.from_llm(
    llm, max_checks=3, verbose=True)
text = "Mammals can lay eggs, birds can lay eggs, therefore birds are mammals."
checker_chain.run(text)

logger.info("\n\n[DONE]", bright=True)
