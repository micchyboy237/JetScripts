from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import KayAiRetriever
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
Press Releases Data
=

Press Releases data powered by [Kay.ai](https://kay.ai).

>Press releases are used by companies to announce something noteworthy, including product launches, financial performance reports, partnerships, and other significant news. They are widely used by analysts to track corporate strategy, operational updates and financial performance.
Kay.ai obtains press releases of all US public companies from a variety of sources, which include the company's official press room and partnerships with various data API providers. 
This data is updated till Sept 30th for free access, if you want to access the real-time feed, reach out to us at hello@kay.ai or [tweet at us](https://twitter.com/vishalrohra_)

Setup
=

First you will need to install the `kay` package. You will also need an API key: you can get one for free at [https://kay.ai](https://kay.ai/). Once you have an API key, you must set it as an environment variable `KAY_API_KEY`.

In this example we're going to use the `KayAiRetriever`. Take a look at the [kay notebook](/docs/integrations/retrievers/kay) for more detailed information for the parmeters that it accepts.

Examples
=
"""
logger.info("Press Releases Data")

# from getpass import getpass

# KAY_API_KEY = getpass()
# OPENAI_API_KEY = getpass()


os.environ["KAY_API_KEY"] = KAY_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


model = ChatOllama(model="llama3.2")
retriever = KayAiRetriever.create(
    dataset_id="company", data_types=["PressRelease"], num_contexts=6
)
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

questions = [
    "How is the healthcare industry adopting generative AI tools?",
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    logger.debug(f"-> **Question**: {question} \n")
    logger.debug(f"**Answer**: {result['answer']} \n")

logger.info("\n\n[DONE]", bright=True)