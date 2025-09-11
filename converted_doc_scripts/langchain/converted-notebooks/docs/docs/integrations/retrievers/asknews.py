from datetime import datetime, timedelta
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.retrievers import AskNewsRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
# AskNews

> [AskNews](https://asknews.app) infuses any LLM with the latest global news (or historical news), using a single natural language query. Specifically, AskNews is enriching over 300k articles per day by translating, summarizing, extracting entities, and indexing them into hot and cold vector databases. AskNews puts these vector databases on a low-latency endpoint for you. When you query AskNews, you get back a prompt-optimized string that contains all the most pertinent enrichments (e.g. entities, classifications, translation, summarization). This means that you do not need to manage your own news RAG, and you do not need to worry about how to properly convey news information in a condensed way to your LLM.
> AskNews is also committed to transparency, which is why our coverage is monitored and diversified across hundreds of countries, 13 languages, and 50 thousand sources. If you'd like to track our source coverage, you can visit our [transparency dashboard](https://asknews.app/en/transparency).

## Setup

The integration lives in the `langchain-community` package. We also need to install the `asknews` package itself.

```bash
pip install -U langchain-community asknews
```

We also need to set our AskNews API credentials, which can be generated at the [AskNews console](https://my.asknews.app).
"""
logger.info("# AskNews")

# import getpass

# os.environ["ASKNEWS_CLIENT_ID"] = getpass.getpass()
# os.environ["ASKNEWS_CLIENT_SECRET"] = getpass.getpass()

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability")



"""
## Usage
"""
logger.info("## Usage")


retriever = AskNewsRetriever(k=3)

retriever.invoke("impact of fed policy on the tech sector")


start = (datetime.now() - timedelta(days=7)).timestamp()
end = datetime.now().timestamp()

retriever = AskNewsRetriever(
    k=3,
    categories=["Business", "Technology"],
    start_timestamp=int(start),  # defaults to 48 hours ago
    end_timestamp=int(end),  # defaults to now
    method="kw",  # defaults to "nl", natural language, can also be "kw" for keyword search
    offset=10,  # allows you to paginate results
)

retriever.invoke("federal reserve S&P500")

"""
## Chaining

We can easily combine this retriever in to a chain.
"""
logger.info("## Chaining")

# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass()


prompt = ChatPromptTemplate.from_template(
    """The following news articles may come in handy for answering the question:

{context}

Question:

{question}"""
)
chain = (
    RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | ChatOllama(model="llama3.2")
    | StrOutputParser()
)

chain.invoke({"question": "What is the impact of fed policy on the tech sector?"})

logger.info("\n\n[DONE]", bright=True)