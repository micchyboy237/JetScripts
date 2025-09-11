from deepeval.metrics.answer_relevancy import AnswerRelevancy
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.callbacks.confident_callback import DeepEvalCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import os
import requests
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
# Confident

>[DeepEval](https://confident-ai.com) package for unit testing LLMs.
> Using Confident, everyone can build robust language models through faster iterations
> using both unit testing and integration testing. We provide support for each step in the iteration
> from synthetic data creation to testing.

In this guide we will demonstrate how to test and measure LLMs in performance. We show how you can use our callback to measure performance and how you can define your own metric and log them into our dashboard.

DeepEval also offers:
- How to generate synthetic data
- How to measure performance
- A dashboard to monitor and review results over time

## Installation and Setup
"""
logger.info("# Confident")

# %pip install --upgrade --quiet  langchain langchain-ollama langchain-community deepeval langchain-chroma

"""
### Getting API Credentials

To get the DeepEval API credentials, follow the next steps:

1. Go to https://app.confident-ai.com
2. Click on "Organization"
3. Copy the API Key.


When you log in, you will also be asked to set the `implementation` name. The implementation name is required to describe the type of implementation. (Think of what you want to call your project. We recommend making it descriptive.)
"""
logger.info("### Getting API Credentials")

# !deepeval login

"""
### Setup DeepEval

You can, by default, use the `DeepEvalCallbackHandler` to set up the metrics you want to track. However, this has limited support for metrics at the moment (more to be added soon). It currently supports:
- [Answer Relevancy](https://docs.confident-ai.com/docs/measuring_llm_performance/answer_relevancy)
- [Bias](https://docs.confident-ai.com/docs/measuring_llm_performance/debias)
- [Toxicness](https://docs.confident-ai.com/docs/measuring_llm_performance/non_toxic)
"""
logger.info("### Setup DeepEval")


answer_relevancy_metric = AnswerRelevancy(minimum_score=0.5)

"""
## Get Started

To use the `DeepEvalCallbackHandler`, we need the `implementation_name`.
"""
logger.info("## Get Started")


deepeval_callback = DeepEvalCallbackHandler(
    implementation_name="langchainQuickstart", metrics=[answer_relevancy_metric]
)

"""
### Scenario 1: Feeding into LLM

You can then feed it into your LLM with Ollama.
"""
logger.info("### Scenario 1: Feeding into LLM")


llm = ChatOllama(
    temperature=0,
    callbacks=[deepeval_callback],
    verbose=True,
    ollama_)
output = llm.generate(
    [
        "What is the best evaluation tool out there? (no bias at all)",
    ]
)

"""
You can then check the metric if it was successful by calling the `is_successful()` method.
"""
logger.info(
    "You can then check the metric if it was successful by calling the `is_successful()` method.")

answer_relevancy_metric.is_successful()

"""
Once you have ran that, you should be able to see our dashboard below. 

![Dashboard](https://docs.confident-ai.com/assets/images/dashboard-screenshot-b02db73008213a211b1158ff052d969e.png)

### Scenario 2: Tracking an LLM in a chain without callbacks

To track an LLM in a chain without callbacks, you can plug into it at the end.

We can start by defining a simple chain as shown below.
"""
logger.info("### Scenario 2: Tracking an LLM in a chain without callbacks")


text_file_url = "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"

ollama_

with open("state_of_the_union.txt", "w") as f:
    response = requests.get(text_file_url)
    f.write(response.text)

loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=Ollama(ollama_api_key=ollama_api_key),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
)

query = "Who is the president?"
result = qa.run(query)

"""
After defining a chain, you can then manually check for answer similarity.
"""
logger.info(
    "After defining a chain, you can then manually check for answer similarity.")

answer_relevancy_metric.measure(result, query)
answer_relevancy_metric.is_successful()

"""
### What's next?

You can create your own custom metrics [here](https://docs.confident-ai.com/docs/quickstart/custom-metrics). 

DeepEval also offers other features such as being able to [automatically create unit tests](https://docs.confident-ai.com/docs/quickstart/synthetic-data-creation), [tests for hallucination](https://docs.confident-ai.com/docs/measuring_llm_performance/factual_consistency).

If you are interested, check out our Github repository here [https://github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval). We welcome any PRs and discussions on how to improve LLM performance.
"""
logger.info("### What's next?")

logger.info("\n\n[DONE]", bright=True)
