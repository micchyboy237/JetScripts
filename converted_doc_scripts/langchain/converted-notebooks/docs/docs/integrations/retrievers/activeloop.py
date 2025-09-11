from bs4 import BeautifulSoup
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain.chains.ollama_functions import (
create_structured_output_chain,
)
from langchain_community.document_loaders.async_html import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import DeepLake
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from ragas.langchain import RagasEvaluatorChain
from ragas.metrics import (
context_recall,
)
from tqdm import tqdm
from typing import List
from urllib.parse import urljoin
import os
import random
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
# Activeloop Deep Memory

>[Activeloop Deep Memory](https://docs.activeloop.ai/performance-features/deep-memory) is a suite of tools that enables you to optimize your Vector Store for your use-case and achieve higher accuracy in your LLM apps.

`Retrieval-Augmented Generatation` (`RAG`) has recently gained significant attention. As advanced RAG techniques and agents emerge, they expand the potential of what RAGs can accomplish. However, several challenges may limit the integration of RAGs into production. The primary factors to consider when implementing RAGs in production settings are accuracy (recall), cost, and latency. For basic use cases, Ollama's Ada model paired with a naive similarity search can produce satisfactory results. Yet, for higher accuracy or recall during searches, one might need to employ advanced retrieval techniques. These methods might involve varying data chunk sizes, rewriting queries multiple times, and more, potentially increasing latency and costs.  Activeloop's [Deep Memory](https://www.activeloop.ai/resources/use-deep-memory-to-boost-rag-apps-accuracy-by-up-to-22/) a feature available to `Activeloop Deep Lake` users, addresses these issuea by introducing a tiny neural network layer trained to match user queries with relevant data from a corpus. While this addition incurs minimal latency during search, it can boost retrieval accuracy by up to 27
% and remains cost-effective and simple to use, without requiring any additional advanced rag techniques.

For this tutorial we will parse `DeepLake` documentation, and create a RAG system that could answer the question from the docs.

## 1. Dataset Creation

We will parse activeloop's docs for this tutorial using `BeautifulSoup` library and LangChain's document parsers like `Html2TextTransformer`, `AsyncHtmlLoader`. So we will need to install the following libraries:
"""
logger.info("# Activeloop Deep Memory")

# %pip install --upgrade --quiet  tiktoken langchain-ollama python-dotenv datasets langchain deeplake beautifulsoup4 html2text ragas

"""
Also you'll need to create a [Activeloop](https://activeloop.ai) account.
"""
logger.info(
    "Also you'll need to create a [Activeloop](https://activeloop.ai) account.")

ORG_ID = "..."


# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Ollama API token: ")
if "ACTIVELOOP_TOKEN" not in os.environ:
#     os.environ["ACTIVELOOP_TOKEN"] = getpass.getpass(
        "Enter your ActiveLoop API token: "
    )  # Get your API token from https://app.activeloop.ai, click on your profile picture in the top right corner, and select "API Tokens"

token = os.getenv("ACTIVELOOP_TOKEN")
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = DeepLake(
    dataset_path=f"hub://{ORG_ID}/deeplake-docs-deepmemory",  # org_id stands for your username or organization from activeloop
    embedding=ollama_embeddings,
    runtime={"tensor_db": True},
    token=token,
    read_only=False,
)

"""
parsing all links in the webpage using `BeautifulSoup`
"""
logger.info("parsing all links in the webpage using `BeautifulSoup`")




def get_all_links(url):
    response = requests.get(url)
    if response.status_code != 200:
        logger.debug(f"Failed to retrieve the page: {url}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    links = [
        urljoin(url, a["href"]) for a in soup.find_all("a", href=True) if a["href"]
    ]

    return links


base_url = "https://docs.deeplake.ai/en/latest/"
all_links = get_all_links(base_url)

"""
Loading data:
"""
logger.info("Loading data:")


loader = AsyncHtmlLoader(all_links)
docs = loader.load()


"""
Converting data into user readable format:
"""
logger.info("Converting data into user readable format:")


html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

"""
Now, let us chunk further the documents as some of the contain too much text:
"""
logger.info("Now, let us chunk further the documents as some of the contain too much text:")


chunk_size = 4096
docs_new = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
)

for doc in docs_transformed:
    if len(doc.page_content) < chunk_size:
        docs_new.append(doc)
    else:
        docs = text_splitter.create_documents([doc.page_content])
        docs_new.extend(docs)

"""
Populating VectorStore:
"""
logger.info("Populating VectorStore:")

docs = db.add_documents(docs_new)

"""
## 2. Generating synthetic queries and training Deep Memory

Next step would be to train a deep_memory model that will align your users queries with the dataset that you already have. If you don't have any user queries yet, no worries, we will generate them using LLM!

#### TODO: Add image

Here above we showed the overall schema how deep_memory works. So as you can see, in order to train it you need relevance, queries together with corpus data (data that we want to query). Corpus data was already populated in the previous section, here we will be generating questions and relevance. 

1. `questions` - is a text of strings, where each string represents a query
2. `relevance` - contains links to the ground truth for each question. There might be several docs that contain answer to the given question. Because of this relevenve is `List[List[tuple[str, float]]]`, where outer list represents queries and inner list relevant documents. Tuple contains str, float pair where string represent the id of the source doc (corresponds to the `id` tensor in the dataset), while float corresponds to how much current document is related to the question.

Now, let us generate synthetic questions and relevance:
"""
logger.info("## 2. Generating synthetic queries and training Deep Memory")



docs = db.vectorstore.dataset.text.data(fetch_chunks=True, aslist=True)["value"]
ids = db.vectorstore.dataset.id.data(fetch_chunks=True, aslist=True)["value"]

llm = ChatOllama(model="llama3.2")


class Questions(BaseModel):
    """Identifying information about a person."""

    question: str = Field(..., description="Questions about text")


prompt_msgs = [
    SystemMessage(
        content="You are a world class expert for generating questions based on provided context. \
                You make sure the question can be answered by the text."
    ),
    HumanMessagePromptTemplate.from_template(
        "Use the given text to generate a question from the following input: {input}"
    ),
    HumanMessage(content="Tips: Make sure to answer in the correct format"),
]
prompt = ChatPromptTemplate(messages=prompt_msgs)
chain = create_structured_output_chain(Questions, llm, prompt, verbose=True)

text = "# Understanding Hallucinations and Bias ## **Introduction** In this lesson, we'll cover the concept of **hallucinations** in LLMs, highlighting their influence on AI applications and demonstrating how to mitigate them using techniques like the retriever's architectures. We'll also explore **bias** within LLMs with examples."
questions = chain.run(input=text)
logger.debug(questions)




def generate_queries(docs: List[str], ids: List[str], n: int = 100):
    questions = []
    relevances = []
    pbar = tqdm(total=n)
    while len(questions) < n:
        r = random.randint(0, len(docs) - 1)
        text, label = docs[r], ids[r]

        generated_qs = [chain.run(input=text).question]
        questions.extend(generated_qs)
        relevances.extend([[(label, 1)] for _ in generated_qs])
        pbar.update(len(generated_qs))
        if len(questions) % 10 == 0:
            logger.debug(f"q: {len(questions)}")
    return questions[:n], relevances[:n]


chain = create_structured_output_chain(Questions, llm, prompt, verbose=False)
questions, relevances = generate_queries(docs, ids, n=200)

train_questions, train_relevances = questions[:100], relevances[:100]
test_questions, test_relevances = questions[100:], relevances[100:]

"""
Now we created 100 training queries as well as 100 queries for testing. Now let us train the deep_memory:
"""
logger.info("Now we created 100 training queries as well as 100 queries for testing. Now let us train the deep_memory:")

job_id = db.vectorstore.deep_memory.train(
    queries=train_questions,
    relevance=train_relevances,
)

"""
Let us track the training progress:
"""
logger.info("Let us track the training progress:")

db.vectorstore.deep_memory.status("6538939ca0b69a9ca45c528c")

"""
## 3. Evaluating Deep Memory performance

Great we've trained the model! It's showing some substantial improvement in recall, but how can we use it now and evaluate on unseen new data? In this section we will delve into model evaluation and inference part and see how it can be used with LangChain in order to increase retrieval accuracy

### 3.1 Deep Memory evaluation

For the beginning we can use deep_memory's builtin evaluation method. 
It calculates several `recall` metrics.
It can be done easily in a few lines of code.
"""
logger.info("## 3. Evaluating Deep Memory performance")

recall = db.vectorstore.deep_memory.evaluate(
    queries=test_questions,
    relevance=test_relevances,
)

"""
It is showing quite substatntial improvement on an unseen test dataset too!!!

### 3.2 Deep Memory + RAGas
"""
logger.info("### 3.2 Deep Memory + RAGas")


"""
Let us convert recall into ground truths:
"""
logger.info("Let us convert recall into ground truths:")

def convert_relevance_to_ground_truth(docs, relevance):
    ground_truths = []

    for rel in relevance:
        ground_truth = []
        for doc_id, _ in rel:
            ground_truth.append(docs[doc_id])
        ground_truths.append(ground_truth)
    return ground_truths

ground_truths = convert_relevance_to_ground_truth(docs, test_relevances)

for deep_memory in [False, True]:
    logger.debug("\nEvaluating with deep_memory =", deep_memory)
    logger.debug("===================================")

    retriever = db.as_retriever()
    retriever.search_kwargs["deep_memory"] = deep_memory

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOllama(model="llama3.2"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    metrics = {
        "context_recall_score": 0,
    }

    eval_chains = {m.name: RagasEvaluatorChain(metric=m) for m in [context_recall]}

    for question, ground_truth in zip(test_questions, ground_truths):
        result = qa_chain({"query": question})
        result["ground_truths"] = ground_truth
        for name, eval_chain in eval_chains.items():
            score_name = f"{name}_score"
            metrics[score_name] += eval_chain(result)[score_name]

    for metric in metrics:
        metrics[metric] /= len(test_questions)
        logger.debug(f"{metric}: {metrics[metric]}")
    logger.debug("===================================")

"""
### 3.3 Deep Memory Inference

#### TODO: Add image

with deep_memory
"""
logger.info("### 3.3 Deep Memory Inference")

retriever = db.as_retriever()
retriever.search_kwargs["deep_memory"] = True
retriever.search_kwargs["k"] = 10

query = "Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophic G-to-A mutations in the viral genome."
qa = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"), chain_type="stuff", retriever=retriever
)
logger.debug(qa.run(query))

"""
without deep_memory
"""
logger.info("without deep_memory")

retriever = db.as_retriever()
retriever.search_kwargs["deep_memory"] = False
retriever.search_kwargs["k"] = 10

query = "Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophic G-to-A mutations in the viral genome."
qa = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"), chain_type="stuff", retriever=retriever
)
qa.run(query)

"""
### 3.4 Deep Memory cost savings

Deep Memory increases retrieval accuracy without altering your existing workflow. Additionally, by reducing the top_k input into the LLM, you can significantly cut inference costs via lower token usage.
"""
logger.info("### 3.4 Deep Memory cost savings")

logger.info("\n\n[DONE]", bright=True)