from athina.evals import RagasAnswerRelevancy
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.loaders import Loader
from datasets import Dataset
from google.colab import userdata
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.document_loaders import CSVLoader
from langchain.load import dumps, loads
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# **RAG Fusion**
RAG-Fusion is an enhanced version of the traditional Retrieval-Augmented Generation (RAG) model. In RAG-Fusion, after receiving a query, the model first generates related sub-queries using a large language model. These sub-queries help find more relevant documents. Instead of simply sending the retrieved documents to the model, RAG-Fusion uses a technique called Reciprocal Rank Fusion (RRF) to score and reorder the documents based on their relevance. The best-ranked documents are then used to generate a more accurate response.

Research Paper: [RAG Fusion](https://arxiv.org/pdf/2402.03367)

## **Initial Setup**
"""
logger.info("# **RAG Fusion**")

# ! pip install --q athina langsmith

# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ['ATHINA_API_KEY'] = userdata.get('ATHINA_API_KEY')
os.environ['QDRANT_API_KEY'] = userdata.get('QDRANT_API_KEY')

"""
## **Indexing**
"""
logger.info("## **Indexing**")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

loader = CSVLoader("./context.csv")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

"""
## **Qdrant Vector Database**
"""
logger.info("## **Qdrant Vector Database**")

vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    url="your_qdrant_url",
    prefer_grpc=True,
    collection_name="documents",
    api_key=os.environ["QDRANT_API_KEY"],
)

"""
## **Chromadb (Optional)**
"""
logger.info("## **Chromadb (Optional)**")


"""
## **Retriever**
"""
logger.info("## **Retriever**")

retriever = vectorstore.as_retriever()

"""
## **Reciprocal Rank Fusion Chain**
"""
logger.info("## **Reciprocal Rank Fusion Chain**")

llm = ChatOllama(model="llama3.2")

client = Client()
prompt = client.pull_prompt("langchain-ai/rag-fusion-query-generation")

generate_queries = (
    prompt | ChatOllama(model="llama3.2") | StrOutputParser() | (
        lambda x: x.split("\n"))
)


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


chain = generate_queries | retriever.map() | reciprocal_rank_fusion

chain.input_schema.schema()

chain.invoke("what are points on a mortgage")

"""
## **RAG Chain**
"""
logger.info("## **RAG Chain**")


template = """Answer the question based only on the following context.
If you don't find the answer in the context, just say that you don't know.

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_fusion_chain = (
    {
        "context": chain,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

rag_fusion_chain.invoke("what are points on a mortgage")

"""
## **Preparing Data for Evaluation**
"""
logger.info("## **Preparing Data for Evaluation**")

question = ["what are points on a mortgage"]
response = []
contexts = []
ground_truths = [
    "Points, sometimes also called a 'discount point', are a form of pre-paid interest."]

for query in question:
    response.append(rag_fusion_chain.invoke(query))
    contexts.append(
        [docs.page_content for docs in retriever.get_relevant_documents(query)])

data = {
    "query": question,
    "response": response,
    "context": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(data)

df = pd.DataFrame(dataset)

df

df_dict = df.to_dict(orient='records')

for record in df_dict:
    if not isinstance(record.get('context'), list):
        if record.get('context') is None:
            record['context'] = []
        else:
            record['context'] = [record['context']]

"""
## **Evaluation in Athina AI**

We will use **Answer Relevancy** eval here. It Measures how pertinent the generated response is to the given prompt. Please refer to our [documentation](https://docs.athina.ai/api-reference/evals/preset-evals/overview) for further details.
"""
logger.info("## **Evaluation in Athina AI**")

# OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

dataset = Loader().load_dict(df_dict)

RagasAnswerRelevancy(
    model="llama3.2", log_dir=f"{LOG_DIR}/chats").run_batch(data=dataset).to_df()

logger.info("\n\n[DONE]", bright=True)
