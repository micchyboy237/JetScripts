from athina.evals import RagasContextRelevancy
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.loaders import Loader
from datasets import Dataset
from google.colab import userdata
from jet.llm.ollama.base_langchain import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.document_loaders import CSVLoader
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
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
# **Hybrid RAG**

Hybrid RAG refers to an advanced retrieval technique that combines vector similarity search with traditional search methods, such as full-text search or BM25. This approach enables more comprehensive and flexible information retrieval by leveraging the strengths of both methods, vector similarity for semantic understanding and traditional techniques for precise keyword or text-based matching.

Research Paper: [paper1](https://arxiv.org/pdf/2408.05141) and [paper2](https://arxiv.org/pdf/2408.04948)

## **Initial Setup**
"""
logger.info("# **Hybrid RAG**")

# ! pip install --q athina chromadb rank_bm25

# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ['ATHINA_API_KEY'] = userdata.get('ATHINA_API_KEY')

"""
## **Indexing**
"""
logger.info("## **Indexing**")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

loader = CSVLoader("./context.csv")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents, embeddings)

"""
## **Retrievers**
"""
logger.info("## **Retrievers**")

retriever = vectorstore.as_retriever()

"""
### **Keyword Retriever**
"""
logger.info("### **Keyword Retriever**")

keyword_retriever = BM25Retriever.from_documents(documents)
keyword_retriever.k =  3

keyword_retriever.get_relevant_documents("what bacteria grow on macconkey agar")

"""
### **Ensemble Retriever**
"""
logger.info("### **Ensemble Retriever**")

ensemble_retriever = EnsembleRetriever(retrievers=[retriever, keyword_retriever], weights=[0.5, 0.5])

ensemble_retriever.get_relevant_documents("what bacteria grow on macconkey agar")

"""
## **RAG Chain**
"""
logger.info("## **RAG Chain**")

llm = ChatOllama(model="llama3.2")


template = """"
You are a helpful assistant that answers questions based on the following context.
If you don't find the answer in the context, just say that you don't know.
Context: {context}

Question: {input}

Answer:

"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": ensemble_retriever,  "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke('what bacteria grow on macconkey agar')
response

"""
## **Preparing Data for Evaluation**
"""
logger.info("## **Preparing Data for Evaluation**")

questions = ["what bacteria grow on macconkey agar", "who wrote a rose is a rose is a rose"]
response = []
contexts = []

for query in questions:
  response.append(rag_chain.invoke(query))
  contexts.append([docs.page_content for docs in ensemble_retriever.get_relevant_documents(query)])

data = {
    "query": questions,
    "response": response,
    "context": contexts,
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

We will use **Context Relevancy** eval here. It Measures the relevancy of the retrieved context, calculated based on both the query and contexts. Please refer to our [documentation](https://docs.athina.ai/api-reference/evals/preset-evals/overview) for further details
"""
logger.info("## **Evaluation in Athina AI**")

# OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

dataset = Loader().load_dict(df_dict)

RagasContextRelevancy(model="llama3.2", log_dir=f"{LOG_DIR}/chats").run_batch(data=dataset).to_df()

logger.info("\n\n[DONE]", bright=True)