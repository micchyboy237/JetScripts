from athina.evals import DoesResponseAnswerQuery
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.loaders import Loader
from datasets import Dataset
from google.colab import userdata
from jet.llm.ollama.base_langchain import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.document_loaders import CSVLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# **Naive RAG**
The Naive RAG is the simplest technique in the RAG ecosystem, providing a straightforward approach to combining retrieved data with LLM models for efficient user responses.

Research Paper: [RAG](https://arxiv.org/pdf/2005.11401)

## **Initial Setup**
"""
logger.info("# **Naive RAG**")

# ! pip install --q athina

# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ['ATHINA_API_KEY'] = userdata.get('ATHINA_API_KEY')
os.environ['PINECONE_API_KEY'] = userdata.get('PINECONE_API_KEY')

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
## **Pinecone Vector Database**
"""
logger.info("## **Pinecone Vector Database**")

pc = PineconeClient(
    api_key=os.environ.get("PINECONE_API_KEY"),
)

pc.create_index(
    name='my-index',
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index_name = "my-index"

vectorstore = Pinecone.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)

"""
## **FAISS (Optional)**
"""
logger.info("## **FAISS (Optional)**")


"""
## **Retriever**
"""
logger.info("## **Retriever**")

retriever = vectorstore.as_retriever()

"""
## **RAG Chain**
"""
logger.info("## **RAG Chain**")

llm = ChatOllama(model="llama3.1")


template = """"
You are a helpful assistant that answers questions based on the provided context.
Use the provided context to answer the question.
Question: {input}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever,  "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("when did ww1 end?")
response

"""
## **Preparing Data for Evaluation**
"""
logger.info("## **Preparing Data for Evaluation**")

question = ["when did ww1 end?"]
response = []
contexts = []

for query in question:
    response.append(rag_chain.invoke(query))
    contexts.append(
        [docs.page_content for docs in retriever.get_relevant_documents(query)])

data = {
    "query": question,
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

We will use **Does Response Answer Query** eval here. It Checks if the response answer the user's query. To learn more about this. Please refer to our [documentation](https://docs.athina.ai/api-reference/evals/preset-evals/overview) for further details.
"""
logger.info("## **Evaluation in Athina AI**")

# OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

dataset = Loader().load_dict(df_dict)

DoesResponseAnswerQuery(model="llama3.1").run_batch(data=dataset).to_df()

logger.info("\n\n[DONE]", bright=True)
