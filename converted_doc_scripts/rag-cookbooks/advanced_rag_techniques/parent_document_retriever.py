from athina.evals import RagasContextRecall
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.loaders import Loader
from datasets import Dataset
from google.colab import userdata
from jet.llm.ollama.base_langchain import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.document_loaders import CSVLoader
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# **Parent Document Retriver**

Parent Document retriever is a technique where large documents are split into smaller pieces, called "child chunks." These chunks are stored in a way that lets the system find and compare specific parts of a document with a user’s query. The large document, or "parent," is still kept but is only retrieved if one of its child chunks is relevant to the query.

Reference: [Parent Document Retriver](https://python.langchain.com/docs/how_to/parent_document_retriever/)

## **Initial Setup**
"""
logger.info("# **Parent Document Retriver**")

# ! pip install --q athina chromadb

# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ['ATHINA_API_KEY'] = userdata.get('ATHINA_API_KEY')

"""
## **Indexing**
"""
logger.info("## **Indexing**")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

loader = CSVLoader("./context.csv")
documents = loader.load()

"""
### **Parent Child Text Spliting**
"""
logger.info("### **Parent Child Text Spliting**")


parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryStore()

vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings)

"""
## **Retriever**
"""
logger.info("## **Retriever**")

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)

"""
## **RAG Chain**
"""
logger.info("## **RAG Chain**")

llm = ChatOllama(model="llama3.1")


template = """"
You are a helpful assistant that answers questions based on the following context
Context: {context}

Question: {input}

Answer:

"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever,  "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("who played the lead roles in the movie leaving las vegas")
response

"""
## **Preparing Data for Evaluation**
"""
logger.info("## **Preparing Data for Evaluation**")

question = ["who played the lead roles in the movie leaving las vegas"]
response = []
contexts = []
ground_truth = ["Nicolas Cage stars as a suicidal alcoholic who has ended his personal and professional life to drink himself to death in Las Vegas ."]
for query in question:
  response.append(rag_chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

data = {
    "query": question,
    "response": response,
    "context": contexts,
    "expected_response": ground_truth
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

We will use **Context Recall** eval here. It Measures the extent to which the retrieved context aligns with the expected response. Please refer to our [documentation](https://docs.athina.ai/api-reference/evals/preset-evals/overview) for further details
"""
logger.info("## **Evaluation in Athina AI**")

# OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

dataset = Loader().load_dict(df_dict)

RagasContextRecall(model="llama3.1", request_timeout=300.0, context_window=4096).run_batch(data=dataset).to_df()

logger.info("\n\n[DONE]", bright=True)