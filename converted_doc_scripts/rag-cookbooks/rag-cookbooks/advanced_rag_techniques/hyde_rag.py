from athina.evals import RagasContextRelevancy
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.loaders import Loader
from datasets import Dataset
from google.colab import userdata
from jet.llm.ollama.base_langchain import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.init import Auth
import os
import pandas as pd
import weaviate

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# **Hypothetical Document Embeddings (HyDE) RAG**

HyDE operates by creating hypothetical document embeddings that represent ideal documents relevant to a given query. This method contrasts with conventional RAG systems, which typically rely on the similarity between a user's query and existing document embeddings. By generating these hypothetical embeddings, HyDE effectively guides the retrieval process towards documents that are more likely to contain pertinent information.

Research Paper: [HyDE](https://arxiv.org/pdf/2212.10496)

## **Initial Setup**
"""
logger.info("# **Hypothetical Document Embeddings (HyDE) RAG**")

# ! pip install --q -U athina langchain-weaviate

# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ['ATHINA_API_KEY'] = userdata.get('ATHINA_API_KEY')
os.environ['WEAVIATE_API_KEY'] = userdata.get('WEAVIATE_API_KEY')

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
## **Weaviate Vector Database**
"""
logger.info("## **Weaviate Vector Database**")


wcd_url = 'your_url'

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=Auth.api_key(os.environ['WEAVIATE_API_KEY']),
#     headers={'X-Ollama-Api-key': os.environ["OPENAI_API_KEY"]}
)

vectorstore = WeaviateVectorStore.from_documents(documents,embedding=OllamaEmbeddings(model="mxbai-embed-large"),client=client, index_name="your_collection_name",text_key="text")

vectorstore.similarity_search("world war II", k=3)

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
## **Hypothetical Answer Chain**
"""
logger.info("## **Hypothetical Answer Chain**")

llm = ChatOllama(model="llama3.1")


template ="""
You are a helpful assistant that answers questions.
Question: {input}
Answer:
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{input}"),
    ]
)
qa_no_context = prompt | llm | StrOutputParser()

question = 'how does interlibrary loan work'
answer = qa_no_context.invoke({"input": question})
answer

"""
## **Combined RAG Chain**
"""
logger.info("## **Combined RAG Chain**")

retrieval_chain = qa_no_context | retriever
retrieved_docs = retrieval_chain.invoke({"input":question})

template = """
You are a helpful assistant that answers questions based on the provided context.
Use the provided context to answer the question.
Question: {input}
Context: {context}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context":retrieved_docs,"input":question})

"""
## **Preparing Data for Evaluation**
"""
logger.info("## **Preparing Data for Evaluation**")

question = ["how does interlibrary loan work"]
response = []
contexts = []

for query in question:
  response.append(final_rag_chain.invoke({"context":retrieved_docs,"input":query}))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

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

We will use **Context Relevancy** eval here. It Measures the relevancy of the retrieved context, calculated based on both the query and contexts. Please refer to our [documentation](https://docs.athina.ai/api-reference/evals/preset-evals/overview) for further details
"""
logger.info("## **Evaluation in Athina AI**")

# OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

dataset = Loader().load_dict(df_dict)

RagasContextRelevancy(model="llama3.1", request_timeout=300.0, context_window=4096).run_batch(data=dataset).to_df()

logger.info("\n\n[DONE]", bright=True)