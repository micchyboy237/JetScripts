# from athina.evals import RagasContextRelevancy
# from athina.keys import AthinaApiKey, OpenAiApiKey
# from athina.loaders import Loader
from datasets import Dataset
# from google.colab import userdata
from jet.adapters.langchain.ragas_context_relevancy import RagasContextRelevancy
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.document_loaders import CSVLoader
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# **Contextual RAG**

Contextual Retrieval-Augmented Generation (RAG) is an advanced RAG technique that improves response relevance and efficiency by incorporating contextual compression during the retrieval process. Traditional RAG retrieves and sends full documents to the generation model, which may include irrelevant information, leading to higher costs and less accurate responses.

In Contextual RAG, the retrieved documents are processed through a Document Compressor before being passed to the language model. This compressor extracts and retains only the most relevant information for the query, or even discards entire irrelevant documents. This approach reduces the noise in the retrieved context, resulting in more precise, concise, and cost-effective responses from the generation model.

Reference: [Contextual RAG](https://python.langchain.com/docs/how_to/contextual_compression/)

## **Initial Setup**
"""
logger.info("# **Contextual RAG**")

# !pip install --q athina chromadb

# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
# os.environ['ATHINA_API_KEY'] = userdata.get('ATHINA_API_KEY')

"""
## **Indexing**
"""
logger.info("## **Indexing**")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

loader = CSVLoader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/rag-cookbooks/data/context.csv")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents, embeddings)

"""
## **Retriever**
"""
logger.info("## **Retriever**")

retriever = vectorstore.as_retriever()

"""
## **Contextual Retriever**
"""
logger.info("## **Contextual Retriever**")

llm = ChatOllama(model="llama3.1")


compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("what are points on a mortgage")
compressed_docs

"""
## **RAG Chain**
"""
logger.info("## **RAG Chain**")


template = """"
You are a helpful assistant that answers questions based on the following context.
If you don't find the answer in the context, just say that you don't know.
Context: {context}

Question: {input}

Answer:

"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": compression_retriever,  "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("what are points on a mortgage")
response

"""
## **Preparing Data for Evaluation**
"""
logger.info("## **Preparing Data for Evaluation**")

questions = ["what are points on a mortgage"]
response = []
contexts = []

for query in questions:
    response.append(rag_chain.invoke(query))
    contexts.append(
        [docs.page_content for docs in compression_retriever.get_relevant_documents(query)])

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

We will use **Context Relevancy** eval here. It Measures the relevancy of the retrieved context, calculated based on both the query and contexts. To learn more about this. Please refer to our [documentation](https://docs.athina.ai/api-reference/evals/preset-evals/overview) for further details
"""
logger.info("## **Evaluation in Athina AI**")

# OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
# AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

dataset = Dataset.from_dict(df_dict)

RagasContextRelevancy(model="llama3.1", request_timeout=300.0,
                      context_window=4096).run_batch(data=dataset).to_df()

logger.info("\n\n[DONE]", bright=True)
