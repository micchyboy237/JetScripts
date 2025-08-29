from athina.evals import RagasAnswerRelevancy
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
from langchain.vectorstores import Chroma
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# **Rewrite-Retrieve-Read (RRR)**
Rewrite-Retrieve-Read is a three-step framework for tasks that involve retrieval augmentation, such as open-domain question answering. It focuses on improving the quality of retrieved information and generating accurate outputs by refining the input query.

Research Paper: [Rewrite-Retrieve-Read](https://arxiv.org/pdf/2305.14283)

## **Initial Setup**
"""
logger.info("# **Rewrite-Retrieve-Read (RRR)**")

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents, embeddings)

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
You are a helpful assistant that answers questions based on the following context.
If you don't find the answer in the context, just say that you don't know.
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

"""
## **Simple Query**
"""
logger.info("## **Simple Query**")

simple_query = "who directed the matrix"

response = rag_chain.invoke(simple_query)
response

"""
## **Distracted Query**
"""
logger.info("## **Distracted Query**")

distracted_query = "who create the matrics"

response = rag_chain.invoke(distracted_query)
response

"""
## **Rewrite Retrieve Read**
"""
logger.info("## **Rewrite Retrieve Read**")

template = """Provide a better search query for \
web search engine to answer the given question, end \
the queries with ’**’. Question: \
{x} Answer:"""

rewrite_prompt = ChatPromptTemplate.from_template(template)


def _parse(text):
    return text.strip('"').strip("**")


rewriter = rewrite_prompt | ChatOllama(
    model="llama3.1") | StrOutputParser() | _parse

rewriter.invoke({"x": distracted_query})

rewrite_retrieve_read_chain = (
    {
        "context": {"x": RunnablePassthrough()} | rewriter | retriever,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

rewrite_retrieve_read_chain.invoke(distracted_query)

"""
## **Preparing Data for Evaluation**
"""
logger.info("## **Preparing Data for Evaluation**")

response = []
contexts = []
questions = []

rewritten_query = rewriter.invoke({"x": distracted_query})
questions.append(rewritten_query)
response.append(rewrite_retrieve_read_chain.invoke(distracted_query))
contexts.append(
    [docs.page_content for docs in retriever.get_relevant_documents(rewritten_query)])


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

We will use **Answer Relevancy** eval here. It Measures how pertinent the generated response is to the given prompt. Please refer to our [documentation](https://docs.athina.ai/api-reference/evals/preset-evals/overview) for further details
"""
logger.info("## **Evaluation in Athina AI**")

# OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

dataset = Loader().load_dict(df_dict)

RagasAnswerRelevancy(model="llama3.1").run_batch(data=dataset).to_df()

logger.info("\n\n[DONE]", bright=True)
