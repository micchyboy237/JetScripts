from athina.evals import DoesResponseAnswerQuery
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.loaders import Loader
from collections import Counter
from datasets import Dataset
from google.colab import userdata
from jet.llm.ollama.base_langchain import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from unstructured.partition.pdf import partition_pdf
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# **Unstructured RAG**
Unstructured or (Semi-Structured) RAG is a method designed to handle documents that combine text, tables, and images. It addresses challenges like broken tables caused by text splitting and the difficulty of embedding tables for semantic search.

Here we are using unstructured.io to parse and separate text, tables, and images.

Tool Reference: [Unstructured](https://unstructured.io/)

## **Initial Setup**
"""
logger.info("# **Unstructured RAG**")

# ! pip install --q athina faiss-gpu pytesseract unstructured-client "unstructured[all-docs]"

# !apt-get install poppler-utils
# !apt-get install tesseract-ocr
# !apt-get install libtesseract-dev

# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ['ATHINA_API_KEY'] = userdata.get('ATHINA_API_KEY')

"""
## **Indexing**
"""
logger.info("## **Indexing**")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")


filename = "/content/sample.pdf"

pdf_elements = partition_pdf(
    filename=filename,
    extract_images_in_pdf=True,
    strategy="hi_res",
    hi_res_model_name="yolox",
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=3000,
    combine_text_under_n_chars=200,
)

category_counts = Counter(str(type(element)) for element in pdf_elements)
unique_categories = set(category_counts)
category_counts

unique_types = {el.to_dict()['type'] for el in pdf_elements}
unique_types


documents = [Document(page_content=el.text, metadata={
                      "source": filename}) for el in pdf_elements]

"""
## **Vector Store**
"""
logger.info("## **Vector Store**")

vectorstore = FAISS.from_documents(documents, embeddings)

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
You are a helpful assistant that answers questions based on the provided context, which can include text and tables.
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

response = rag_chain.invoke(
    "Compare all the Training Results on MATH Test Set")
response

"""
## **Preparing Data for Evaluation**
"""
logger.info("## **Preparing Data for Evaluation**")

question = ["Compare all the Training Results on MATH Test Set"]
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
