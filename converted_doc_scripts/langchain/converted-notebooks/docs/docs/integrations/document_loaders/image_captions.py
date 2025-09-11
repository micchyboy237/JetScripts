from PIL import Image
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import ImageCaptionLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# Image captions

By default, the loader utilizes the pre-trained [Salesforce BLIP image captioning model](https://huggingface.co/Salesforce/blip-image-captioning-base).

This notebook shows how to use the `ImageCaptionLoader` to generate a queryable index of image captions.
"""
logger.info("# Image captions")

# %pip install -qU transformers jet.adapters.langchain.chat_ollama langchain_chroma

# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

"""
### Prepare a list of image urls from Wikimedia
"""
logger.info("### Prepare a list of image urls from Wikimedia")


list_image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Ara_ararauna_Luc_Viatour.jpg/1554px-Ara_ararauna_Luc_Viatour.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/1928_Model_A_Ford.jpg/640px-1928_Model_A_Ford.jpg",
]

"""
### Create the loader
"""
logger.info("### Create the loader")

loader = ImageCaptionLoader(images=list_image_urls)
list_docs = loader.load()
list_docs


Image.open(requests.get(list_image_urls[0], stream=True).raw).convert("RGB")

"""
### Create the index
"""
logger.info("### Create the index")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(list_docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="mxbai-embed-large"))

retriever = vectorstore.as_retriever(k=2)

"""
### Query
"""
logger.info("### Query")


model = ChatOllama(model="llama3.2")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "What animals are in the images?"})

logger.debug(response["answer"])

response = rag_chain.invoke({"input": "What kind of images are there?"})

logger.debug(response["answer"])

logger.info("\n\n[DONE]", bright=True)