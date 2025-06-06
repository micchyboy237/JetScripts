from PIL import Image
from dotenv import load_dotenv
from jet.logger import CustomLogger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import fitz  # PyMuPDF
import google.generativeai as genai
import io
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/multi_model_rag_with_captioning.ipynb)

### Overview: 
This code implements one of the multiple ways of multi-model RAG. It extracts and processes text and images from PDFs, utilizing a multi-modal Retrieval-Augmented Generation (RAG) system for summarizing and retrieving content for question answering.

### Key Components:
   - **PyMuPDF**: For extracting text and images from PDFs.
   - **Gemini 1.5-flash model**: To summarize images and tables.
   - **Cohere Embeddings**: For embedding document splits.
   - **Chroma Vectorstore**: To store and retrieve document embeddings.
   - **LangChain**: To orchestrate the retrieval and generation pipeline.

### Diagram:
   <img src="../images/multi_model_rag_with_captioning.svg" alt="Reliable-RAG" width="300">

### Motivation: 
Efficiently summarize complex documents to facilitate easy retrieval and concise responses for multi-modal data.

### Method Details:
   - Text and images are extracted from the PDF using PyMuPDF.
   - Summarization is performed on extracted images and tables using Gemini.
   - Embeddings are generated via Cohere for storage in Chroma.
   - A similarity-based retriever fetches relevant sections based on the query.

### Benefits:
   - Simplified retrieval from complex, multi-modal documents.
   - Streamlined Q&A process for both text and images.
   - Flexible architecture for expanding to more document types.

### Implementation:
   - Documents are split into chunks with overlap using a text splitter.
   - Summarized text and image content are stored as vectors.
   - Queries are handled by retrieving relevant document segments and generating concise answers.

### Summary: 
The project enables multi-modal document processing and retrieval, providing concise, relevant responses by combining state-of-the-art LLMs and vector-based retrieval systems.

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("### Overview:")

# !pip install langchain langchain-community pillow pymupdf python-dotenv



load_dotenv()

"""
### Download the "Attention is all you need" paper
"""
logger.info("### Download the "Attention is all you need" paper")

# !wget https://arxiv.org/pdf/1706.03762
# !mv 1706.03762 attention_is_all_you_need.pdf

"""
### Data Extraction
"""
logger.info("### Data Extraction")

text_data = []
img_data = []

with fitz.open('attention_is_all_you_need.pdf') as pdf_file:
    if not os.path.exists("extracted_images"):
        os.makedirs("extracted_images")

    for page_number in range(len(pdf_file)):
        page = pdf_file[page_number]

        text = page.get_text().strip()
        text_data.append({"response": text, "name": page_number+1})
        images = page.get_images(full=True)

        for image_index, img in enumerate(images, start=0):
            xref = img[0]  # Get the XREF of the image
            base_image = pdf_file.extract_image(xref)  # Extract the image
            image_bytes = base_image["image"]  # Get the image bytes
            image_ext = base_image["ext"]  # Get the image extension

            image = Image.open(io.BytesIO(image_bytes))
            image.save(f"extracted_images/image_{page_number+1}_{image_index+1}.{image_ext}")

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

"""
### Image Captioning
"""
logger.info("### Image Captioning")

for img in os.listdir("extracted_images"):
    image = Image.open(f"extracted_images/{img}")
    response = model.generate_content([image, "You are an assistant tasked with summarizing tables, images and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text or image:"])
    img_data.append({"response": response.text, "name": img})

"""
### Vectostore
"""
logger.info("### Vectostore")

embedding_model = CohereEmbeddings(model="embed-english-v3.0")

docs_list = [Document(page_content=text['response'], metadata={"name": text['name']}) for text in text_data]
img_list = [Document(page_content=img['response'], metadata={"name": img['name']}) for img in img_data]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=400, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)
img_splits = text_splitter.split_documents(img_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits + img_splits, # adding the both text and image splits
    collection_name="multi_model_rag",
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 1}, # number of documents to retrieve
            )

"""
### Query
"""
logger.info("### Query")

query = "What is the BLEU score of the Transformer (base model)?"

docs = retriever.invoke(query)

"""
### Output
"""
logger.info("### Output")


system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge.
Use three-to-five sentences maximum and keep the answer concise."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>"),
    ]
)

llm = ChatCohere(model="command-r-plus", temperature=0)

rag_chain = prompt | llm | StrOutputParser()

generation = rag_chain.invoke({"documents":docs[0].page_content, "question": query})
logger.debug(generation)

logger.info("\n\n[DONE]", bright=True)