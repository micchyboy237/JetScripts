from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import ScaNN
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models.google_palm import ChatGooglePalm

initialize_ollama_settings()

"""
# ScaNN

ScaNN (Scalable Nearest Neighbors) is a method for efficient vector similarity search at scale.

ScaNN includes search space pruning and quantization for Maximum Inner Product Search and also supports other distance functions such as Euclidean distance. The implementation is optimized for x86 processors with AVX2 support. See its [Google Research github](https://github.com/google-research/google-research/tree/master/scann) for more details.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

## Installation
Install ScaNN through pip. Alternatively, you can follow instructions on the [ScaNN Website](https://github.com/google-research/google-research/tree/master/scann#building-from-source) to install from source.
"""

# %pip install --upgrade --quiet  scann

"""
## Retrieval Demo

Below we show how to use ScaNN in conjunction with Huggingface Embeddings.
"""


loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

db = ScaNN.from_documents(docs, embeddings)
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

docs[0]

"""
## RetrievalQA Demo

Next, we demonstrate using ScaNN in conjunction with Google PaLM API.

You can obtain an API key from https://developers.generativeai.google/tutorials/setup
"""


palm_client = ChatGooglePalm(google_api_key="YOUR_GOOGLE_PALM_API_KEY")

qa = RetrievalQA.from_chain_type(
    llm=palm_client,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 10}),
)

logger.debug(qa.run("What did the president say about Ketanji Brown Jackson?"))

logger.debug(qa.run("What did the president say about Michael Phelps?"))

"""
## Save and loading local retrieval index
"""

db.save_local("/tmp/db", "state_of_union")
restored_db = ScaNN.load_local(
    "/tmp/db", embeddings, index_name="state_of_union")

logger.info("\n\n[DONE]", bright=True)
