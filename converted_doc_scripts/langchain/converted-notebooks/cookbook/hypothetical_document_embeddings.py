from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
import os
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
# Improve document indexing with HyDE
This notebook goes over how to use Hypothetical Document Embeddings (HyDE), as described in [this paper](https://arxiv.org/abs/2212.10496). 

At a high level, HyDE is an embedding technique that takes queries, generates a hypothetical answer, and then embeds that generated document and uses that as the final example. 

In order to use HyDE, we therefore need to provide a base embedding model, as well as an LLMChain that can be used to generate those documents. By default, the HyDE class comes with some default prompts to use (see the paper for more details on them), but we can also create our own.
"""
logger.info("# Improve document indexing with HyDE")


base_embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama()

"""

"""

embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm, base_embeddings, "web_search")

result = embeddings.embed_query("Where is the Taj Mahal?")

"""
## Multiple generations
We can also generate multiple documents and then combine the embeddings for those. By default, we combine those by taking the average. We can do this by changing the LLM we use to generate documents to return multiple things.
"""
logger.info("## Multiple generations")

multi_llm = ChatOllama(n=4, best_of=4)

embeddings = HypotheticalDocumentEmbedder.from_llm(
    multi_llm, base_embeddings, "web_search"
)

result = embeddings.embed_query("Where is the Taj Mahal?")

"""
## Using our own prompts
Besides using preconfigured prompts, we can also easily construct our own prompts and use those in the LLMChain that is generating the documents. This can be useful if we know the domain our queries will be in, as we can condition the prompt to generate text more similar to that.

In the example below, let's condition it to generate text about a state of the union address (because we will use that in the next example).
"""
logger.info("## Using our own prompts")

prompt_template = """Please answer the user's question about the most recent state of the union address
Question: {question}
Answer:"""
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain, base_embeddings=base_embeddings
)

result = embeddings.embed_query(
    "What did the president say about Ketanji Brown Jackson"
)

"""
## Using HyDE
Now that we have HyDE, we can use it as we would any other embedding class! Here is using it to find similar passages in the state of the union example.
"""
logger.info("## Using HyDE")


with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

docsearch = Chroma.from_texts(texts, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)

logger.debug(docs[0].page_content)

logger.info("\n\n[DONE]", bright=True)
