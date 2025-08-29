from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.monsterapi import MonsterLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/monsterapi.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Monster API <> LLamaIndex

MonsterAPI Hosts wide range of popular LLMs as inference service and this notebook serves as a tutorial about how to use llama-index to access MonsterAPI LLMs.


Check us out here: https://monsterapi.ai/

Install Required Libraries
"""
logger.info("# Monster API <> LLamaIndex")

# %pip install llama-index-llms-monsterapi

# !python3 -m pip install llama-index --quiet -y
# !python3 -m pip install monsterapi --quiet
# !python3 -m pip install sentence_transformers --quiet

"""
Import required modules
"""
logger.info("Import required modules")


"""
### Set Monster API Key env variable

Sign up on [MonsterAPI](https://monsterapi.ai/signup?utm_source=llama-index-colab&utm_medium=referral) and get a free auth key. Paste it below:
"""
logger.info("### Set Monster API Key env variable")

os.environ["MONSTER_API_KEY"] = ""

"""
## Basic Usage Pattern

Set the model
"""
logger.info("## Basic Usage Pattern")

model = "meta-llama/Meta-Llama-3-8B-Instruct"

"""
Initiate LLM module
"""
logger.info("Initiate LLM module")

llm = MonsterLLM(model=model, temperature=0.75)

"""
### Completion Example
"""
logger.info("### Completion Example")

result = llm.complete("Who are you?")
logger.debug(result)

"""
### Chat Example
"""
logger.info("### Chat Example")


history_message = ChatMessage(
    **{
        "role": "user",
        "content": (
            "When asked 'who are you?' respond as 'I am qblocks llm model'"
            " everytime."
        ),
    }
)
current_message = ChatMessage(**{"role": "user", "content": "Who are you?"})

response = llm.chat([history_message, current_message])
logger.debug(response)

"""
##RAG Approach to import external knowledge into LLM as context

Source Paper: https://arxiv.org/pdf/2005.11401.pdf

Retrieval-Augmented Generation (RAG) is a method that uses a combination of pre-defined rules or parameters (non-parametric memory) and external information from the internet (parametric memory) to generate responses to questions or create new ones. By lever

Install pypdf library needed to install pdf parsing library
"""
logger.info("##RAG Approach to import external knowledge into LLM as context")

# !python3 -m pip install pypdf --quiet

"""
Lets try to augment our LLM with RAG source paper PDF as external information.
Lets download the pdf into data dir
"""
logger.info(
    "Lets try to augment our LLM with RAG source paper PDF as external information.")

# !rm -r ./data
# !mkdir -p data&&cd data&&curl 'https://arxiv.org/pdf/2005.11401.pdf' -o "RAG.pdf"

"""
Load the document
"""
logger.info("Load the document")

documents = SimpleDirectoryReader(
    f"{os.path.dirname(__file__)}/data").load_data()

"""
Initiate LLM and Embedding Model
"""
logger.info("Initiate LLM and Embedding Model")

llm = MonsterLLM(model=model, temperature=0.75, context_window=1024)
embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
splitter = SentenceSplitter(chunk_size=1024)

"""
Create embedding store and create index
"""
logger.info("Create embedding store and create index")

index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], embed_model=embed_model
)
query_engine = index.as_query_engine(llm=llm)

"""
Actual LLM output without RAG:
"""
logger.info("Actual LLM output without RAG:")

response = llm.complete("What is Retrieval-Augmented Generation?")
logger.debug(response)

"""
LLM Output with RAG
"""
logger.info("LLM Output with RAG")

response = query_engine.query("What is Retrieval-Augmented Generation?")
logger.debug(response)

"""
## LLM with RAG using our Monster Deploy service

Monster Deploy enables you to host any vLLM supported large language model (LLM) like Tinyllama, Mixtral, Phi-2 etc as a rest API endpoint on MonsterAPI's cost optimised GPU cloud. 

With MonsterAPI's integration in Llama index, you can use your deployed LLM API endpoints to create RAG system or RAG bot for use cases such as: 
- Answering questions on your documents 
- Improving the content of your documents 
- Finding context of importance in your documents 


Once deployment is launched use the base_url and api_auth_token once deployment is live and use them below.

Note: When using LLama index to access Monster Deploy LLMs, you need to create a prompt with required template and send compiled prompt as input. 
See `LLama Index Prompt Template Usage example` section for more details.

see [here](https://developer.monsterapi.ai/docs/monster-deploy-beta) for more details

Once deployment is launched use the base_url and api_auth_token once deployment is live and use them below. 

Note: When using LLama index to access Monster Deploy LLMs, you need to create a prompt with reqhired template and send compiled prompt as input. see section `LLama Index Prompt Template Usage example` for more details.
"""
logger.info("## LLM with RAG using our Monster Deploy service")

deploy_llm = MonsterLLM(
    model="<Replace with basemodel used to deploy>",
    api_base="https://ecc7deb6-26e0-419b-a7f2-0deb934af29a.monsterapi.ai",
    api_key="a0f8a6ba-c32f-4407-af0c-169f1915490c",
    temperature=0.75,
)

"""
### General Usage Pattern
"""
logger.info("### General Usage Pattern")

deploy_llm.complete("What is Retrieval-Augmented Generation?")

"""
#### Chat Example
"""
logger.info("#### Chat Example")


history_message = ChatMessage(
    **{
        "role": "user",
        "content": (
            "When asked 'who are you?' respond as 'I am qblocks llm model'"
            " everytime."
        ),
    }
)
current_message = ChatMessage(**{"role": "user", "content": "Who are you?"})

response = deploy_llm.chat([history_message, current_message])
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
