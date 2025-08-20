from google.colab import userdata
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from llama_parse import LlamaParse
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# NVIDIA NIMs

The `llama-index-llms-nvidia` package contains LlamaIndex integrations building applications with models on 
NVIDIA NIM inference microservice. NIM supports models across domains like chat, embedding, and re-ranking models 
from the community as well as NVIDIA. These models are optimized by NVIDIA to deliver the best performance on NVIDIA 
accelerated infrastructure and deployed as a NIM, an easy-to-use, prebuilt containers that deploy anywhere using a single 
command on NVIDIA accelerated infrastructure.

NVIDIA hosted deployments of NIMs are available to test on the [NVIDIA API catalog](https://build.nvidia.com/). After testing, 
NIMs can be exported from NVIDIA’s API catalog using the NVIDIA AI Enterprise license and run on-premises or in the cloud, 
giving enterprises ownership and full control of their IP and AI application.

NIMs are packaged as container images on a per model basis and are distributed as NGC container images through the NVIDIA NGC Catalog. 
At their core, NIMs provide easy, consistent, and familiar APIs for running inference on an AI model.
"""
logger.info("# NVIDIA NIMs")

# !pip install llama-index-core
# !pip install llama-index-readers-file
# !pip install llama-index-llms-nvidia
# !pip install llama-index-embeddings-nvidia
# !pip install llama-index-postprocessor-nvidia-rerank

"""
Bring in a test dataset, a PDF about housing construction in San Francisco in 2021.
"""
logger.info("Bring in a test dataset, a PDF about housing construction in San Francisco in 2021.")

# !mkdir data
# !wget "https://www.dropbox.com/scl/fi/p33j9112y0ysgwg77fdjz/2021_Housing_Inventory.pdf?rlkey=yyok6bb18s5o31snjd2dxkxz3&dl=0" -O f"{GENERATED_DIR}/housing_data.pdf"

"""
## Setup

Import our dependencies and set up our NVIDIA API key from the API catalog, https://build.nvidia.com for the two models we'll use hosted on the catalog (embedding and re-ranking models).

**To get started:**

1. Create a free account with [NVIDIA](https://build.nvidia.com/), which hosts NVIDIA AI Foundation models.

2. Click on your model of choice.

3. Under Input select the Python tab, and click `Get API Key`. Then click `Generate Key`.

4. Copy and save the generated key as NVIDIA_API_KEY. From there, you should have access to the endpoints.
"""
logger.info("## Setup")


os.environ["NVIDIA_API_KEY"] = userdata.get("nvidia-api-key")

"""
Let's use a NVIDIA hosted NIM for the embedding model.

NVIDIA's default embeddings only embed the first 512 tokens so we've set our chunk size to 500 to maximize the accuracy of our embeddings.
"""
logger.info("Let's use a NVIDIA hosted NIM for the embedding model.")

Settings.text_splitter = SentenceSplitter(chunk_size=500)

documents = SimpleDirectoryReader("./data").load_data()

"""
We set our embedding model to NVIDIA's default. If a chunk exceeds the number of tokens the model can encode, the default is to throw an error, so we set `truncate="END"` to instead discard tokens that go over the limit (hopefully not many because of our chunk size above).
"""
logger.info("We set our embedding model to NVIDIA's default. If a chunk exceeds the number of tokens the model can encode, the default is to throw an error, so we set `truncate="END"` to instead discard tokens that go over the limit (hopefully not many because of our chunk size above).")

Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

index = VectorStoreIndex.from_documents(documents)

"""
Now we've embedded our data and indexed it in memory, we set up our LLM that's self-hosted locally. NIM can be hosted locally using Docker in 5 minutes, following this [NIM quick start guide](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html).

Below, we show how to:
- use Meta's open-source `meta/llama3-8b-instruct` model as a local NIM and
- `meta/llama3-70b-instruct` as a NIM from the API catalog hosted by NVIDIA.

If you are using a local NIM, make sure you change the `base_url` to your deployed NIM URL!

We're going to retrieve the top 5 most relevant chunks to answer our question.
"""
logger.info("Now we've embedded our data and indexed it in memory, we set up our LLM that's self-hosted locally. NIM can be hosted locally using Docker in 5 minutes, following this [NIM quick start guide](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html).")

Settings.llm = NVIDIA(model="meta/llama3-70b-instruct")

query_engine = index.as_query_engine(similarity_top_k=20)

"""
Let's ask it a simple question we know is answered in one place in the document (on page 18).
"""
logger.info("Let's ask it a simple question we know is answered in one place in the document (on page 18).")

response = query_engine.query(
    "How many new housing units were built in San Francisco in 2021?"
)
logger.debug(response)

"""
Now let's ask it a more complicated question that requires reading a table (it's on page 41 of the document):
"""
logger.info("Now let's ask it a more complicated question that requires reading a table (it's on page 41 of the document):")

response = query_engine.query(
    "What was the net gain in housing units in the Mission in 2021?"
)
logger.debug(response)

"""
That's no good! This is net new, which isn't the number we wanted. Let's try a more advanced PDF parser, LlamaParse:
"""
logger.info("That's no good! This is net new, which isn't the number we wanted. Let's try a more advanced PDF parser, LlamaParse:")

# !pip install llama-parse


# import nest_asyncio

# nest_asyncio.apply()

os.environ["LLAMA_CLOUD_API_KEY"] = userdata.get("llama-cloud-key")

parser = LlamaParse(
    result_type="markdown"  # "markdown" and "text" are available
)

file_extractor = {".pdf": parser}
documents2 = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

index2 = VectorStoreIndex.from_documents(documents2)
query_engine2 = index2.as_query_engine(similarity_top_k=20)

response = query_engine2.query(
    "What was the net gain in housing units in the Mission in 2021?"
)
logger.debug(response)

"""
Perfect! With a better parser, the LLM is able to answer the question.

Let's now try a trickier question:
"""
logger.info("Perfect! With a better parser, the LLM is able to answer the question.")

response = query_engine2.query(
    "How many affordable housing units were completed in 2021?"
)
logger.debug(response)

"""
The LLM is getting confused; this appears to be the percentage increase in housing units.

Let's try giving the LLM more context (40 instead of 20) and then sorting those chunks with a reranker. We'll use NVIDIA's reranker for this:
"""
logger.info("The LLM is getting confused; this appears to be the percentage increase in housing units.")


query_engine3 = index2.as_query_engine(
    similarity_top_k=40, node_postprocessors=[NVIDIARerank(top_n=10)]
)

response = query_engine3.query(
    "How many affordable housing units were completed in 2021?"
)
logger.debug(response)

"""
Excellent! Now the figure is correct (this is on page 35, in case you're wondering).
"""
logger.info("Excellent! Now the figure is correct (this is on page 35, in case you're wondering).")

logger.info("\n\n[DONE]", bright=True)