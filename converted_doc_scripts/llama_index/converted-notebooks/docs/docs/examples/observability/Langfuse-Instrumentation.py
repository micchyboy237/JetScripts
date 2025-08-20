from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from langfuse.llama_index import LlamaIndexInstrumentor
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/Langfuse-Instrumentation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Cookbook LlamaIndex Integration (Instrumentation Module)

This is a simple cookbook that demonstrates how to use the [LlamaIndex Langfuse integration](https://langfuse.com/docs/integrations/llama-index/get-started) using the [instrumentation module](https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/) by LlamaIndex (available in llama-index v0.10.20 and later).

<div style="position: relative; padding-top: 69.85769728331177%;">
  <iframe
    src="https://customer-xnej9vqjtgxpafyk.cloudflarestream.com/221d302474945951062d06767c475c0a/iframe?muted=true&loop=true&autoplay=true&poster=https%3A%2F%2Fcustomer-xnej9vqjtgxpafyk.cloudflarestream.com%2F221d302474945951062d06767c475c0a%2Fthumbnails%2Fthumbnail.jpg%3Ftime%3D%26height%3D600"
    loading="lazy"
    style="border: white; position: absolute; top: 0; left: 0; height: 100%; width: 100%; border-radius: 10px;"
    allow="accelerometer; gyroscope; autoplay; encrypted-media; picture-in-picture;"
    allowfullscreen="true"
  ></iframe>
</div>

## Setup

Make sure you have both `llama-index` and `langfuse` installed.
"""
logger.info("# Cookbook LlamaIndex Integration (Instrumentation Module)")

# %pip install langfuse llama_index --upgrade

"""
Initialize the integration. Get your API keys from the Langfuse project settings. This example uses MLX for embeddings and chat completions. You can also use any other model supported by LlamaIndex.
"""
logger.info("Initialize the integration. Get your API keys from the Langfuse project settings. This example uses MLX for embeddings and chat completions. You can also use any other model supported by LlamaIndex.")


os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # 🇪🇺 EU region

# os.environ["OPENAI_API_KEY"] = ""

"""
Register the Langfuse `LlamaIndexInstrumentor`:
"""
logger.info("Register the Langfuse `LlamaIndexInstrumentor`:")


instrumentor = LlamaIndexInstrumentor()
instrumentor.start()

"""
## Index
"""
logger.info("## Index")


doc1 = Document(
    text="""
Maxwell "Max" Silverstein, a lauded movie director, screenwriter, and producer, was born on October 25, 1978, in Boston, Massachusetts. A film enthusiast from a young age, his journey began with home movies shot on a Super 8 camera. His passion led him to the University of Southern California (USC), majoring in Film Production. Eventually, he started his career as an assistant director at Paramount Pictures. Silverstein's directorial debut, “Doors Unseen,” a psychological thriller, earned him recognition at the Sundance Film Festival and marked the beginning of a successful directing career.
"""
)
doc2 = Document(
    text="""
Throughout his career, Silverstein has been celebrated for his diverse range of filmography and unique narrative technique. He masterfully blends suspense, human emotion, and subtle humor in his storylines. Among his notable works are "Fleeting Echoes," "Halcyon Dusk," and the Academy Award-winning sci-fi epic, "Event Horizon's Brink." His contribution to cinema revolves around examining human nature, the complexity of relationships, and probing reality and perception. Off-camera, he is a dedicated philanthropist living in Los Angeles with his wife and two children.
"""
)


index = VectorStoreIndex.from_documents([doc1, doc2])

"""
## Query
"""
logger.info("## Query")

response = index.as_query_engine().query("What did he do growing up?")
logger.debug(response)

"""
Example trace: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/d933c7cc-20bf-4db3-810d-bab1c8d9a2a1
"""
logger.info("Example trace: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/d933c7cc-20bf-4db3-810d-bab1c8d9a2a1")

response = index.as_chat_engine().chat("What did he do growing up?")
logger.debug(response)

"""
Example trace: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/4e285b8f-9789-4cf0-a8b4-45473ac420f1

![LlamaIndex Chat Engine Trace in Langfuse (via instrumentation module)](https://langfuse.com/images/cookbook/integration_llama-index_instrumentation_chatengine_trace.png)

## Custom trace properties

You can use the `instrumentor.observe` context manager to manage trace IDs, set custom trace properties, and access the trace client for later scoring.
"""
logger.info("## Custom trace properties")

with instrumentor.observe(user_id="my-user", session_id="my-session") as trace:
    response = index.as_query_engine().query("What did he do growing up?")

trace.score(name="my-score", value=0.5)

"""
Example trace: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/6f554d6b-a2bc-4fba-904f-aa54de2897ca
"""
logger.info("Example trace: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/6f554d6b-a2bc-4fba-904f-aa54de2897ca")

logger.info("\n\n[DONE]", bright=True)