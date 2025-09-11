from jet.logger import logger
from langchain_sambanova import ChatSambaNovaCloud
from langchain_sambanova import ChatSambaStudio
from langchain_sambanova import SambaNovaCloudEmbeddings
from langchain_sambanova import SambaStudioEmbeddings
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
# SambaNova

Customers are turning to [SambaNova](https://sambanova.ai/) to quickly deploy state-of-the-art AI capabilities to gain competitive advantage. Our purpose-built enterprise-scale AI platform is the technology backbone for the next generation of AI computing. We power the foundation models that unlock the valuable business insights trapped in data.

Designed for AI, the SambaNova RDU was built with a revolutionary dataflow architecture. This design makes the RDU significantly more efficient for these workloads than GPUs as it eliminates redundant calls to memory, which are an inherent limitation of how GPUs function. This built-in efficiency is one of the features that makes the RDU capable of much higher performance than GPUs in a fraction of the footprint.

On top of our architecture We have developed some platforms that allow companies and developers to get full advantage of the RDU processors and open source models.

### SambaNovaCloud

SambaNova's [SambaNova Cloud](https://cloud.sambanova.ai/) is a platform for performing inference with open-source models

You can obtain a free SambaNovaCloud API key [here](https://cloud.sambanova.ai/)

### SambaStudio

SambaNova's [SambaStudio](https://docs.sambanova.ai/sambastudio/latest/sambastudio-intro.html) is a rich, GUI-based platform that provides the functionality to train, deploy, and manage models in SambaNova DataScale systems.

## Installation and Setup

Install the integration package:

```bash
pip install langchain-sambanova
```

set your API key it as an environment variable:

If you are a SambaNovaCloud user:

```bash
export SAMBANOVA_API_KEY="your-sambanova-cloud-api-key-here"
```

or if you are SambaStudio User

```bash
export SAMBASTUDIO_API_KEY="your-sambastudio-api-key-here"
```

## Chat models
"""
logger.info("# SambaNova")


llm = ChatSambaNovaCloud(model="Meta-Llama-3.3-70B-Instruct", temperature=0.7)
llm.invoke("Tell me a joke about artificial intelligence.")

"""
For a more detailed walkthrough of the ChatSambaNovaCloud component, see [this notebook](https://python.langchain.com/docs/integrations/chat/sambanova/)
"""
logger.info("For a more detailed walkthrough of the ChatSambaNovaCloud component, see [this notebook](https://python.langchain.com/docs/integrations/chat/sambanova/)")


llm = ChatSambaStudio(model="Meta-Llama-3.3-70B-Instruct", temperature=0.7)
llm.invoke("Tell me a joke about artificial intelligence.")

"""
For a more detailed walkthrough of the ChatSambaStudio component, see [this notebook](https://python.langchain.com/docs/integrations/chat/sambastudio/)

## Embedding Models
"""
logger.info("## Embedding Models")


embeddings = SambaNovaCloudEmbeddings(model="E5-Mistral-7B-Instruct")
embeddings.embed_query("What is the meaning of life?")

"""
For a more detailed walkthrough of the SambaNovaCloudEmbeddings component, see [this notebook](https://python.langchain.com/docs/integrations/text_embedding/sambanova/)
"""
logger.info("For a more detailed walkthrough of the SambaNovaCloudEmbeddings component, see [this notebook](https://python.langchain.com/docs/integrations/text_embedding/sambanova/)")


embeddings = SambaStudioEmbeddings(model="e5-mistral-7b-instruct")
embeddings.embed_query("What is the meaning of life?")

"""
For a more detailed walkthrough of the SambaStudioEmbeddings component, see [this notebook](https://python.langchain.com/docs/integrations/text_embedding/sambastudio/)

API Reference [langchain-sambanova](https://docs.sambanova.ai/cloud/api-reference)
"""
logger.info("For a more detailed walkthrough of the SambaStudioEmbeddings component, see [this notebook](https://python.langchain.com/docs/integrations/text_embedding/sambastudio/)")

logger.info("\n\n[DONE]", bright=True)