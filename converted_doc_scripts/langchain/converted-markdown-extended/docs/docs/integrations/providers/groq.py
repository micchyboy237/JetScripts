from jet.logger import logger
from langchain_groq import ChatGroq
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
# Groq

>[Groq](https://groq.com) developed the world's first Language Processing Unit™, or `LPU`.
> The `Groq LPU` has a deterministic, single core streaming architecture that sets the standard
> for GenAI inference speed with predictable and repeatable performance for any given workload.
>
>Beyond the architecture, `Groq` software is designed to empower developers like you with
> the tools you need to create innovative, powerful AI applications.
>
>With Groq as your engine, you can:
>* Achieve uncompromised low latency and performance for real-time AI and HPC inferences 🔥
>* Know the exact performance and compute time for any given workload 🔮
>* Take advantage of our cutting-edge technology to stay ahead of the competition 💪


## Installation and Setup

Install the integration package:
"""
logger.info("# Groq")

pip install langchain-groq

"""
Request an [API key](https://console.groq.com/login?utm_source=langchain&utm_content=provider_page) and set it as an environment variable:
"""
logger.info("Request an [API key](https://console.groq.com/login?utm_source=langchain&utm_content=provider_page) and set it as an environment variable:")

export GROQ_API_KEY=gsk_...

"""
## Chat models

See a [usage example](/docs/integrations/chat/groq).
"""
logger.info("## Chat models")


logger.info("\n\n[DONE]", bright=True)