import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.maritalk import Maritalk
import asyncio
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/maritalk.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Maritalk

## Introduction

MariTalk is an assistant developed by the Brazilian company [Maritaca AI](https://www.maritaca.ai).
MariTalk is based on language models that have been specially trained to understand Portuguese well.

This notebook demonstrates how to use MariTalk with Llama Index through two examples:

1. Get pet name suggestions with chat method;
2. Classify film reviews as negative or positive with few-shot examples with complete method.

## Installation
If you're opening this Notebook on colab, you will probably need to install LlamaIndex.
"""
logger.info("# Maritalk")

# !pip install llama-index
# !pip install llama-index-llms-maritalk
# !pip install asyncio

"""
## API Key
You will need an API key that can be obtained from chat.maritaca.ai ("Chaves da API" section).

### Example 1 - Pet Name Suggestions with Chat
"""
logger.info("## API Key")



llm = Maritalk(api_key="<your_maritalk_api_key>", model="sabia-2-medium")

messages = [
    ChatMessage(
        role="system",
        content="You are an assistant specialized in suggesting pet names. Given the animal, you must suggest 4 names.",
    ),
    ChatMessage(role="user", content="I have a dog."),
]

response = llm.chat(messages)
logger.debug(response)


async def get_dog_name(llm, messages):
    async def run_async_code_c21ac29f():
        async def run_async_code_65496cfb():
            response = llm.chat(messages)
            return response
        response = asyncio.run(run_async_code_65496cfb())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_c21ac29f())
    logger.success(format_json(response))
    logger.debug(response)


asyncio.run(get_dog_name(llm, messages))

"""
#### Stream Generation

For tasks involving the generation of long text, such as creating an extensive article or translating a large document, it can be advantageous to receive the response in parts, as the text is generated, instead of waiting for the complete text. This makes the application more responsive and efficient, especially when the generated text is extensive. We offer two approaches to meet this need: one synchronous and another asynchronous.
"""
logger.info("#### Stream Generation")

response = llm.stream_chat(messages)
for chunk in response:
    logger.debug(chunk.delta, end="", flush=True)


async def get_dog_name_streaming(llm, messages):
    async def run_async_code_b5433fad():
        for chunk in llm.stream_chat(messages):
        return 
     = asyncio.run(run_async_code_b5433fad())
    logger.success(format_json())
        logger.debug(chunk.delta, end="", flush=True)


asyncio.run(get_dog_name_streaming(llm, messages))

"""
### Example 2 - Few-shot Examples with Complete

We recommend using the `llm.complete()` method when using the model with few-shot examples
"""
logger.info("### Example 2 - Few-shot Examples with Complete")

prompt = """Classifique a resenha de filme como "positiva" ou "negativa".

Resenha: Gostei muito do filme, é o melhor do ano!
Classe: positiva

Resenha: O filme deixa muito a desejar.
Classe: negativa

Resenha: Apesar de longo, valeu o ingresso..
Classe:"""

response = llm.complete(prompt)
logger.debug(response)


async def classify_review(llm, prompt):
    async def run_async_code_9b8fdb37():
        async def run_async_code_4512bb86():
            response = llm.complete(prompt)
            return response
        response = asyncio.run(run_async_code_4512bb86())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_9b8fdb37())
    logger.success(format_json(response))
    logger.debug(response)


asyncio.run(classify_review(llm, prompt))

response = llm.stream_complete(prompt)
for chunk in response:
    logger.debug(chunk.delta, end="", flush=True)


async def classify_review_streaming(llm, prompt):
    async def run_async_code_d3eedec4():
        for chunk in llm.stream_complete(prompt):
        return 
     = asyncio.run(run_async_code_d3eedec4())
    logger.success(format_json())
        logger.debug(chunk.delta, end="", flush=True)


asyncio.run(classify_review_streaming(llm, prompt))

logger.info("\n\n[DONE]", bright=True)