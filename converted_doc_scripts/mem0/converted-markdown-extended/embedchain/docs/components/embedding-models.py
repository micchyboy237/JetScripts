from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: 🧩 Embedding models
---

## Overview

Embedchain supports several embedding models from the following providers:

<CardGroup cols={4}>
  <Card title="MLX" href="#openai"></Card>
  <Card title="GoogleAI" href="#google-ai"></Card>
  <Card title="Azure MLX" href="#azure-openai"></Card>
  <Card title="AWS Bedrock" href="#aws-bedrock"></Card>
  <Card title="GPT4All" href="#gpt4all"></Card>
  <Card title="Hugging Face" href="#hugging-face"></Card>
  <Card title="Vertex AI" href="#vertex-ai"></Card>
  <Card title="NVIDIA AI" href="#nvidia-ai"></Card>
  <Card title="Cohere" href="#cohere"></Card>
  <Card title="Ollama" href="#ollama"></Card>
  <Card title="Clarifai" href="#clarifai"></Card>
</CardGroup>

## MLX

# To use MLX embedding function, you have to set the `OPENAI_API_KEY` environment variable. You can obtain the MLX API key from the [MLX Platform](https://platform.openai.com/account/api-keys).

Once you have obtained the key, you can use it like this:

<CodeGroup>
"""
logger.info("## Overview")


# os.environ['OPENAI_API_KEY'] = 'xxx'

app = App.from_config(config_path="config.yaml")

app.add("https://en.wikipedia.org/wiki/MLX")
app.query("What is MLX?")

"""

"""

embedder:
  provider: openai
  config:
    model: 'mxbai-embed-large'

"""
</CodeGroup>

* MLX announced two new embedding models: `mxbai-embed-large` and `text-embedding-3-large`. Embedchain supports both these models. Below you can find YAML config for both:

<CodeGroup>
"""

embedder:
  provider: openai
  config:
    model: 'mxbai-embed-large'

"""

"""

embedder:
  provider: openai
  config:
    model: 'text-embedding-3-large'

"""
</CodeGroup>

## Google AI

To use Google AI embedding function, you have to set the `GOOGLE_API_KEY` environment variable. You can obtain the Google API key from the [Google Maker Suite](https://makersuite.google.com/app/apikey)

<CodeGroup>
"""
logger.info("## Google AI")


os.environ["GOOGLE_API_KEY"] = "xxx"

app = App.from_config(config_path="config.yaml")

"""

"""

embedder:
  provider: google
  config:
    model: 'models/embedding-001'
    task_type: "retrieval_document"
    title: "Embeddings for Embedchain"

"""
</CodeGroup>
<br/>
<Note>
For more details regarding the Google AI embedding model, please refer to the [Google AI documentation](https://ai.google.dev/tutorials/python_quickstart#use_embeddings).
</Note>

## AWS Bedrock

To use AWS Bedrock embedding function, you have to set the AWS environment variable.

<CodeGroup>
"""
logger.info("## AWS Bedrock")


os.environ["AWS_ACCESS_KEY_ID"] = "xxx"
os.environ["AWS_SECRET_ACCESS_KEY"] = "xxx"
os.environ["AWS_REGION"] = "us-west-2"

app = App.from_config(config_path="config.yaml")

"""

"""

embedder:
  provider: aws_bedrock
  config:
    model: 'amazon.titan-embed-text-v2:0'
    vector_dimension: 1024
    task_type: "retrieval_document"
    title: "Embeddings for Embedchain"

"""
</CodeGroup>
<br/>
<Note>
For more details regarding the AWS Bedrock embedding model, please refer to the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html).
</Note>

## Azure MLX

To use Azure MLX embedding model, you have to set some of the azure openai related environment variables as given in the code block below:

<CodeGroup>
"""
logger.info("## Azure MLX")


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://xxx.openai.azure.com/"
# os.environ["AZURE_OPENAI_API_KEY"] = "xxx"
os.environ["OPENAI_API_VERSION"] = "xxx"

app = App.from_config(config_path="config.yaml")

"""

"""

llm:
  provider: azure_openai
  config:
    model: gpt-35-turbo
    deployment_name: your_llm_deployment_name
    temperature: 0.5
    max_tokens: 1000
    top_p: 1
    stream: false

embedder:
  provider: azure_openai
  config:
    model: text-embedding-ada-002
    deployment_name: you_embedding_model_deployment_name

"""
</CodeGroup>

You can find the list of models and deployment name on the [Azure MLX Platform](https://oai.azure.com/portal).

## GPT4ALL

GPT4All supports generating high quality embeddings of arbitrary length documents of text using a CPU optimized contrastively trained Sentence Transformer.

<CodeGroup>
"""
logger.info("## GPT4ALL")


app = App.from_config(config_path="config.yaml")

"""

"""

llm:
  provider: gpt4all
  config:
    model: 'orca-mini-3b-gguf2-q4_0.gguf'
    temperature: 0.5
    max_tokens: 1000
    top_p: 1
    stream: false

embedder:
  provider: gpt4all

"""
</CodeGroup>

## Hugging Face

Hugging Face supports generating embeddings of arbitrary length documents of text using Sentence Transformer library. Example of how to generate embeddings using hugging face is given below:

<CodeGroup>
"""
logger.info("## Hugging Face")


app = App.from_config(config_path="config.yaml")

"""

"""

llm:
  provider: huggingface
  config:
    model: 'google/flan-t5-xxl'
    temperature: 0.5
    max_tokens: 1000
    top_p: 0.5
    stream: false

embedder:
  provider: huggingface
  config:
    model: 'sentence-transformers/all-mpnet-base-v2'
    model_kwargs:
        trust_remote_code: True # Only use if you trust your embedder

"""
</CodeGroup>

## Vertex AI

Embedchain supports Google's VertexAI embeddings model through a simple interface. You just have to pass the `model_name` in the config yaml and it would work out of the box.

<CodeGroup>
"""
logger.info("## Vertex AI")


app = App.from_config(config_path="config.yaml")

"""

"""

llm:
  provider: vertexai
  config:
    model: 'chat-bison'
    temperature: 0.5
    top_p: 0.5

embedder:
  provider: vertexai
  config:
    model: 'textembedding-gecko'

"""
</CodeGroup>

## NVIDIA AI

[NVIDIA AI Foundation Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) let you quickly use NVIDIA's AI models, such as Mixtral 8x7B, Llama 2 etc, through our API. These models are available in the [NVIDIA NGC catalog](https://catalog.ngc.nvidia.com/ai-foundation-models), fully optimized and ready to use on NVIDIA's AI platform. They are designed for high speed and easy customization, ensuring smooth performance on any accelerated setup.


### Usage

In order to use embedding models and LLMs from NVIDIA AI, create an account on [NVIDIA NGC Service](https://catalog.ngc.nvidia.com/).

Generate an API key from their dashboard. Set the API key as `NVIDIA_API_KEY` environment variable. Note that the `NVIDIA_API_KEY` will start with `nvapi-`.

Below is an example of how to use LLM model and embedding model from NVIDIA AI:

<CodeGroup>
"""
logger.info("## NVIDIA AI")


os.environ['NVIDIA_API_KEY'] = 'nvapi-xxxx'

config = {
    "app": {
        "config": {
            "id": "my-app",
        },
    },
    "llm": {
        "provider": "nvidia",
        "config": {
            "model": "nemotron_steerlm_8b",
        },
    },
    "embedder": {
        "provider": "nvidia",
        "config": {
            "model": "nvolveqa_40k",
            "vector_dimension": 1024,
        },
    },
}

app = App.from_config(config=config)

app.add("https://www.forbes.com/profile/elon-musk")
answer = app.query("What is the net worth of Elon Musk today?")

"""
</CodeGroup>


## Cohere

To use embedding models and LLMs from COHERE, create an account on [COHERE](https://dashboard.cohere.com/welcome/login?redirect_uri=%2Fapi-keys).

Generate an API key from their dashboard. Set the API key as `COHERE_API_KEY` environment variable.

Once you have obtained the key, you can use it like this:

<CodeGroup>
"""
logger.info("## Cohere")


os.environ['COHERE_API_KEY'] = 'xxx'

app = App.from_config(config_path="config.yaml")

"""

"""

embedder:
  provider: cohere
  config:
    model: 'embed-english-light-v3.0'

"""
</CodeGroup>

* Cohere has few embedding models: `embed-english-v3.0`, `embed-multilingual-v3.0`, `embed-multilingual-light-v3.0`, `embed-english-v2.0`, `embed-english-light-v2.0` and `embed-multilingual-v2.0`. Embedchain supports all these models. Below you can find YAML config for all:

<CodeGroup>
"""

embedder:
  provider: cohere
  config:
    model: 'embed-english-v3.0'
    vector_dimension: 1024

"""

"""

embedder:
  provider: cohere
  config:
    model: 'embed-multilingual-v3.0'
    vector_dimension: 1024

"""

"""

embedder:
  provider: cohere
  config:
    model: 'embed-multilingual-light-v3.0'
    vector_dimension: 384

"""

"""

embedder:
  provider: cohere
  config:
    model: 'embed-english-v2.0'
    vector_dimension: 4096

"""

"""

embedder:
  provider: cohere
  config:
    model: 'embed-english-light-v2.0'
    vector_dimension: 1024

"""

"""

embedder:
  provider: cohere
  config:
    model: 'embed-multilingual-v2.0'
    vector_dimension: 768

"""
</CodeGroup>

## Ollama

Ollama enables the use of embedding models, allowing you to generate high-quality embeddings directly on your local machine. Make sure to install [Ollama](https://ollama.com/download) and keep it running before using the embedding model.

You can find the list of models at [Ollama Embedding Models](https://ollama.com/blog/embedding-models).

Below is an example of how to use embedding model Ollama:

<CodeGroup>
"""
logger.info("## Ollama")


app = App.from_config(config_path="config.yaml")

"""

"""

embedder:
  provider: ollama
  config:
    model: 'all-minilm:latest'

"""
</CodeGroup>

## Clarifai

Install related dependencies using the following command:
"""
logger.info("## Clarifai")

pip install --upgrade 'embedchain[clarifai]'

"""
set the `CLARIFAI_PAT` as environment variable which you can find in the [security page](https://clarifai.com/settings/security). Optionally you can also pass the PAT key as parameters in LLM/Embedder class.

Now you are all set with exploring Embedchain.

<CodeGroup>
"""
logger.info("set the `CLARIFAI_PAT` as environment variable which you can find in the [security page](https://clarifai.com/settings/security). Optionally you can also pass the PAT key as parameters in LLM/Embedder class.")


os.environ["CLARIFAI_PAT"] = "XXX"

app = App.from_config(config_path="config.yaml")

app.add("https://www.forbes.com/profile/elon-musk")

response = app.query("what college degrees does elon musk have?")

"""
Head to [Clarifai Platform](https://clarifai.com/explore/models?page=1&perPage=24&filterData=%5B%7B%22field%22%3A%22output_fields%22%2C%22value%22%3A%5B%22embeddings%22%5D%7D%5D) to explore all the State of the Art embedding models available to use.
For passing LLM model inference parameters use `model_kwargs` argument in the config file. Also you can use `api_key` argument to pass `CLARIFAI_PAT` in the config.
"""
logger.info("Head to [Clarifai Platform](https://clarifai.com/explore/models?page=1&perPage=24&filterData=%5B%7B%22field%22%3A%22output_fields%22%2C%22value%22%3A%5B%22embeddings%22%5D%7D%5D) to explore all the State of the Art embedding models available to use.")

llm:
 provider: clarifai
 config:
   model: "https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct"
   model_kwargs:
     temperature: 0.5
     max_tokens: 1000
embedder:
 provider: clarifai
 config:
   model: "https://clarifai.com/clarifai/main/models/BAAI-bge-base-en-v15"

"""
</CodeGroup>
"""

logger.info("\n\n[DONE]", bright=True)