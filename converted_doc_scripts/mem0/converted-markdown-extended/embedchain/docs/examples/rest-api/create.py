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
openapi: post /create
---

<RequestExample>
"""
logger.info("openapi: post /create")

curl --request POST \
  --url http://localhost:8080/create?app_id=app1 \
  -F "config=@/path/to/config.yaml"

"""
</RequestExample>

<ResponseExample>
"""

{ "response": "App created successfully. App ID: app1" }

"""
</ResponseExample>

By default we will use the opensource **gpt4all** model to get started. You can also specify your own config by uploading a config YAML file.

For example, create a `config.yaml` file (adjust according to your requirements):
"""
logger.info("By default we will use the opensource **gpt4all** model to get started. You can also specify your own config by uploading a config YAML file.")

app:
  config:
    id: "default-app"

llm:
  provider: openai
  config:
    model: "llama-3.2-3b-instruct"
    temperature: 0.5
    max_tokens: 1000
    top_p: 1
    stream: false
    prompt: |
      Use the following pieces of context to answer the query at the end.
      If you don't know the answer, just say that you don't know, don't try to make up an answer.

      $context

      Query: $query

      Helpful Answer:

vectordb:
  provider: chroma
  config:
    collection_name: "rest-api-app"
    dir: db
    allow_reset: true

embedder:
  provider: openai
  config:
    model: "text-embedding-ada-002"

"""
To learn more about custom configurations, check out the [custom configurations docs](https://docs.embedchain.ai/advanced/configuration). To explore more examples of config yamls for embedchain, visit [embedchain/configs](https://github.com/embedchain/embedchain/tree/main/configs).

Now, you can upload this config file in the request body.

For example,
"""
logger.info("To learn more about custom configurations, check out the [custom configurations docs](https://docs.embedchain.ai/advanced/configuration). To explore more examples of config yamls for embedchain, visit [embedchain/configs](https://github.com/embedchain/embedchain/tree/main/configs).")

curl --request POST \
  --url http://localhost:8080/create?app_id=my-app \
  -F "config=@/path/to/config.yaml"

"""
**Note:** To use custom models, an **API key** might be required. Refer to the table below to determine the necessary API key for your provider.

| Keys                       | Providers                      |
| -------------------------- | ------------------------------ |
# | `OPENAI_API_KEY `          | MLX, Azure MLX, Jina etc |
| `OPENAI_API_TYPE`          | Azure MLX                   |
| `OPENAI_API_BASE`          | Azure MLX                   |
| `OPENAI_API_VERSION`       | Azure MLX                   |
| `COHERE_API_KEY`           | Cohere                         |
| `TOGETHER_API_KEY`         | Together                       |
# | `ANTHROPIC_API_KEY`        | Anthropic                      |
| `JINACHAT_API_KEY`         | Jina                           |
| `HUGGINGFACE_ACCESS_TOKEN` | Huggingface                    |
| `REPLICATE_API_TOKEN`      | LLAMA2                         |

To add env variables, you can simply run the docker command with the `-e` flag.

For example,
"""
logger.info("To add env variables, you can simply run the docker command with the `-e` flag.")

# docker run --name embedchain -p 8080:8080 -e OPENAI_API_KEY=<YOUR_OPENAI_API_KEY> embedchain/rest-api:latest

logger.info("\n\n[DONE]", bright=True)