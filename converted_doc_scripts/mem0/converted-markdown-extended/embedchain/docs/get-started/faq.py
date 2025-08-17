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
title: ❓ FAQs
description: 'Collections of all the frequently asked questions'
---
<AccordionGroup>
<Accordion title="Does Embedchain support MLX's Assistant APIs?">
Yes, it does. Please refer to the [MLX Assistant docs page](/examples/openai-assistant).
</Accordion>
<Accordion title="How to use MistralAI language model?">
Use the model provided on huggingface: `mistralai/Mistral-7B-v0.1`
<CodeGroup>
"""
logger.info("title: ❓ FAQs")


os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "hf_your_token"

app = App.from_config("huggingface.yaml")

llm:
  provider: huggingface
  config:
    model: 'mistralai/Mistral-7B-v0.1'
    temperature: 0.5
    max_tokens: 1000
    top_p: 0.5
    stream: false

embedder:
  provider: huggingface
  config:
    model: 'sentence-transformers/all-mpnet-base-v2'

"""
</CodeGroup>
</Accordion>
<Accordion title="How to use ChatGPT 4 turbo model released on MLX DevDay?">
Use the model `gpt-4-turbo` provided my openai.
<CodeGroup>
"""
logger.info("Use the model `gpt-4-turbo` provided my openai.")


# os.environ['OPENAI_API_KEY'] = 'xxx'

app = App.from_config(config_path="gpt4_turbo.yaml")

"""

"""

llm:
  provider: openai
  config:
    model: 'gpt-4-turbo'
    temperature: 0.5
    max_tokens: 1000
    top_p: 1
    stream: false

"""
</CodeGroup>
</Accordion>
<Accordion title="How to use GPT-4 as the LLM model?">
<CodeGroup>
"""


# os.environ['OPENAI_API_KEY'] = 'xxx'

app = App.from_config(config_path="gpt4.yaml")

"""

"""

llm:
  provider: openai
  config:
    model: 'gpt-4'
    temperature: 0.5
    max_tokens: 1000
    top_p: 1
    stream: false

"""
</CodeGroup>
</Accordion>
<Accordion title="I don't have MLX credits. How can I use some open source model?">
<CodeGroup>
"""


app = App.from_config(config_path="opensource.yaml")

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
  config:
    model: 'all-MiniLM-L6-v2'

"""
</CodeGroup>

</Accordion>
<Accordion title="How to stream response while using MLX model in Embedchain?">
You can achieve this by setting `stream` to `true` in the config file.

<CodeGroup>
"""
logger.info("You can achieve this by setting `stream` to `true` in the config file.")

llm:
  provider: openai
  config:
    model: 'llama-3.2-3b-instruct'
    temperature: 0.5
    max_tokens: 1000
    top_p: 1
    stream: true

"""

"""


# os.environ['OPENAI_API_KEY'] = 'sk-xxx'

app = App.from_config(config_path="openai.yaml")

app.add("https://www.forbes.com/profile/elon-musk")

response = app.query("What is the net worth of Elon Musk?")

"""
</CodeGroup>
</Accordion>

<Accordion title="How to persist data across multiple app sessions?">
  Set up the app by adding an `id` in the config file. This keeps the data for future use. You can include this `id` in the yaml config or input it directly in `config` dict.
  ```python app1.py

#   os.environ['OPENAI_API_KEY'] = 'sk-xxx'

  app1 = App.from_config(config={
    "app": {
      "config": {
        "id": "your-app-id",
      }
    }
  })

  app1.add("https://www.forbes.com/profile/elon-musk")

  response = app1.query("What is the net worth of Elon Musk?")
  ```
  ```python app2.py

#   os.environ['OPENAI_API_KEY'] = 'sk-xxx'

  app2 = App.from_config(config={
    "app": {
      "config": {
        # this will persist and load data from app1 session
        "id": "your-app-id",
      }
    }
  })

  response = app2.query("What is the net worth of Elon Musk?")
  ```
</Accordion>
</AccordionGroup>

#### Still have questions?
If docs aren't sufficient, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("# this will persist and load data from app1 session")

logger.info("\n\n[DONE]", bright=True)