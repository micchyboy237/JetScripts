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
title: '⚡ Quickstart'
description: '💡 Create an AI app on your own data in a minute'
---

## Installation

First install the Python package:
"""
logger.info("## Installation")

pip install embedchain

"""
Once you have installed the package, depending upon your preference you can either use:

<CardGroup cols={2}>
  <Card title="Open Source Models" icon="osi" href="#open-source-models">
  This includes Open source LLMs like Mistral, Llama, etc.<br/>
  Free to use, and runs locally on your machine.
  </Card>
  <Card title="Paid Models" icon="dollar-sign" href="#paid-models" color="#4A154B">
    This includes paid LLMs like GPT 4, Claude, etc.<br/>
    Cost money and are accessible via an API.
  </Card>
</CardGroup>

## Open Source Models

This section gives a quickstart example of using Mistral as the Open source LLM and Sentence transformers as the Open source embedding model. These models are free and run mostly on your local machine.

We are using Mistral hosted at Hugging Face, so will you need a Hugging Face token to run this example. Its *free* and you can create one [here](https://huggingface.co/docs/hub/security-tokens).

<CodeGroup>
"""
logger.info("## Open Source Models")

os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "hf_xxxx"


config = {
  'llm': {
    'provider': 'huggingface',
    'config': {
      'model': 'mistralai/Mistral-7B-Instruct-v0.2',
      'top_p': 0.5
    }
  },
  'embedder': {
    'provider': 'huggingface',
    'config': {
      'model': 'sentence-transformers/all-mpnet-base-v2'
    }
  }
}
app = App.from_config(config=config)
app.add("https://www.forbes.com/profile/elon-musk")
app.add("https://en.wikipedia.org/wiki/Elon_Musk")
app.query("What is the net worth of Elon Musk today?")

"""
</CodeGroup>

## Paid Models

In this section, we will use both LLM and embedding model from MLX.
"""
logger.info("## Paid Models")


# os.environ["OPENAI_API_KEY"] = "sk-xxxx"

app = App()
app.add("https://www.forbes.com/profile/elon-musk")
app.add("https://en.wikipedia.org/wiki/Elon_Musk")
app.query("What is the net worth of Elon Musk today?")

"""
# Next Steps

Now that you have created your first app, you can follow any of the links:

* [Introduction](/get-started/introduction)
* [Customization](/components/introduction)
* [Use cases](/use-cases/introduction)
* [Deployment](/get-started/deployment)
"""
logger.info("# Next Steps")

logger.info("\n\n[DONE]", bright=True)