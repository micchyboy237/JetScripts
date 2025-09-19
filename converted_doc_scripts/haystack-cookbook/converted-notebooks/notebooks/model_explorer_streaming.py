from dataclasses import dataclass
from google.colab import userdata
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.generators import OllamaFunctionCallingAdapterGenerator
from haystack.utils import Secret
from haystack_integrations.components.generators.cohere import CohereGenerator
from jet.logger import CustomLogger
import ipywidgets as widgets
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
## Streaming model explorer for Haystack

*notebook by Tilde Thurium:
 [Mastodon](https://tech.lgbt/@annthurium) || [Twitter](https://twitter.com/annthurium) || [LinkedIn](https://www.linkedin.com/in/annthurium/)*

*Problem*: there are so many LLMs these days! Which model is the best for my use case?

This notebook uses [Haystack](https://docs.haystack.deepset.ai/docs/intro) to compare the results of sending the same prompt to several different models.

This is a very basic demo where you can only compare a few models that support streaming responses. I'd like to support more models in the future, so watch this space for updates.


### Models

Haystack's [OllamaFunctionCallingAdapterGenerator](https://docs.haystack.deepset.ai/docs/openaigenerator) and [CohereGenerator](https://docs.haystack.deepset.ai/docs/coheregenerator) support streaming out of the box.

The other models use the [HuggingFaceAPIGenerator](https://docs.haystack.deepset.ai/docs/huggingfaceapigenerator).

### Prerequisites

- You need [HuggingFace](https://huggingface.co/docs/hub/security-tokens), [Cohere](https://docs.cohere.com/docs/connector-authentication), and [OllamaFunctionCalling](https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key) API keys. Save them as secrets in your Colab. Click on the key icon in the left menu or [see detailed instructions here](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75).
- To use Mistral-7B-v0.1, you should also accept Mistral conditions here: https://huggingface.co/mistralai/Mistral-7B-v0.1
"""
logger.info("## Streaming model explorer for Haystack")

# !pip install -U haystack-ai cohere-haystack "huggingface_hub>=0.22.0"

"""
In order for `userdata.get` to work, these keys need to be saved as secrets in your Colab. Click on the key icon in the left menu or [see detailed instructions here](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75).
"""
logger.info("In order for `userdata.get` to work, these keys need to be saved as secrets in your Colab. Click on the key icon in the left menu or [see detailed instructions here](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75).")


# open_ai_generator = OllamaFunctionCallingAdapterGenerator(api_key=Secret.from_token(userdata.get('OPENAI_API_KEY')))

cohere_generator = CohereGenerator(api_key=Secret.from_token(userdata.get('COHERE_API_KEY')))

hf_generator = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "mistralai/Mistral-7B-Instruct-v0.1"},
    token=Secret.from_token(userdata.get('HF_API_KEY')))


hf_generator_2 = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "tiiuae/falcon-7b-instruct"},
    token=Secret.from_token(userdata.get('HF_API_KEY')))


hf_generator_3 = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "bigscience/bloom"},
    token=Secret.from_token(userdata.get('HF_API_KEY')))

MODELS = [open_ai_generator, cohere_generator, hf_generator, hf_generator_2, hf_generator_3]

"""
The `AppendToken` dataclass formats the output so that the model name is printed, and the text follows in chunks of 5 tokens.
"""
logger.info("The `AppendToken` dataclass formats the output so that the model name is printed, and the text follows in chunks of 5 tokens.")


def output():...

@dataclass
class AppendToken:
  output: widgets.Output
  chunks = []
  chunk_size = 5

  def __call__(self, chunk):
      with self.output:
        text = getattr(chunk, 'content', '')
        self.chunks.append(text)
        if len(self.chunks) == self.chunk_size:
          output_string = ' '.join(self.chunks)
          self.output.append_display_data(output_string)
          self.chunks.clear()

def multiprompt(prompt, models=MODELS):
  outputs = [widgets.Output(layout={'border': '1px solid black'}) for _ in models]
  display(widgets.HBox(children=outputs))

  for i, model in enumerate(models):
    model_name = getattr(model, 'model', '')
    outputs[i].append_display_data(f'Model name: {model_name}')
    model.streaming_callback = AppendToken(outputs[i])
    model.run(prompt)

multiprompt("Tell me a cyberpunk story about a black cat.")

"""
This was a very silly example prompt. If you found this demo useful, let me know the kinds of prompts you tested it with!

 [Mastodon](https://tech.lgbt/@annthurium) || [Twitter](https://twitter.com/annthurium) || [LinkedIn](https://www.linkedin.com/in/annthurium/)

Thanks for following along.
"""
logger.info("This was a very silly example prompt. If you found this demo useful, let me know the kinds of prompts you tested it with!")

logger.info("\n\n[DONE]", bright=True)