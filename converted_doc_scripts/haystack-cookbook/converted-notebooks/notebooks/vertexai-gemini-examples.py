# from google.colab import auth
from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack.dataclasses import ImageContent
from haystack.tools import tool
# from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from jet.adapters.haystack.ollama_chat_generator import OllamaChatGenerator
from jet.logger import logger
from typing import Annotated
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
# Haystack 💙 Google Gemini

*by Tuana Celik: [Twitter](https://twitter.com/tuanacelik), [LinkedIn](https://www.linkedin.com/in/tuanacelik/), Tilde Thurium: [Twitter](https://twitter.com/annthurium), [LinkedIn](https://www.linkedin.com/in/annthurium/) and Silvano Cerza: [LinkedIn](https://www.linkedin.com/in/silvanocerza/)*

This is a notebook showing how you can use Gemini + Vertex AI with Haystack.

To use Gemini models on the Gemini Developer API with Haystack, check out our [documentation](https://docs.haystack.deepset.ai/docs/googlegenaichatgenerator).



Gemini is Google's newest model. You can read more about its capabilities [here](https://deepmind.google/technologies/gemini/#capabilities).

## Install dependencies

As a prerequisite, you need to have a Google Cloud Project set up that has access to Vertex AI and Gemini.

Useful resources:
- [Vertex AI quick start](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)
- [Gemini API in Vertex AI quickstart](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart)

Following that, you'll only need to authenticate yourself in this Colab.

First thing first we need to install our dependencies including [Google Gen AI](https://haystack.deepset.ai/integrations/google-genai) integration:
"""
logger.info("# Haystack 💙 Google Gemini")

# ! pip install haystack-ai google-genai-haystack trafilatura

"""
Let's login using Application Default Credentials (ADCs). For more info see the [official documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).
"""
logger.info("Let's login using Application Default Credentials (ADCs). For more info see the [official documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).")


# auth.authenticate_user()

"""
Remember to set the `project_id` variable to a valid project ID that you have enough authorization to use for Gemini.
We're going to use this one throughout the example!

To find your project ID you can find it in the [GCP resource manager](https://console.cloud.google.com/cloud-resource-manager) or locally by running `gcloud projects list` in your terminal. For more info on the gcloud CLI see the [official documentation](https://cloud.google.com/cli).
"""
logger.info("Remember to set the `project_id` variable to a valid project ID that you have enough authorization to use for Gemini.")

project_id = input("Enter your project ID:")

"""
## Use `gemini-2.5-flash`

### Answer Questions

Now that we setup everything we can create an instance of our [`GoogleGenAIChatGenerator`](https://docs.haystack.deepset.ai/docs/googlegenaichatgenerator). This component supports both Gemini and Vertex AI. For this demo, we will set `api="vertex"`, and pass our project_id as vertex_ai_project.
"""
logger.info("## Use `gemini-2.5-flash`")


gemini = OllamaChatGenerator(model="qwen3:4b-q4_K_M")

"""
Let's start by asking something simple.

This component expects a list of `ChatMessage` as input to the `run()` method. You can pass text or function calls through the messages.
"""
logger.info("Let's start by asking something simple.")


messages = [ChatMessage.from_user("What is the most interesting thing you know?")]
result = gemini.run(messages = messages)
for answer in result["replies"]:
    logger.debug(answer.text)

"""
### Answer Questions about Images

Let's try something a bit different! `gemini-2.5-flash` can also work with images, let's see if we can have it answer questions about some robots 👇

We're going to download some images for this example. 🤖
"""
logger.info("### Answer Questions about Images")


urls = [
    "https://upload.wikimedia.org/wikipedia/en/5/5c/C-3PO_droid.png",
    "https://platform.theverge.com/wp-content/uploads/sites/2/chorus/assets/4658579/terminator_endoskeleton_1020.jpg",
    "https://upload.wikimedia.org/wikipedia/en/3/39/R2-D2_Droid.png",
]

images = [ImageContent.from_url(url) for url in urls]

messages = [ChatMessage.from_user(content_parts=["What can you tell me about these robots? Be short and graceful.", *images])]
result = gemini.run(messages = messages)
for answer in result["replies"]:
    logger.debug(answer.text)

"""
## Function Calling with `gemini-2.5-flash`

With `gemini-2.5-flash`, we can also use function calling!
So let's see how we can do that 👇

Let's see if we can build a system that can run a `get_current_weather` function, based on a question asked in natural language.

First we create our function definition and tool (learn more about [Tools](https://docs.haystack.deepset.ai/docs/tool) in the docs).

For demonstration purposes, we're simply creating a `get_current_weather` function that returns an object which will _always_ tell us it's 'Sunny, and 21.8 degrees'... If it's Celsius, that's a good day! ☀️
"""
logger.info("## Function Calling with `gemini-2.5-flash`")


@tool
def get_current_weather(
    location: Annotated[str, "The city for which to get the weather, e.g. 'San Francisco'"] = "Munich",
    unit: Annotated[str, "The unit for the temperature, e.g. 'celsius'"] = "celsius",
):
  return {"weather": "sunny", "temperature": 21.8, "unit": unit}

user_message = [ChatMessage.from_user("What is the temperature in celsius in Berlin?")]
replies = gemini.run(messages=user_message, tools=[get_current_weather])["replies"]
logger.debug(replies)

"""
Look at that! We go a message with some interesting information now.
We can use that information to call a real function locally.

Let's do exactly that and pass the result back to Gemini.
"""
logger.info("Look at that! We go a message with some interesting information now.")

tool_invoker = ToolInvoker(tools=[get_current_weather])
tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
logger.debug(tool_messages)

messages = user_message + replies + tool_messages

res = gemini.run(messages = messages)
logger.debug(res["replies"][0].text)

"""
Seems like the weather is nice and sunny, remember to put on your sunglasses. 😎

## Build a full Retrieval-Augmented Generation Pipeline with `gemini-2.5-flash`

As a final exercise, let's add the `GoogleGenAIChatGenerator` to a full RAG pipeline. In the example below, we are building a RAG pipeline that does question answering on the web, using `gemini-2.5-flash`
"""
logger.info("## Build a full Retrieval-Augmented Generation Pipeline with `gemini-2.5-flash`")


fetcher = LinkContentFetcher()
converter = HTMLToDocument()
document_splitter = DocumentSplitter(split_by="word", split_length=50)
similarity_ranker = SentenceTransformersSimilarityRanker(top_k=3)
gemini = GoogleGenAIChatGenerator(model="gemini-2.5-flash", api="vertex", vertex_ai_project=project_id, vertex_ai_location="europe-west1")

prompt_template = [ChatMessage.from_user("""
According to these documents:

{% for doc in documents %}
  {{ doc.content }}
{% endfor %}

Answer the given question: {{question}}
Answer:
""")]
prompt_builder = ChatPromptBuilder(template=prompt_template)

pipeline = Pipeline()
pipeline.add_component("fetcher", fetcher)
pipeline.add_component("converter", converter)
pipeline.add_component("splitter", document_splitter)
pipeline.add_component("ranker", similarity_ranker)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("gemini", gemini)

pipeline.connect("fetcher.streams", "converter.sources")
pipeline.connect("converter.documents", "splitter.documents")
pipeline.connect("splitter.documents", "ranker.documents")
pipeline.connect("ranker.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "gemini")

"""
Let's try asking Gemini to tell us about Haystack and how to use it.
"""
logger.info("Let's try asking Gemini to tell us about Haystack and how to use it.")

question = "What do graphs have to do with Haystack?"
result = pipeline.run({"prompt_builder": {"question": question},
                   "ranker": {"query": question},
                   "fetcher": {"urls": ["https://haystack.deepset.ai/blog/introducing-haystack-2-beta-and-advent"]}})

for message in result["gemini"]["replies"]:
  logger.debug(message.text)

"""
Now you've seen some of what Gemini can do, as well as how to integrate it with Haystack. If you want to learn more, check out the Haystack [docs](https://docs.haystack.deepset.ai/docs) or [tutorials](https://haystack.deepset.ai/tutorials)
"""
logger.info("Now you've seen some of what Gemini can do, as well as how to integrate it with Haystack. If you want to learn more, check out the Haystack [docs](https://docs.haystack.deepset.ai/docs) or [tutorials](https://haystack.deepset.ai/tutorials)")

logger.info("\n\n[DONE]", bright=True)