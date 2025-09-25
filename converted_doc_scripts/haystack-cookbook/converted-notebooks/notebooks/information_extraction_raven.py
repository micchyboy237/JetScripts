from IPython.display import display, HTML
from haystack import Document
from haystack import Pipeline
from haystack import component
from haystack.components.builders import PromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.preprocessors import DocumentCleaner
from jet.logger import logger
from typing import List, Optional
import ast
import os
import re
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
# 🧪🐦‍⬛ Needle in a Jungle - Information Extraction via LLMs

<img src="https://haystack.deepset.ai/images/haystack-ogimage.png" width="430" style="display:inline;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://huggingface.co/Nexusflow/NexusRaven-V2-13B/resolve/main/NexusRaven.png" width="250" style="display:inline;">

*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*

In this experiment, we will use Large Language Models to perform Information Extraction from textual data.

🎯 Goal: create an application that, given a URL and a specific structure provided by the user, extracts information from the source.

The "**function calling**" capabilities of Ollama models unlock this task: the user can describe a structure, by defining a mock up function with all its typed and specific parameters. The LLM will prepare the data in this specific form and send it back to the user.

A nice example of using Ollama Function Calling for information extraction is this [gist by Kyle McDonald](https://gist.github.com/kylemcdonald/dbac21de2d7855633689f5526225154c).

**What is changing now is that open models such as NexusRaven are emerging, with function calling capabilities...**

*This is an improved version of an older experiment, using Gorilla Open Functions*


**Stack**
- **[NexusRaven](https://huggingface.co/Nexusflow/NexusRaven-V2-13B)**: an open-source and commercially viable function calling model that surpasses the state-of-the-art in function calling capabilities.
- **[Haystack](https://haystack.deepset.ai/)**: open-source LLM orchestration framework that streamlines the development of your LLM applications.

## Install the dependencies
"""
logger.info("# 🧪🐦‍⬛ Needle in a Jungle - Information Extraction via LLMs")

# %%capture
# ! pip install haystack-ai "huggingface_hub>=0.22.0" trafilatura

"""
## Load and try the model
We use the [`HuggingFaceAPIGenerator`](https://docs.haystack.deepset.ai/docs/huggingfaceapigenerator), which allows to use models hosted on Hugging Face endpoints.
In particular, we use a paid endpoint kindly provided by Nexusflow to test the LLM.

Alternative inference options:
- load the model on Colab using the HuggingFaceLocalGenerator. This is a bit impractical because the model is quite big (13B parameters) and even using quantization, there would be few GPU resources left for inference.
- local inference via TGI or vLLM: this is a good option if you have GPU avalaible.
- local inference via Ollama/llama.cpp: this is suitable for machines with few resources and no GPU. Keep in mind that in this case a quantized GGUF version of the model would be used, with lower quality than the original model.
"""
logger.info("## Load and try the model")


generator = HuggingFaceAPIGenerator(
    api_type="inference_endpoints",
    api_params={"url": "http://38.142.9.20:10240"},
    stop_words=["<bot_end>"],
    generation_kwargs={"temperature":0.001,
                    "do_sample" : False,
                    "max_new_tokens" : 1000})

"""
To understand how to prompt the model, give a look at the [Prompting notebook](https://github.com/nexusflowai/NexusRaven-V2/blob/master/How-To-Prompt.ipynb).
Later we will see how to better organize the prompt for our purpose.
"""
logger.info("To understand how to prompt the model, give a look at the [Prompting notebook](https://github.com/nexusflowai/NexusRaven-V2/blob/master/How-To-Prompt.ipynb).")

prompt='''
Function:
def get_weather_data(coordinates):
    """
    Fetches weather data from the Open-Meteo API for the given latitude and longitude.

    Args:
    coordinates (tuple): The latitude of the location.

    Returns:
    float: The current temperature in the coordinates you've asked for
    """

Function:
def get_coordinates_from_city(city_name):
    """
    Fetches the latitude and longitude of a given city name using the Maps.co Geocoding API.

    Args:
    city_name (str): The name of the city.

    Returns:
    tuple: The latitude and longitude of the city.
    """

User Query: What's the weather like in Seattle right now?<human_end>

'''

logger.debug(generator.run(prompt=prompt))

"""
All good! ✅

## Prompt template and Prompt Builder

- The Prompt template to apply is model specific. In our case, we customize a bit the original prompt which is available on [Prompting notebook](https://github.com/nexusflowai/NexusRaven-V2/blob/master/How-To-Prompt.ipynb).
- In Haystack, the prompt template is rendered using the [Prompt Builder component](https://docs.haystack.deepset.ai/docs/promptbuilder).
"""
logger.info("## Prompt template and Prompt Builder")


prompt_template = '''
Function:
{{function}}
User Query: Save data from the provided text. START TEXT:{{docs[0].content|replace("\n"," ")|truncate(10000)}} END TEXT
<human_end>'''

prompt_builder = PromptBuilder(template=prompt_template)

logger.debug(prompt_builder.run(docs=[Document(content="my fake document")], function="my fake function definition"))

"""
Nice ✅

## Other Components

The following Components are required for the Pipeline we are about to create. However, they are simple and there is no need to customize and try them, so we can instantiate them directly during Pipeline creation.

- [LinkContentFetcher](https://docs.haystack.deepset.ai/docs/linkcontentfetcher): fetches the contents of the URLs you give it and returns a list of content streams.
- [HTMLToDocument](https://docs.haystack.deepset.ai/docs/htmltodocument): converts HTML files to Documents.
- [DocumentCleaner](https://docs.haystack.deepset.ai/docs/documentcleaner): make text documents more readable.

## Define a custom Component to parse and visualize the result

The output of the model generation is a function call string.

We are going to create a simple Haystack Component to appropriately parse this string and create a nice HTML visualization.

For more information on Creating custom Components, see the [docs](https://docs.haystack.deepset.ai/docs/custom-components).
"""
logger.info("## Other Components")


def val_to_color(val):
  """
  Helper function to return a color based on the type/value of a variable
  """
  if isinstance(val, list):
    return "#FFFEE0"
  if val is True:
    return "#90EE90"
  if val is False:
    return "#FFCCCB"
  return ""

@component
class FunctionCallParser:
  """
  A component that parses the function call string and creates a HTML visualization
  """
  @component.output_types(html_visualization=str)
  def run(self, replies:List[str]):

    logger.debug(replies)

    func_call_str = replies[0].replace("Call:", "").strip()

    func_call_str=func_call_str.replace("'=","=")
    func_call_str=re.sub("'([a-z]+)=", "\g<1>=", func_call_str)

    func_call=ast.parse(func_call_str).body[0].value
    kwargs = {arg.arg: ast.literal_eval(arg.value) for arg in func_call.keywords}

    html_content = '<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">'
    for key, value in kwargs.items():
        html_content += f'<p><span style="font-family: Cursive; font-size: 30px;">{key}:</span>'
        html_content += f'&emsp;<span style="background-color:{val_to_color(value)}; font-family: Cursive; font-size: 20px;">{value}</span></p>'
    html_content += '</div>'

    return {"html_visualization": html_content}

"""
## Create an Information Extraction Pipeline

To combine the Components in an appropriate and reproducible way, we resort to Haystack Pipelines.
The syntax should be easily understood. You can find more infomation [in the docs](https://docs.haystack.deepset.ai/docs/pipelines).

This pipeline will extract the information from the given URL following the provided structure.
"""
logger.info("## Create an Information Extraction Pipeline")


pipe = Pipeline()

pipe.add_component("fetcher", LinkContentFetcher())
pipe.add_component("converter", HTMLToDocument(extractor_type="DefaultExtractor"))
pipe.add_component("cleaner", DocumentCleaner())
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("generator", generator)
pipe.add_component("parser", FunctionCallParser())

pipe.connect("fetcher", "converter")
pipe.connect("converter", "cleaner")
pipe.connect("cleaner.documents", "prompt_builder.docs")
pipe.connect("prompt_builder", "generator")
pipe.connect("generator", "parser")

"""
Now we create an `extract` function that wraps the Pipeline and displays the result in the HTML format.
This will accept:
- a `function` dict, containing the structure definition of the information we want to extract
- a `url`, to use as data source
"""
logger.info("Now we create an `extract` function that wraps the Pipeline and displays the result in the HTML format.")


def extract(function:str, url:str) -> dict:
  if not function:
    raise ValueError("function definition is needed")
  if not url:
    raise ValueError("URL is needed")

  data_for_pipeline = {"fetcher":{"urls":[url]},
                       "prompt_builder":{"function":function}}

  html_visualization = pipe.run(data=data_for_pipeline)['parser']['html_visualization']
  display(HTML(html_visualization))

"""
## 🕹️ Try our application!

Let's first define the structure to extract.

We are going to parse some news articles about animals... 🦆🐻🦌
"""
logger.info("## 🕹️ Try our application!")

function = '''def save_data(about_animals: bool, about_ai: bool, habitat:List[string], predators:List[string], diet:List[string]):
    """
    Save data extracted from source text

    Args:
    about_animals (bool): Is the article about animals?
    about_ai (bool): Is the article about artificial intelligence?
    habitat (List[string]): List of places where the animal lives
    predators (List[string]): What are the animals that threaten them?
    diet (List[string]): What does the animal eat?
    """'''

"""
Let's start with an article about **Capybaras**
"""
logger.info("Let's start with an article about **Capybaras**")

extract(function=function, url="https://www.rainforest-alliance.org/species/capybara/")

"""
Now let's try with an article about the **Andean cock of the rock**
"""
logger.info("Now let's try with an article about the **Andean cock of the rock**")

extract(function=function, url="https://www.rainforest-alliance.org/species/cock-rock/")

"""
Now, the **Yucatan Deer**!
"""
logger.info("Now, the **Yucatan Deer**!")

extract(function=function, url="https://www.rainforest-alliance.org/species/yucatan-deer/")

"""
A completely different example, about AI...
"""
logger.info("A completely different example, about AI...")

function='''def save_data(people:List[string], companies:List[string], summary:string, topics:List[string], about_animals: bool, about_ai: bool):
    """
    Save data extracted from source text

    Args:
    people (List[string]): List of the mentioned people
    companies (List[string]): List of the mentioned companies.
    summary (string): Summarize briefly what happened in one sentence of max 15 words.
    topics (List[string]): what are the five most important topics?
    about_animals (bool): Is the article about animals?
    about_ai (bool): Is the article about artificial intelligence?
    """'''

extract(function=function, url="https://www.theverge.com/2023/11/22/23967223/sam-altman-returns-ceo-open-ai")

extract(function=function, url="https://www.theguardian.com/business/2023/dec/30/sam-bankman-fried-will-not-face-second-trial-after-multibillion-dollar-crypto-conviction")

extract(function=function, url="https://lite.cnn.com/2023/11/05/tech/nvidia-amd-ceos-taiwan-intl-hnk/index.html")

"""
## ✨ Conclusions and caveats
- Nexus Raven seems to work much better than Gorilla Open Functions (v0) for this use case.
- I would also expect it to work significantly better than generic models to which grammars are added to make them produce JSON.
- ⚠️ When the content of the web page is cluttered with extraneous information such as advertisements and interruptions, the model encounters difficulty in extracting relevant information, leading to occasional instances where it returns empty responses.
- ⚠️ As a statistical model, the LLM is highly responsive to prompts. For instance, modifying the order and description of the specified arguments can yield different extraction results.

## 📚 References
*Related to the experiment*
- [Haystack LLM framework](https://haystack.deepset.ai/)
- [Using Ollama Function Calling for Information Extraction: gist by Kyle McDonald](https://gist.github.com/kylemcdonald/dbac21de2d7855633689f5526225154c)
- [NexusRaven-V2: Surpassing GPT-4 for Zero-shot Function Calling](https://nexusflow.ai/blogs/ravenv2)
"""
logger.info("## ✨ Conclusions and caveats")

logger.info("\n\n[DONE]", bright=True)