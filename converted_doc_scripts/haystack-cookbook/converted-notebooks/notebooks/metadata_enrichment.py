from haystack import Document, component, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ChatMessage, StreamingChunk
from jet.logger import logger
from ollama import Stream
from ollama.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel
from typing import List, Any, Dict, Optional, Callable, Union
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
# Advanced RAG: Automated Structured Metadata Enrichment

by Tuana Celik ([LI](https://www.linkedin.com/in/tuanacelik/), [Twitter](https://x.com/tuanacelik))

> This is part one of the **Advanced Use Cases** series:
>
> 1️⃣ Extract Metadata from Queries to Improve Retrieval [cookbook](/cookbook/extracting_metadata_filters_from_a_user_query) & [full article](/blog/extracting-metadata-filter)
>
> 2️⃣ Query Expansion [cookbook](/cookbook/query-expansion) & [full article](/blog/query-expansion)
>
> 3️⃣ Query Decomposition [cookbook](/cookbook/query_decomposition) & the [full article](/blog/query-decomposition)
>
> 4️⃣ **Automated Metadata Enrichment**

In this example, you'll see how you can make use of structured outputs which is an option for some LLMs, and a custom Haystack component, to automate the enrichment of metadata from documents.

You will see how you can define your own metadata fields as a Pydantic Model, as well as the data types each field should have. Finally, you will get a custom `MetadataEnricher` to extract the required fields and add them to the document meta information.

In this example, we will be enriching metadata with information relating the funding announements.

Once we populate the metadata of a document with our own fields, we are able to use Metadata Filtering during the retrieval step of RAG pipelines. We can even combine this with [Metadata Extraction from Queries to Improve Retrieval](https://haystack.deepset.ai/blog/extracting-metadata-filter) to be very precise about what documents we are providing as context to an LLM.

## 📺 Code Along

<iframe width="560" height="315" src="https://www.youtube.com/embed/vk0U1V-cBK0?si=-MHeM23RRfdlAlgm" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Install requirements
"""
logger.info("# Advanced RAG: Automated Structured Metadata Enrichment")

# !pip install haystack-ai
# !pip install trafilatura



"""
## 🧪 Experimental Addition to the OpenAIGenerator for Structured Output Support

> 🚀 This is the same extension to the `OpenAIGenerator` that was used in the [Advanced RAG: Query Decomposition and Reasoning](https://haystack.deepset.ai/cookbook/query_decomposition) example

Let's extend the `OpenAIGeneraotor` to be able to make use of the [strctured output option by Ollama](https://platform.ollama.com/docs/guides/structured-outputs/introduction). Below, we extend the class to call `self.client.beta.chat.completions.parse` if the user has provides a `respose_format` in `generation_kwargs`. This will allow us to provifde a Pydantic Model to the gnerator and request our generator to respond with structured outputs that adhere to this Pydantic schema.
"""
logger.info("## 🧪 Experimental Addition to the OpenAIGenerator for Structured Output Support")

class OpenAIGenerator(OpenAIGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]], structured_reply=BaseModel)
    def run(self, prompt: str, streaming_callback: Optional[Callable[[StreamingChunk], None]] = None, generation_kwargs: Optional[Dict[str, Any]] = None,):
      generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
      if "response_format" in generation_kwargs.keys():
        message = ChatMessage.from_user(prompt)
        if self.system_prompt:
            messages = [ChatMessage.from_system(self.system_prompt), message]
        else:
            messages = [message]

        streaming_callback = streaming_callback or self.streaming_callback
        openai_formatted_messages = [message.to_openai_dict_format() for message in messages]
        completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=openai_formatted_messages,
            **generation_kwargs)
        completions = [self._build_structured_message(completion, choice) for choice in completion.choices]
        for response in completions:
            self._check_finish_reason(response)

        return {
            "replies": [message.text for message in completions],
            "meta": [message.meta for message in completions],
            "structured_reply": completions[0].text
        }
      else:
          return super().run(prompt, streaming_callback, generation_kwargs)

    def _build_structured_message(self, completion: Any, choice: Any) -> ChatMessage:
        chat_message = ChatMessage.from_assistant(choice.message.parsed or "")
        chat_message.meta.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage),
            }
        )
        return chat_message

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#   os.environ["OPENAI_API_KEY"] = getpass("Ollama API Key:")

"""
## Custom `MetadataEnricher`

We create a custom Haystack component that is able ti accept `metadata_model` and `prompt`. If no prompt is provided, it usees the `DEFAULT_PROMPT`.

This component returns `documents` enriched with the requested metadata fileds.
"""
logger.info("## Custom `MetadataEnricher`")

DEFAULT_PROMPT = """
Given the contents of the documents, extract the requested metadata.
The requested metadata is {{ metadata_model }}
Document:
{{document}}
Metadata:
"""
@component
class MetadataEnricher:

    def __init__(self, metadata_model: BaseModel, prompt:str = DEFAULT_PROMPT):
        self.metadata_model = metadata_model
        self.metadata_prompt = prompt

        builder = PromptBuilder(self.metadata_prompt)
        llm = OpenAIGenerator(generation_kwargs={"response_format": metadata_model})
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        documents_with_meta = []
        for document in documents:
          result = self.pipeline.run({'builder': {'document': document.content, 'metadata_model': self.metadata_model}})
          metadata = result['llm']['structured_reply']
          document.meta.update(metadata.dict())
          documents_with_meta.append(document)
        return {"documents": documents_with_meta}

"""
## Define Metadata Fields as a Pydantic Model

For automatic metadata enrichment, we want to be able to provide a structure describing what fields we want to extract, as well as what types they should be.

Below, I have defined a `Metadata` model, with 4 fields.

> 💡 **Note:** In some cases, it might make sense to make each field optional, or provide default values.
"""
logger.info("## Define Metadata Fields as a Pydantic Model")

class Metadata(BaseModel):
    company: str
    year: int
    funding_value: int
    funding_currency: str

"""
Next, we initialize a `MetadataEnricher` and provide `Metadata` as the `metadata_model` we want to abide by.
"""
logger.info("Next, we initialize a `MetadataEnricher` and provide `Metadata` as the `metadata_model` we want to abide by.")

enricher = MetadataEnricher(metadata_model=Metadata)

"""
## Build an Automated Metadata Enrichment Pipeline

Now that we have our `enricher`, we can use it in a pipeline. Below is an example of a pipeline that fetches the contents of some URLs (in this case, urls that contain information about funding announcements). The pipeline then adds the requested metadata fields to each `Document`'s `meta` field 👇
"""
logger.info("## Build an Automated Metadata Enrichment Pipeline")

pipeline = Pipeline()
pipeline.add_component("fetcher", LinkContentFetcher())
pipeline.add_component("converter", HTMLToDocument())
pipeline.add_component("enricher", enricher)


pipeline.connect("fetcher", "converter")
pipeline.connect("converter.documents", "enricher.documents")

pipeline.run({"fetcher": {"urls": ['https://techcrunch.com/2023/08/09/deepset-secures-30m-to-expand-its-llm-focused-mlops-offerings/',
                                   'https://www.prnewswire.com/news-releases/arize-ai-raises-38-million-series-b-to-scale-machine-learning-observability-platform-301620603.html']}})

pipeline.show()

"""
## Extra: Metadata Inheritance

This is just an extra step to show how metadata that belongs to a document is inherited by the document chunks if you use a component such as the `DocumentSplitter`.
"""
logger.info("## Extra: Metadata Inheritance")

pipeline.add_component("splitter", DocumentSplitter())

pipeline.connect("enricher", "splitter")

pipeline.run({"fetcher": {"urls": ['https://techcrunch.com/2023/08/09/deepset-secures-30m-to-expand-its-llm-focused-mlops-offerings/',
                                   'https://www.prnewswire.com/news-releases/arize-ai-raises-38-million-series-b-to-scale-machine-learning-observability-platform-301620603.html']}})

logger.info("\n\n[DONE]", bright=True)