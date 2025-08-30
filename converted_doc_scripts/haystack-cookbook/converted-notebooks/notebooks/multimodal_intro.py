async def main():
    from jet.transformers.formatters import format_json
    from haystack.components.agents import Agent
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.converters.image import ImageFileToImageContent
    from haystack.components.converters.image import PDFToImageContent
    from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.dataclasses import ChatMessage, ImageContent
    from haystack.dataclasses import Document
    from haystack.dataclasses import ImageContent, ChatMessage
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.tools import tool
    from jet.logger import CustomLogger
    from pprint import pp as print
    from typing import Annotated
    import asyncio
    import base64
    import gdown
    import glob
    import os
    import python_weather
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # 🖼️ Introduction to Multimodal Text Generation
    
    In this notebook, we introduce the features that enable multimodal text generation in Haystack.
    
    - We introduced the `ImageContent` dataclass, which represents the image content of a user `ChatMessage`.
    - We developed some image converter components.
    - The `OllamaFunctionCallingAdapterChatGenerator` was extended to support multimodal messages.
    - The `ChatPromptBuilder` was refactored to also work with string templates, making it easier to support multimodal use cases.
    
    In this notebook, we'll introduce all these features, show an application using **textual retrieval + multimodal generation**, and a **multimodal Agent**.
    
    ## Setup Development Environment
    """
    logger.info("# 🖼️ Introduction to Multimodal Text Generation")
    
    # !pip install haystack-ai gdown nest_asyncio pillow pypdfium2 python-weather
    
    # from getpass import getpass
    
    
    # if "OPENAI_API_KEY" not in os.environ:
    #   os.environ["OPENAI_API_KEY"] = getpass("Enter OllamaFunctionCallingAdapter API key:")
    
    """
    ## Introduction to `ImageContent`
    
    [`ImageContent`](https://docs.haystack.deepset.ai/docs/chatmessage#types-of-content) is a new dataclass that stores the image content of a user `ChatMessage`.
    
    It has the following attributes:
    - `base64_image`: A base64 string representing the image.
    - `mime_type`: The optional MIME type of the image (e.g. "image/png", "image/jpeg").
    - `detail`: Optional detail level of the image (only supported by OllamaFunctionCallingAdapter). One of "auto", "high", or "low".
    - `meta`: Optional metadata for the image.
    
    ### Creating an `ImageContent` Object
    
    Let's start by downloading an image from the web and manually creating an `ImageContent` object. We'll see more convenient ways to do this later.
    """
    logger.info("## Introduction to `ImageContent`")
    
    # ! wget "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Cattle_tyrant_%28Machetornis_rixosa%29_on_Capybara.jpg/960px-Cattle_tyrant_%28Machetornis_rixosa%29_on_Capybara.jpg?download" -O capybara.jpg
    
    
    with open("capybara.jpg", "rb") as fd:
      base64_image = base64.b64encode(fd.read()).decode("utf-8")
    
    image_content = ImageContent(
        base64_image=base64_image,
        mime_type="image/jpeg",
        detail="low")
    
    image_content
    
    image_content.show()
    
    """
    Nice!
    
    To perform text generation based on this image, we need to pass it in a user message with a prompt. Let's do that.
    """
    logger.info("Nice!")
    
    user_message = ChatMessage.from_user(content_parts=["Describe the image in short.", image_content])
    llm = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2")
    
    logger.debug(llm.run([user_message])["replies"][0].text)
    
    """
    ### Creating an `ImageContent` Object from URL or File Path
    
    `ImageContent` features two utility class methods:
    - `from_url`: downloads an image file and wraps it in `ImageContent`
    - `from_file_path`: loads an image from disk and wraps it in `ImageContent`
    
    Using `from_url`, we can simplify the previous example. `mime_type` is automatically inferred.
    """
    logger.info("### Creating an `ImageContent` Object from URL or File Path")
    
    capybara_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Cattle_tyrant_%28Machetornis_rixosa%29_on_Capybara.jpg/960px-Cattle_tyrant_%28Machetornis_rixosa%29_on_Capybara.jpg?download"
    
    image_content = ImageContent.from_url(capybara_image_url, detail="low")
    image_content
    
    """
    Since we downloaded the image file, we can also see `from_file_path` in action.
    
    In this case, we will also use the `size` parameter, that resizes the image to fit within the specified dimensions while maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial when working with models that have resolution constraints or when transmitting images to remote services.
    """
    logger.info("Since we downloaded the image file, we can also see `from_file_path` in action.")
    
    image_content = ImageContent.from_file_path("capybara.jpg", detail="low", size=(300, 300))
    image_content
    
    image_content.show()
    
    """
    ## Image Converters for `ImageContent`
    
    To perform image conversion in multimodal pipelines, we also introduced two image converters:
    - [`ImageFileToImageContent`](https://docs.haystack.deepset.ai/docs/imagefiletoimagecontent), which converts image files to `ImageContent` objects (similar to `from_file_path`).
    - [`PDFToImageContent`](https://docs.haystack.deepset.ai/docs/pdftoimagecontent), which converts PDF files to `ImageContent` objects.
    """
    logger.info("## Image Converters for `ImageContent`")
    
    
    converter = ImageFileToImageContent(detail="low", size=(300, 300))
    result = converter.run(sources=["capybara.jpg"])
    
    result["image_contents"][0]
    
    """
    Let's see a more interesting example. We want our LLM to interpret a figure in this influential paper by Google: [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416).
    """
    logger.info("Let's see a more interesting example. We want our LLM to interpret a figure in this influential paper by Google: [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416).")
    
    # ! wget "https://arxiv.org/pdf/2210.11416.pdf" -O flan_paper.pdf
    
    
    pdf_converter = PDFToImageContent()
    paper_page_image = pdf_converter.run(sources=["flan_paper.pdf"], page_range="9")["image_contents"][0]
    paper_page_image
    
    paper_page_image.show()
    
    user_message = ChatMessage.from_user(content_parts=["What is the main takeaway of Figure 6? Be brief and accurate.", paper_page_image])
    
    logger.debug(llm.run([user_message])["replies"][0].text)
    
    """
    ## Extended `ChatPromptBuilder` with String Templates
    
    As we explored multimodal use cases, it became clear that the existing `ChatPromptBuilder` had some limitations. Specifically, we need a way to pass structured objects like `ImageContent` when building `ChatMessage`, and to handle a variable number of such objects.
    
    To address this, we are introducing [support for string templates in the `ChatPromptBuilder`](https://docs.haystack.deepset.ai/docs/chatpromptbuilder#string-templates). The syntax is pretty simple, as you can see below.
    """
    logger.info("## Extended `ChatPromptBuilder` with String Templates")
    
    template = """
    {% message role="system" %}
    You are a {{adjective}} assistant.
    {% endmessage %}
    
    {% message role="user" %}
    Compare these images:
    {% for img in image_contents %}
      {{ img | templatize_part }}
    {% endfor %}
    {% endmessage %}
    """
    
    """
    Note the `| templatize_part` Jinja2 filter: this is used to indicate that the content part is a structured object, not plain text, and needs special treatment.
    """
    logger.info("Note the `| templatize_part` Jinja2 filter: this is used to indicate that the content part is a structured object, not plain text, and needs special treatment.")
    
    
    builder = ChatPromptBuilder(template, required_variables="*")
    
    image_contents = [ImageContent.from_url("https://1000logos.net/wp-content/uploads/2017/02/Apple-Logosu.png", detail="low"),
                      ImageContent.from_url("https://upload.wikimedia.org/wikipedia/commons/2/26/Pink_Lady_Apple_%284107712628%29.jpg", detail="low")]
    
    messages = builder.run(image_contents=image_contents, adjective="joking")["prompt"]
    logger.debug(messages)
    
    logger.debug(llm.run(messages)["replies"][0].text)
    
    """
    ## Textual Retrieval and Multimodal Generation
    
    Let's see a more advanced example.
    
    In this case, we have a collection of images from papers about Language Models.
    
    Our goal is to build a system that can:
    1. Retrieve the most relevant image from this collection based on a user's textual question.
    2. Use this image, along with the original question, to have an LLM generate an answer.
    
    We start by downloading the images.
    """
    logger.info("## Textual Retrieval and Multimodal Generation")
    
    
    url = "https://drive.google.com/drive/folders/1KLMow1NPq6GIuoNfOmUbjUmAcwFNmsCc"
    
    gdown.download_folder(url, quiet=True, output=".")
    
    """
    We create an `InMemoryDocumentStore` and write a Document there for each image: the content is a textual description of the image; the image path is stored in `meta`.
    
    The content of the Documents here is minimal. You can think of more sophisticated ways to create a representive content: perform OCR or use a Vision Language Model.
    We'll explore this direction in the future.
    """
    logger.info("We create an `InMemoryDocumentStore` and write a Document there for each image: the content is a textual description of the image; the image path is stored in `meta`.")
    
    
    docs = []
    
    for image_path in glob.glob("arxiv/*.png"):
        text = "image from '" + image_path.split("/")[-1].replace(".png", "").replace("_", " ") + "' paper"
        docs.append(Document(content=text, meta={"image_path": image_path}))
    
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs)
    
    """
    We perform text-based retrieval (using BM25) to get the most relevant Document. Then an `ImageContent` object is created using the image file path. Finally, the `ImageContent` is passed to the LLM with the user question.
    """
    logger.info("We perform text-based retrieval (using BM25) to get the most relevant Document. Then an `ImageContent` object is created using the image file path. Finally, the `ImageContent` is passed to the LLM with the user question.")
    
    
    retriever = InMemoryBM25Retriever(document_store=document_store)
    llm = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2")
    
    def retrieve_and_generate(question):
      doc = retriever.run(query=question, top_k=1)["documents"][0]
      image_content = ImageContent.from_file_path(doc.meta["image_path"], detail="auto")
      image_content.show()
    
      message = ChatMessage.from_user(content_parts=[question, image_content])
      response = llm.run(messages=[message])["replies"][0].text
      logger.debug(response)
    
    retrieve_and_generate("Describe the image of the Direct Preference Optimization paper")
    
    """
    Here are some other example questions to test:
    """
    logger.info("Here are some other example questions to test:")
    
    examples = [
        "Describe the image of the LoRA vs Full Fine-tuning paper",
        "Describe the image of the Online AI Feedback paper",
        "Describe the image of the Spectrum paper",
        "Describe the image of the Textgrad paper",
        "Describe the image of the Tulu 3 paper",
    ]
    
    """
    ## Multimodal Agent
    
    Let's combine multimodal messages with the [Agent](https://docs.haystack.deepset.ai/docs/agents) component. 
    
    We start with creating a weather [Tool](https://docs.haystack.deepset.ai/docs/tool), based on the [python-weather library](https://python-weather.readthedocs.io/). The library is asynchronous while the Tool abstraction expects a synchronous invocation method, so we make some adaptations.
    
    > Learn more about creating agents in [Tutorial: Build a Tool-Calling Agent](https://haystack.deepset.ai/tutorials/43_building_a_tool_calling_agent)
    """
    logger.info("## Multimodal Agent")
    
    
    
    
    # import nest_asyncio
    # nest_asyncio.apply()
    
    
    @tool
    def get_weather(location: Annotated[str, "The location to get the weather for"]) -> dict:
        """A function to get the weather for a given location"""
        async def _fetch_weather():
            async with python_weather.Client(unit=python_weather.METRIC) as client:
                    weather = await client.get(location)
                    return {
                        "description": weather.description,
                        "temperature": weather.temperature,
                        "humidity": weather.humidity,
                        "precipitation": weather.precipitation,
                        "wind_speed": weather.wind_speed,
                        "wind_direction": weather.wind_direction
                    }
                
            logger.success(format_json(result))
        return asyncio.run(_fetch_weather())
    
    """
    Let's test our Tool by invoking it with the required parameter:
    """
    logger.info("Let's test our Tool by invoking it with the required parameter:")
    
    get_weather.invoke(location="New York")
    
    """
    We can now define an Agent, provide it with the weather Tool and see if it can find the weather based on a geographical map.
    """
    logger.info("We can now define an Agent, provide it with the weather Tool and see if it can find the weather based on a geographical map.")
    
    generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2")
    agent = Agent(chat_generator=generator, tools=[get_weather])
    
    map_image = ImageContent.from_file_path("map.png")
    map_image.show()
    
    content_parts = ["What is the weather in the area of the map?", map_image]
    messages = agent.run([ChatMessage.from_user(content_parts=content_parts)])["messages"]
    
    logger.debug(messages[-1].text)
    
    """
    ## What's next?
    
    We also support image capabilities across a variety of LLM providers, including Amazon Bedrock, Google, Mistral, Ollama, and more.
    
    To learn how to build more advanced multimodal pipelines, with different file formats and multimodal embedding models, check out the [Creating Vision+Text RAG Pipelines tutorial](https://haystack.deepset.ai/tutorials/46_multimodal_rag).
    
    (*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*)
    """
    logger.info("## What's next?")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())