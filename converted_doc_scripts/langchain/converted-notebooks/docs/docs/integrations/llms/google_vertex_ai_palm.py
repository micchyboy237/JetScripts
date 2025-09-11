from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_core.messages import (
from langchain_core.messages import (
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import (
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIModelGarden
from langchain_google_vertexai.model_garden import ChatOllamaVertex
from vertexai.preview.generative_models import Image
import base64
import os
import shutil

async def main():
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    )
    AIMessage,
    HumanMessage,
    )
    GemmaChatVertexAIModelGarden,
    GemmaVertexAIModelGarden,
    )
    
    
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
    ---
    keywords: [gemini, vertex, VertexAI, gemini-pro]
    ---
    
    # Google Cloud Vertex AI
    
    :::caution
    You are currently on a page documenting the use of Google Vertex [text completion models](/docs/concepts/text_llms). Many Google models are [chat completion models](/docs/concepts/chat_models).
    
    You may be looking for [this page instead](/docs/integrations/chat/google_vertex_ai_palm/).
    :::
    
    **Note:** This is separate from the `Google Generative AI` integration, it exposes [Vertex AI Generative API](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) on `Google Cloud`.
    
    VertexAI exposes all foundational models available in google cloud.
    
    For a full and updated list of available models visit [VertexAI documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models)
    
    ## Setup
    
    By default, Google Cloud [does not use](https://cloud.google.com/vertex-ai/docs/generative-ai/data-governance#foundation_model_development) customer data to train its foundation models as part of Google Cloud's AI/ML Privacy Commitment. More details about how Google processes data can also be found in [Google's Customer Data Processing Addendum (CDPA)](https://cloud.google.com/terms/data-processing-addendum).
    
    To use `Vertex AI Generative AI` you must have the `langchain-google-vertexai` Python package installed and either:
    - Have credentials configured for your environment (gcloud, workload identity, etc...)
    - Store the path to a service account JSON file as the `GOOGLE_APPLICATION_CREDENTIALS` environment variable
    
    This codebase uses the `google.auth` library which first looks for the application credentials variable mentioned above, and then looks for system-level auth.
    
    For more information, see:
    - https://cloud.google.com/docs/authentication/application-default-credentials#GAC
    - https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth
    """
    logger.info("# Google Cloud Vertex AI")
    
    # %pip install --upgrade --quiet  langchain-core langchain-google-vertexai
    
    """
    ## Usage
    
    VertexAI supports all [LLM](/docs/how_to#llms) functionality.
    """
    logger.info("## Usage")
    
    
    model = VertexAI(model_name="gemini-2.5-pro")
    
    message = "What are some of the pros and cons of Python as a programming language?"
    model.invoke(message)
    
    await model.ainvoke(message)
    
    for chunk in model.stream(message):
        logger.debug(chunk, end="", flush=True)
    
    model.batch([message])
    
    """
    We can use the `generate` method to get back extra metadata like [safety attributes](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/responsible-ai#safety_attribute_confidence_scoring) and not just text completions.
    """
    logger.info("We can use the `generate` method to get back extra metadata like [safety attributes](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/responsible-ai#safety_attribute_confidence_scoring) and not just text completions.")
    
    result = model.generate([message])
    result.generations
    
    """
    ### OPTIONAL : Managing [Safety Attributes](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/responsible-ai#safety_attribute_confidence_scoring)
    - If your use case requires your to manage thresholds for saftey attributes, you can do so using below snippets
    >NOTE : We recommend exercising extreme caution when adjusting Safety Attributes thresholds
    """
    logger.info("### OPTIONAL : Managing [Safety Attributes](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/responsible-ai#safety_attribute_confidence_scoring)")
    
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    llm = VertexAI(model_name="gemini-1.0-pro-001", safety_settings=safety_settings)
    
    output = llm.invoke(["How to make a molotov cocktail?"])
    output
    
    llm = VertexAI(model_name="gemini-2.5-pro")
    
    output = llm.invoke(
        ["How to make a molotov cocktail?"], safety_settings=safety_settings
    )
    output
    
    result = await model.ainvoke([message])
    logger.success(format_json(result))
    result
    
    """
    You can also easily combine with a prompt template for easy structuring of user input. We can do this using [LCEL](/docs/concepts/lcel)
    """
    logger.info("You can also easily combine with a prompt template for easy structuring of user input. We can do this using [LCEL](/docs/concepts/lcel)")
    
    
    template = """Question: {question}
    
    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)
    
    chain = prompt | model
    
    question = """
    I have five apples. I throw two away. I eat one. How many apples do I have left?
    """
    logger.debug(chain.invoke({"question": question}))
    
    """
    You can use different foundational models for specialized in different tasks. 
    For an updated list of available models visit [VertexAI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/overview)
    """
    logger.info("You can use different foundational models for specialized in different tasks.")
    
    llm = VertexAI(model_name="code-bison", max_tokens=1000, temperature=0.3)
    question = "Write a python function that checks if a string is a valid email address"
    
    logger.debug(model.invoke(question))
    
    """
    ## Multimodality
    
    With Gemini, you can use LLM in a multimodal mode:
    """
    logger.info("## Multimodality")
    
    
    llm = ChatVertexAI(model="gemini-pro-vision")
    
    image_message = {
        "type": "image_url",
        "image_url": {"url": "image_example.jpg"},
    }
    text_message = {
        "type": "text",
        "text": "What is shown in this image?",
    }
    
    message = HumanMessage(content=[text_message, image_message])
    
    output = llm.invoke([message])
    logger.debug(output.content)
    
    """
    Let's double-check it's a cat :)
    """
    logger.info("Let's double-check it's a cat :)")
    
    
    i = Image.load_from_file("image_example.jpg")
    i
    
    """
    You can also pass images as bytes:
    """
    logger.info("You can also pass images as bytes:")
    
    
    with open("image_example.jpg", "rb") as image_file:
        image_bytes = image_file.read()
    
    image_message = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        },
    }
    text_message = {
        "type": "text",
        "text": "What is shown in this image?",
    }
    
    message = HumanMessage(content=[text_message, image_message])
    
    output = llm.invoke([message])
    logger.debug(output.content)
    
    """
    Please, note that you can also use the image stored in GCS (just point the `url` to the full GCS path, starting with `gs://` instead of a local one).
    
    And you can also pass a history of a previous chat to the LLM:
    """
    logger.info("Please, note that you can also use the image stored in GCS (just point the `url` to the full GCS path, starting with `gs://` instead of a local one).")
    
    message2 = HumanMessage(content="And where the image is taken?")
    
    output2 = llm.invoke([message, output, message2])
    logger.debug(output2.content)
    
    """
    You can also use the public image URL:
    """
    logger.info("You can also use the public image URL:")
    
    image_message = {
        "type": "image_url",
        "image_url": {
            "url": "gs://github-repo/img/vision/google-cloud-next.jpeg",
        },
    }
    text_message = {
        "type": "text",
        "text": "What is shown in this image?",
    }
    
    message = HumanMessage(content=[text_message, image_message])
    
    output = llm.invoke([message])
    logger.debug(output.content)
    
    """
    ### Using Pdfs with Gemini Models
    """
    logger.info("### Using Pdfs with Gemini Models")
    
    
    llm = ChatVertexAI(model="gemini-2.5-pro")
    
    pdf_message = {
        "type": "image_url",
        "image_url": {"url": "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"},
    }
    
    text_message = {
        "type": "text",
        "text": "Summarize the provided document.",
    }
    
    message = HumanMessage(content=[text_message, pdf_message])
    
    llm.invoke([message])
    
    """
    ### Using Video with Gemini Models
    """
    logger.info("### Using Video with Gemini Models")
    
    
    llm = ChatVertexAI(model="gemini-2.5-pro")
    
    media_message = {
        "type": "image_url",
        "image_url": {
            "url": "gs://cloud-samples-data/generative-ai/video/pixel8.mp4",
        },
    }
    
    text_message = {
        "type": "text",
        "text": """Provide a description of the video.""",
    }
    
    message = HumanMessage(content=[media_message, text_message])
    
    llm.invoke([message])
    
    """
    ### Using Audio with Gemini Models
    """
    logger.info("### Using Audio with Gemini Models")
    
    
    llm = ChatVertexAI(model="gemini-2.5-pro")
    
    media_message = {
        "type": "image_url",
        "image_url": {
            "url": "gs://cloud-samples-data/generative-ai/audio/pixel.mp3",
        },
    }
    
    text_message = {
        "type": "text",
        "text": """Can you transcribe this interview, in the format of timecode, speaker, caption.
      Use speaker A, speaker B, etc. to identify speakers.""",
    }
    
    message = HumanMessage(content=[media_message, text_message])
    
    llm.invoke([message])
    
    """
    ## Vertex Model Garden
    
    Vertex Model Garden [exposes](https://cloud.google.com/vertex-ai/docs/start/explore-models) open-sourced models that can be deployed and served on Vertex AI. 
    
    Hundreds popular [open-sourced models](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models#oss-models) like Llama, Falcon and are available for  [One Click Deployment](https://cloud.google.com/vertex-ai/generative-ai/docs/deploy/overview)
    
    If you have successfully deployed a model from Vertex Model Garden, you can find a corresponding Vertex AI [endpoint](https://cloud.google.com/vertex-ai/docs/general/deployment#what_happens_when_you_deploy_a_model) in the console or via API.
    """
    logger.info("## Vertex Model Garden")
    
    
    llm = VertexAIModelGarden(project="YOUR PROJECT", endpoint_id="YOUR ENDPOINT_ID")
    
    llm.invoke("What is the meaning of life?")
    
    """
    Like all LLMs, we can then compose it with other components:
    """
    logger.info("Like all LLMs, we can then compose it with other components:")
    
    prompt = PromptTemplate.from_template("What is the meaning of {thing}?")
    
    chain = prompt | llm
    logger.debug(chain.invoke({"thing": "life"}))
    
    """
    ### Llama on Vertex Model Garden 
    
    > Llama is a family of open weight models developed by Meta that you can fine-tune and deploy on Vertex AI. Llama models are pre-trained and fine-tuned generative text models. You can deploy Llama 2 and Llama 3 models on Vertex AI.
    [Official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-llama) for more information about Llama on [Vertex Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models)
    
    To use Llama on Vertex Model Garden you must first [deploy it to Vertex AI Endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models#deploy-a-model)
    """
    logger.info("### Llama on Vertex Model Garden")
    
    
    llm = VertexAIModelGarden(project="YOUR PROJECT", endpoint_id="YOUR ENDPOINT_ID")
    
    llm.invoke("What is the meaning of life?")
    
    """
    Like all LLMs, we can then compose it with other components:
    """
    logger.info("Like all LLMs, we can then compose it with other components:")
    
    
    prompt = PromptTemplate.from_template("What is the meaning of {thing}?")
    
    chain = prompt | llm
    logger.debug(chain.invoke({"thing": "life"}))
    
    """
    ### Falcon on Vertex Model Garden
    
    > Falcon is a family of open weight models developed by [Falcon](https://falconllm.tii.ae/) that you can fine-tune and deploy on Vertex AI. Falcon models are pre-trained and fine-tuned generative text models.
    
    To use Falcon on Vertex Model Garden you must first [deploy it to Vertex AI Endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models#deploy-a-model)
    """
    logger.info("### Falcon on Vertex Model Garden")
    
    
    llm = VertexAIModelGarden(project="YOUR PROJECT", endpoint_id="YOUR ENDPOINT_ID")
    
    llm.invoke("What is the meaning of life?")
    
    """
    Like all LLMs, we can then compose it with other components:
    """
    logger.info("Like all LLMs, we can then compose it with other components:")
    
    
    prompt = PromptTemplate.from_template("What is the meaning of {thing}?")
    
    chain = prompt | llm
    logger.debug(chain.invoke({"thing": "life"}))
    
    """
    ### Gemma on Vertex AI Model Garden
    
    > [Gemma](https://ai.google.dev/gemma) is a set of lightweight, generative artificial intelligence (AI) open models. Gemma models are available to run in your applications and on your hardware, mobile devices, or hosted services. You can also customize these models using tuning techniques so that they excel at performing tasks that matter to you and your users. Gemma models are based on [Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/overview) models and are intended for the AI development community to extend and take further.
    
    To use Gemma on Vertex Model Garden you must first [deploy it to Vertex AI Endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models#deploy-a-model)
    """
    logger.info("### Gemma on Vertex AI Model Garden")
    
    
    llm = GemmaVertexAIModelGarden(
        endpoint_id="YOUR PROJECT",
        project="YOUR ENDPOINT_ID",
        location="YOUR REGION",
    )
    
    llm.invoke("What is the meaning of life?")
    
    chat_llm = GemmaChatVertexAIModelGarden(
        endpoint_id="YOUR PROJECT",
        project="YOUR ENDPOINT_ID",
        location="YOUR REGION",
    )
    
    text_question1 = "How much is 2+2?"
    message1 = HumanMessage(content=text_question1)
    
    chat_llm.invoke([message1])
    
    """
    ## Ollama on Vertex AI
    
    > [Ollama Claude 3](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) models on Vertex AI offer fully managed and serverless models as APIs. To use a Claude model on Vertex AI, send a request directly to the Vertex AI API endpoint. Because Ollama Claude 3 models use a managed API, there's no need to provision or manage infrastructure.
    
    NOTE : Ollama Models on Vertex are implemented as Chat Model through class `ChatOllamaVertex`
    """
    logger.info("## Ollama on Vertex AI")
    
    # !pip install -U langchain-google-vertexai anthropic[vertex]
    
    
    """
    NOTE : Specify the correct [Claude 3 Model Versions](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#claude-opus)
    
    We don't recommend using the Ollama Claude 3 model versions that don't include a suffix that starts with an @ symbol (claude-3-opus, claude-3-sonnet, or claude-3-haiku).
    """
    logger.info("NOTE : Specify the correct [Claude 3 Model Versions](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#claude-opus)")
    
    project = "<project_id>"
    location = "<region>"
    
    model = ChatOllamaVertex(
        model_name="claude-3-haiku@20240307",
        project=project,
        location=location,
    )
    
    raw_context = (
        "My name is Peter. You are my personal assistant. My favorite movies "
        "are Lord of the Rings and Hobbit."
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    context = SystemMessage(content=raw_context)
    message = HumanMessage(content=question)
    
    response = model.invoke([context, message])
    logger.debug(response.content)
    
    response = model.invoke([context, message], model_name="claude-3-sonnet@20240229")
    logger.debug(response.content)
    
    sync_response = model.stream([context, message], model_name="claude-3-haiku@20240307")
    for chunk in sync_response:
        logger.debug(chunk.content)
    
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