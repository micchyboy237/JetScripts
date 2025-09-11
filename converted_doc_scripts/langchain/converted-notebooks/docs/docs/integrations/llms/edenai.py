from PIL import Image
from io import BytesIO
from jet.logger import logger
from langchain.chains import LLMChain
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_community.llms import EdenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import base64
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
# Eden AI

Eden AI is revolutionizing the AI landscape by uniting the best AI providers, empowering users to unlock limitless possibilities and tap into the true potential of artificial intelligence. With an all-in-one comprehensive and hassle-free platform, it allows users to deploy AI features to production lightning fast, enabling effortless access to the full breadth of AI capabilities via a single API. (website: https://edenai.co/)

This example goes over how to use LangChain to interact with Eden AI models

-----------------------------------------------------------------------------------

Accessing the EDENAI's API requires an API key, 

which you can get by creating an account https://app.edenai.run/user/register  and heading here https://app.edenai.run/admin/account/settings

Once we have a key we'll want to set it as an environment variable by running:

```bash
export EDENAI_API_KEY="..."
```

If you'd prefer not to set an environment variable you can pass the key in directly via the edenai_api_key named parameter

 when initiating the EdenAI LLM class:
"""
logger.info("# Eden AI")


llm = EdenAI(edenai_provider="ollama", temperature=0.2, max_tokens=250)

"""
## Calling a model

The EdenAI API brings together various providers, each offering multiple models.

To access a specific model, you can simply add 'model' during instantiation.

For instance, let's explore the models provided by Ollama, such as GPT3.5

### text generation
"""
logger.info("## Calling a model")


llm = EdenAI(
    feature="text",
    provider="ollama",
    model="llama3.2",
    temperature=0.2,
    max_tokens=250,
)

prompt = """
User: Answer the following yes/no question by reasoning step by step. Can a dog drive a car?
Assistant:
"""

llm(prompt)

"""
### image generation
"""
logger.info("### image generation")




def print_base64_image(base64_string):
    decoded_data = base64.b64decode(base64_string)

    image_stream = BytesIO(decoded_data)

    image = Image.open(image_stream)

    image.show()

text2image = EdenAI(feature="image", provider="ollama", resolution="512x512")

image_output = text2image("A cat riding a motorcycle by Picasso")

print_base64_image(image_output)

"""
### text generation with callback
"""
logger.info("### text generation with callback")


llm = EdenAI(
    callbacks=[StreamingStdOutCallbackHandler()],
    feature="text",
    provider="ollama",
    temperature=0.2,
    max_tokens=250,
)
prompt = """
User: Answer the following yes/no question by reasoning step by step. Can a dog drive a car?
Assistant:
"""
logger.debug(llm.invoke(prompt))

"""
## Chaining Calls
"""
logger.info("## Chaining Calls")


llm = EdenAI(feature="text", provider="ollama", temperature=0.2, max_tokens=250)
text2image = EdenAI(feature="image", provider="ollama", resolution="512x512")

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a description of a logo for this company: {company_name}, the logo should not contain text at all ",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

third_prompt = PromptTemplate(
    input_variables=["company_logo_description"],
    template="{company_logo_description}",
)
chain_three = LLMChain(llm=text2image, prompt=third_prompt)

overall_chain = SimpleSequentialChain(
    chains=[chain, chain_two, chain_three], verbose=True
)
output = overall_chain.run("hats")

print_base64_image(output)

logger.info("\n\n[DONE]", bright=True)