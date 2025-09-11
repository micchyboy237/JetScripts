from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_gradient import ChatGradient
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
# DigitalOcean Gradient

This will help you getting started with DigitalOcean Gradient [chat models](/docs/concepts/chat_models).


## Setup

langchain-gradient uses DigitalOcean's Gradient™ AI Platform.

Create an account on DigitalOcean, acquire a `DIGITALOCEAN_INFERENCE_KEY` API key from the Gradient Platform, and install the `langchain-gradient` integration package.

### Credentials

Head to [DigitalOcean Gradient](https://www.digitalocean.com/products/gradient)

1. Sign up/Login to DigitalOcean Cloud Console
2. Go to the Gradient Platform and navigate to Serverless Inference.
3. Click on Create model access key, enter a name, and create the key.

Once you've done this set the `DIGITALOCEAN_INFERENCE_KEY` environment variable:
"""
logger.info("# DigitalOcean Gradient")

os.environ["DIGITALOCEAN_INFERENCE_KEY"] = "your-api-key"

"""
### Installation

The LangChain Gradient integration is in the `langchain-gradient` package:
"""
logger.info("### Installation")

pip install -qU langchain-gradient

"""
## Instantiation
"""
logger.info("## Instantiation")


llm = ChatGradient(
    model="llama3.3-70b-instruct",
    api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY")
)

"""
## Invocation
"""
logger.info("## Invocation")

messages = [
    (
        "system",
        "You are a creative storyteller. Continue any story prompt you receive in an engaging and imaginative way.",
    ),
    ("human", "Once upon a time, in a village at the edge of a mysterious forest, a young girl named Mira found a glowing stone..."),
]
ai_msg = llm.invoke(messages)
ai_msg
logger.debug(ai_msg.content)

"""
## Chaining
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a knowledgeable assistant. Carefully read the provided context and answer the user's question. If the answer is present in the context, cite the relevant sentence. If not, reply with \"Not found in context.\"",
        ),
        ("human", "Context: {context}\nQuestion: {question}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "context": (
            "The Eiffel Tower is located in Paris and was completed in 1889. "
            "It was designed by Gustave Eiffel's engineering company. "
            "The tower is one of the most recognizable structures in the world. "
            "The Statue of Liberty was a gift from France to the United States."
        ),
        "question": "Who designed the Eiffel Tower and when was it completed?"
    }
)

logger.info("\n\n[DONE]", bright=True)